#![doc = include_str!("README.md")]
pub(crate) mod channel_pool;
pub(crate) mod tonic_network;

/// Includes the generated protobuf/gRPC service code
mod tonic_gen {
    include!(concat!(env!("OUT_DIR"), "/soma.EncoderService.rs"));
}

use crate::types::certificate::ShardCertificate;
use crate::types::multiaddr::{Multiaddr, Protocol};
use crate::types::serialized::Serialized;
use crate::types::shard::ShardRef;
use crate::types::shard_commit::ShardCommit;
use crate::types::shard_completion_proof::ShardCompletionProof;
use crate::types::shard_endorsement::ShardEndorsement;
use crate::types::shard_input::ShardInput;
use crate::types::shard_removal::ShardRemoval;
use crate::types::shard_reveal::ShardReveal;
use crate::types::shard_slots::ShardSlots;
use crate::types::signed::Signature;
use crate::types::signed::Signed;
use crate::types::verified::Verified;
use crate::{
    crypto::keys::NetworkKeyPair,
    error::ShardResult,
    types::{context::EncoderContext, network_committee::NetworkingIndex},
};
use async_trait::async_trait;
use bytes::Bytes;
use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};
use std::{sync::Arc, time::Duration};

/// Default message timeout for each request.
// TODO: make timeout configurable and tune the default timeout based on measured network latency
pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

/// The Encoder Network client takes pre-Verified versions of the data. E.g. in the case of loading data from a database the Verified data is already
/// available so it would be redundant to deVerified + re-Verified the type. The response from a client fn is always in the form
/// of bytes due to the fact that verification of types for each specific networking implementation would be redundant and would require
/// maintaining multiple versions of the same codebase.
//TODO: to enforce type verification, it should be impossible to create a Verified type without going through verification first
#[async_trait]
pub(crate) trait EncoderNetworkClient: Send + Sync + Sized + 'static {
    /// Send shard input is used by the client to send an input to the shard.
    /// Each input has a finality proof of the transaction guranteeing that funds
    /// are secure, and the threshold signature which acts as entropy for shard selection.
    /// Shard members are verified using the threshold signature which is public, and the
    /// data which is kept secret. By keeping the data hash secret, only individuals that have
    /// the underlying data know who the shard members are which helps to limit any attacks related
    /// to censorship.
    async fn send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input: &Verified<Signed<ShardInput>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Get shard input allows any shard member to request the shard input from a  peer.
    /// This is useful in the case of a client censoring an individual shard member or
    /// non-maliciously crashing in the middle of sending inputs out. Getting the shard input
    /// allows the shard to self-heal by asking the peer for the input instead.
    async fn get_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_ref: &Verified<ShardRef>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// Get shard commit signature is used when a shard member has performed the computation
    /// and encrypted their embeddings using an encryption key. The commit contains
    /// information pertaining to the checksums of embedding data and a download location.
    /// The download location is then used to download probes, encrypted embeddings, peers embeddings etc.
    /// In the future, this commit might route to different encoders / probes.
    /// Commit certificates are produced by aggregating a quorum number of signatures.
    /// Get commit signatures is how one asks for those signatures
    async fn get_shard_commit_signature(
        &self,
        peer: NetworkingIndex,
        shard_commit: &Verified<Signed<ShardCommit>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// After a commit has been certified with quorum number of signatures, it is rebroadcast
    /// to all peers.
    async fn send_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard_commit_certificate: &Verified<ShardCertificate<Signed<ShardCommit>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// In the case that a node has not received a certified commit and it's timeout for waiting has been exhausted,
    /// the node will attempt to ask their peers for the certified commits. If no one returns with the certificates,
    /// the node broadcasts a removal signature.
    async fn batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots: &Verified<ShardSlots>,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Reveals must also be certified with quorum number of signatures from shard members.
    /// To get the signatures, the node reveals the key used to perform commit encryption.
    async fn get_shard_reveal_signature(
        &self,
        peer: NetworkingIndex,
        shard_reveal: &Verified<Signed<ShardReveal>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// Once a node has a certified reveal, they broadcast that to all their shard peers.
    async fn send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate: &Verified<ShardCertificate<Signed<ShardReveal>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// If a node has not received a reveal certificate and the timeout has been exhausted, the node
    /// attempts to ask peers for the certificate. If no one has the certificate, the peer broadcasts a removal.
    async fn batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots: &Verified<ShardSlots>,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Send shard removal signature is used when a node has exhausted all means of
    /// getting some piece of data to move forward. The node first attempts to wait in
    /// case of commit/reveal certificates or download retries in the case of encrypted embeddings
    /// and probes. Next the node attempts to resolve by contacting peers in the shard. At last,
    /// they sign a message expressing the desire to remove that piece of data.
    async fn batch_send_shard_removal_signatures(
        &self,
        peer: NetworkingIndex,
        shard_removal_signatures: &Vec<Verified<Signed<ShardRemoval>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Before sending the final endorsement, shard members batch send all removal certificates to their fellow shard members.
    /// The trigger for syncing removal certificates is after running every embedding across every probe. In the case that a shard member is
    /// missing a certificate, they adopt the new certificate and resend the batch of removal certificates. Upon receiving a quorum of batch removal certificates
    /// that match the shard members own certificates, it continues to endorsement. There is a chance of redundant computation from missing a removal certificate, but
    /// it is worth the tradeoff of eager execution vs needing to wait for full state sync.
    async fn batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates: &Vec<Verified<ShardCertificate<ShardRemoval>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Send endorsement sends the signed final scores for the encoders and embeddings to all other peers.
    /// Once a quorum number of identical endorsements (more importantly signatures) have been aggregated, the aggregate
    /// signature + endorsement can be used to finalize the embeddings.
    async fn send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement: &Verified<Signed<ShardEndorsement>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Once a node has a certified endorsement, they start a countdown to attempt to submit that data on-chain. The countdown
    /// is staggered such that there is less of a chance for redundant transactions. Redundancy does not harm the system it is just less efficient.
    /// A finality proof, stops the nodes countdown because it proves that the embedding was validly submitted on-chain.
    /// While not required for receiving any reward, honest nodes are expected to attempt delivering the final embeddings to the original client/RPC.
    /// The same staggered approach is used to minimize redundant messages. Receiving a delivery proof stops the countdown for the node. Inversely, any
    /// honest node that successfully delivers to the client, receives a signature in receipt and should notify their peers.
    async fn send_shard_completion_proof(
        &self,
        peer: NetworkingIndex,
        shard_completion_proof: &Verified<ShardCompletionProof>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

/// The network service takes bytes as an input, since these types have come over the wire they are already Verified, but verification should
/// occur inside the network service rather than the networking specific implementations due to redundant verification code for all networking protocols.
/// The return types are verified types so that serialization is non-redundant and handle in one place, where the verified type has the Verified form of the type
/// allowing the networking specific implementations. The type inside the verified type is Arc'd so the copy is relatively
/// lightweight, giving the network specific implementation access to any additional information from the type, digest, etc. It's also a way to enforce some type
/// safety on the output of each handled function.
#[async_trait]
pub(crate) trait EncoderNetworkService: Send + Sync + Sized + 'static {
    /// handle the shard input. Must verify the input: transaction proof is valid, shard member, downloaded data matches checksums, etc.
    /// Post-verification, triggers the background processes for the shard of processing the data and downloading probes
    async fn handle_send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input_bytes: Bytes,
    ) -> ShardResult<()>;

    /// Responds to a node requesting an input. The node verifies whether the requesting peer is a member of the shard, otherwise will not return the input
    /// for security reasons. May want to allow anyone to access the input this way in the future?
    async fn handle_get_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_ref_bytes: Bytes,
    ) -> ShardResult<Serialized<Signed<ShardInput>>>;

    /// Validates whether the requesting peer is a member of the shard, if so, they also check for a conflicting commit message. In the case of no
    /// previous commit and valid membership, the handler signes off on the commit using their BLS key
    async fn handle_get_shard_commit_signature(
        &self,
        peer: NetworkingIndex,
        shard_commit_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardCommit>>>>;

    /// Checks validity, then fills the slot for the commit from that shard member. After quorum number of commit certificates have been received,
    /// a countdown is triggered to wait a bit before asking peers for the missing commit certificate and then removing.
    async fn handle_send_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard_commit_certificate_bytes: Bytes,
    ) -> ShardResult<()>;

    /// validates whether a shard member, returns commits if they exist, otherwise returns empty vec. Returns empty for any commits that have a removal certificate.
    async fn handle_batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardCommit>>>>>;

    /// Checks validity of peer and message. In the case of no conflicting reveal messages from the peer, handler returns a
    /// BLS aggregate signature for the certificate
    async fn handle_get_shard_reveal_signature(
        &self,
        peer: NetworkingIndex,
        shard_reveal_bytes: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardReveal>>>>;

    /// checks validity and then if there is no existing certificate, adds the reveal its corresponding slot. After receiving a quorum
    /// number of reveals, a countdown is triggered before asking peers for the missing reveal and then proceeding to broadcast a removal.
    async fn handle_send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate_bytes: Bytes,
    ) -> ShardResult<()>;

    /// returns a certified reveal if the node has it, otherwise an empty vec. Returns empty if the node has a removal certificate.
    async fn handle_batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots_bytes: Bytes,
    ) -> ShardResult<Vec<Serialized<ShardCertificate<Signed<ShardReveal>>>>>;

    /// receives a removal signature from a peer. The removal signature is stored and checked whether quorum has been hit
    /// by the number of removal signatures.
    async fn handle_batch_send_shard_removal_signatures(
        &self,
        peer: NetworkingIndex,
        shard_removal_signatures_bytes_vector: Vec<Bytes>,
    ) -> ShardResult<()>;

    /// verifies the removal certificate and then integrates it into the shard state. Valid removal certificates always trump existing data.
    async fn handle_batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates_bytes_vector: Vec<Bytes>,
    ) -> ShardResult<()>;

    /// receives a signed endorsement from a peer. The endorsement should match the
    async fn handle_send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement_bytes: Bytes,
    ) -> ShardResult<()>;

    /// add finality proof to nodes state. If both delivery proof and finality proof exist,
    /// cancel the countdown to attempt submission on-chain.
    async fn handle_send_shard_completion_proof(
        &self,
        peer: NetworkingIndex,
        shard_completion_proof_bytes: Bytes,
    ) -> ShardResult<()>;
}

/// `EncoderNetworkManager` handles starting and stopping the network related services
/// The network manager also provides clients to other encoders in an efficient way.
pub(crate) trait EncoderNetworkManager<S>: Send + Sync
where
    S: EncoderNetworkService,
{
    /// type alias
    type Client: EncoderNetworkClient;

    /// Creates a new manager by taking an encoder context and a network keypair.
    /// The network keypair is used for TLS authentication.
    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}

/// Converts a multiaddress to a host:port string handling both TCP and UDP
pub(crate) fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Dns(hostname)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", hostname, port))
        }
        (Some(Protocol::Dns(hostname)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", hostname, port))
        }

        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}
/// Converts multiaddress to socket address
pub(crate) fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
    let mut iter = addr.iter();
    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }
        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}
