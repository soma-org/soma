pub(crate) mod channel_pool;
pub(crate) mod encoder_tonic_service;
pub(crate) mod tonic;

mod tonic_gen {
    include!(concat!(env!("OUT_DIR"), "/soma.EncoderService.rs"));
}

use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};

use crate::types::certificate::ShardCertificate;
use crate::types::manifest::Manifest;
use crate::types::multiaddr::{Multiaddr, Protocol};
use crate::types::shard::ShardRef;
// use crate::types::shard::ShardRef;
use crate::types::shard_commit::ShardCommit;
use crate::types::shard_endorsement::ShardEndorsement;
use crate::types::shard_input::ShardInput;
use crate::types::shard_reveal::ShardReveal;
use crate::types::{signed::Signed, verified::Verified};
use crate::ProtocolKeySignature;
use crate::{
    crypto::keys::NetworkKeyPair,
    error::ShardResult,
    types::{context::EncoderContext, network_committee::NetworkingIndex},
};
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait EncoderClient: Send + Sync + Sized + 'static {
    //TODO: ensure everything going over the wire is verified and versioned

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
        input: &Verified<Signed<ShardInput>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Get shard input allows any shard member to request the shard input from a  peer.
    /// This is useful in the case of a client censoring an individual shard member or
    /// non-maliciously crashing in the middle of sending inputs out. Getting the shard input
    /// allows the shard to self-heal by asking the peer for the input instead.
    async fn get_shard_input(
        &self,
        peer: NetworkingIndex,
        input: &Verified<ShardRef>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// Send probes is only used in the the extremely slim case that a race condition happens where the node requested the probe and sent out removals
    /// but other peers have successfully received the probe AFTER the peer requested the probe from peers.
    /// In such case, peers that:
    /// - have the probe
    /// - have not received a removal certificate
    /// - have received a removal signature from the fast node
    /// use send probes as an attempt to self-heal the shard and supply the fast node with the neccessary probe so that it does not
    /// get stuck without the probe but also without a removal certificate.
    async fn send_probes(
        &self,
        peer: NetworkingIndex,
        probes: &Vec<Verified<ShardCertificate<Signed<Probe>>>>,
        timeout: Duration,
    );

    /// Get probes requests specific probes from a peer. Typical flow is to request a probe directly from
    /// the owner, but if that person is unreachable, an encoder can request it from his shard members.
    /// The flow for getting a probe is:
    /// 1. ask the probe owner directly
    /// 2. retry until timeout hits
    /// 3. ask shard peers if they have the probe
    /// 4. if no, send removal signature out
    async fn get_probes(
        &self,
        peer: NetworkingIndex,
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Get commit signature is used when a shard member has performed the computation
    /// and encrypted their embeddings using an encryption key. The commit contains
    /// information pertaining to the checksums of embedding data and download URLs.
    /// Commit certificates are produced by aggregating a quorum number of signatures.
    /// Get commit signatures is how one asks for those signatures
    async fn get_commit_signature(
        &self,
        peer: NetworkingIndex,
        commit: &Verified<Signed<ShardCommit>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// After a commit has been certified with quorum number of signatures, it is rebroadcast
    /// to all peers. Send certified commits is how that method is called. It has a backup functionality
    /// that allows it to fill in any peers that have signaled their inability to get the certificate
    /// with a removal. In the case that a node has a commit, a removal, but no removal certificate they attempt
    /// to backfill their peers.
    async fn send_certified_commits(
        &self,
        peer: NetworkingIndex,
        commits: &Vec<Verified<ShardCertificate<Signed<ShardCommit>>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// In the case that a node has not received a certified commit and it's timeout for waiting has been exhausted,
    /// the node will attempt to ask their peers for the certified commits. If no one returns with the certificates,
    /// the node broadcasts a removal signature.
    async fn get_certified_commits(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type and verify
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Reveals must also be certified with quorum number of signatures from shard members.
    /// To get the signatures, the node reveals the key used to perform commit encryption.
    async fn get_reveal_signature(
        &self,
        peer: NetworkingIndex,
        reveal: &Verified<Signed<ShardReveal>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    /// Once a node has a certified reveal, they broadcast that to all their shard peers.
    /// The endpoint can also be used to backfill peers that have signalled that they need the
    /// reveal certificate. In the case of already receiving a removal certificate, there is no need to
    /// backfill.
    async fn send_certified_reveals(
        &self,
        peer: NetworkingIndex,
        reveals: &Vec<Verified<ShardCertificate<Signed<ShardReveal>>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// If a node has not received a reveal certificate and the timeout has been exhausted, the node
    /// attempts to ask peers for the certificate. If no one has the certificate, the peer broadcasts a removal.
    async fn get_certified_reveals(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type and verify
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Send commit manifests is used to backfill any nodes that have signalled that they wish to remove the commit due to problems
    /// with downloading the data. This is only used by nodes that have received removal signatures, but not removal certificates,
    /// and were able to successfully download the data.
    async fn send_commit_manifests(
        &self,
        peer: NetworkingIndex,
        commits: &Vec<Verified<Signed<Manifest>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Get commit manifests is used to ask peers for alternative commit data manifests. Nodes use this after exhausting all retries
    /// of downloading the commit data from the author. If a peer does in fact have the commit data, they provide a download link.
    /// If no one has it, or those new links fail, the node broadcasts a removal signature.
    async fn get_commit_manifests(
        &self,
        peer: NetworkingIndex,
        //TODO: verify and fix type
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    /// Send removal signatures is used when a node has exhausted all means of
    /// getting some piece of data to move forward. The node first attempts to wait in
    /// case of commit/reveal certificates or download retries in the case of encrypted embeddings
    /// and probes. Next the node attempts to resolve by contacting peers in the shard. At last,
    /// they sign a message expressing the desire to remove that piece of data.
    async fn send_removal_signatures(
        &self,
        peer: NetworkingIndex,
        //TODO: fix type
        removals: &Vec<Verified<Signed<Removal>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// If a node receives quorum number of removals signatures, and they did not know of the removal certificate prior, they broadcast the certificate to all peers.
    /// This is likely a bit redundant, and might be removed in the future.
    async fn send_certified_removals(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type
        removals: &Vec<Verified<ShardCertificate<Removal>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// send removal set is triggered when all stages have been able to proceed, and all embeddings have been run through all probes. There may still be some inconsistencies
    /// in the shard, specifically some removal certificates that haven't percolated. Send removal set is a method of synchronizing those removals. If a new node receives any
    /// new removal certificates, they add that removal set and store the count of messages received for that removal set. When they have received a quorum number of their latest
    /// removal set, the node sends the final endorsement.
    async fn send_removal_set(
        &self,
        peer: NetworkingIndex,
        removal_set: &Verified<Signed<RemovalSet>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Send endorsement sends the signed final scores for the encoders and embeddings to all other peers.
    /// Once a quorum number of identical endorsements (more importantly signatures) have been aggregated, the aggregate
    /// signature + endorsement can be used to finalize the embeddings.
    async fn send_endorsement(
        &self,
        peer: NetworkingIndex,
        removal_set: &Verified<Signed<ShardEndorsement>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// Once a node has a certified endorsement, they start a countdown to attempt to submit that data on-chain. The countdown
    /// is staggered such that there is less of a chance for redundant transactions. Redundancy does not harm the system it is just less efficient.
    /// A finality proof, stops the nodes countdown because it proves that the embedding was validly submitted on-chain.
    async fn send_finality_proof(
        &self,
        peer: NetworkingIndex,
        finality_proof: &Verified<EmbeddingFinalityProof>,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// While not required for receiving any reward, honest nodes are expected to attempt delivering the final embeddings to the original client/RPC.
    /// The same staggered approach is used to minimize redundant messages. Receiving a delivery proof stops the countdown for the node. Inversely, any
    /// honest node that successfully delivers to the client, receives a signature in receipt and should notify their peers.
    async fn send_delivery_proof(
        &self,
        peer: NetworkingIndex,
        delivery_proof: &Verified<EmbeddingDeliveryProof>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderService: Send + Sync + Sized + 'static {
    async fn handle_send_input(&self, peer: NetworkingIndex, input: Bytes) -> ShardResult<()>;
    async fn handle_get_input(
        &self,
        peer: NetworkingIndex,
        shard_ref: Bytes,
    ) -> ShardResult<VerifiedSignedShardInput>;
    async fn handle_send_selection(
        &self,
        peer: NetworkingIndex,
        selection: Bytes,
    ) -> ShardResult<()>;
}

pub(crate) trait EncoderManager<S>: Send + Sync
where
    S: EncoderService,
{
    type Client: EncoderClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self;
    fn client(&self) -> Arc<Self::Client>;
    async fn start(&mut self, service: Arc<S>);
    async fn stop(&mut self);
}

fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
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
fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
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
