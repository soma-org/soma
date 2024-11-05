//! Tonic Network contains all the code related to tonic-specific code implementing the network client, service, and manager traits.
use async_trait::async_trait;
use bytes::Bytes;
use std::{io::Read, sync::Arc, time::Duration};
use tokio::sync::oneshot;
use tonic::{transport::Server, Request, Response};
use tower_http::add_extension::AddExtensionLayer;

use crate::{
    crypto::keys::NetworkKeyPair,
    error::{ShardError, ShardResult},
    networking::messaging::{
        to_socket_addr, tonic_gen::encoder_service_server::EncoderServiceServer,
    },
    types::{
        certificate::ShardCertificate,
        context::{EncoderContext, NetworkingContext},
        network_committee::NetworkingIndex,
        shard::ShardRef,
        shard_commit::ShardCommit,
        shard_delivery_proof::ShardDeliveryProof,
        shard_endorsement::ShardEndorsement,
        shard_finality_proof::ShardFinalityProof,
        shard_input::ShardInput,
        shard_removal::ShardRemoval,
        shard_reveal::ShardReveal,
        shard_slots::ShardSlots,
    },
};
use tracing::info;

use crate::networking::messaging::tonic_gen::encoder_service_client::EncoderServiceClient;

use super::{
    channel_pool::{Channel, ChannelPool},
    tonic_gen::encoder_service_server::EncoderService,
    EncoderNetworkClient, EncoderNetworkManager, EncoderNetworkService,
};

use crate::types::{
    serialized::Serialized,
    signed::{Signature, Signed},
};

// Implements Tonic RPC client for Encoders.
pub(crate) struct EncoderTonicClient {
    /// network_keypair used for TLS
    network_keypair: NetworkKeyPair,
    /// channel pool for tonic channel reuse
    channel_pool: Arc<ChannelPool>,
}

/// Implments the core functionality of the encoder tonic client
impl EncoderTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub(crate) fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            network_keypair,
            channel_pool: Arc::new(ChannelPool::new(context)),
        }
    }

    /// returns an encoder client
    // TODO: re-introduce configuring limits to the client for safety
    async fn get_client(
        &self,
        peer: NetworkingIndex,
        timeout: Duration,
    ) -> ShardResult<EncoderServiceClient<Channel>> {
        // let config = &self.context.parameters.tonic;
        let channel = self.channel_pool.get_channel(peer, timeout).await?;
        Ok(EncoderServiceClient::new(channel))
        // .max_encoding_message_size(config.message_size_limit)
        // .max_decoding_message_size(config.message_size_limit))
    }
}

#[async_trait]
/// each function operates similarly in the sense that every request is packaged, a timeout is set
/// and a peer's client is retrieved from the channel pool.
impl EncoderNetworkClient for EncoderTonicClient {
    async fn send_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_input: &Serialized<Signed<ShardInput>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardInputRequest {
            shard_input: shard_input.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("send_shard_input failed: {e:?}")))?;
        Ok(())
    }

    async fn get_shard_input(
        &self,
        peer: NetworkingIndex,
        shard_ref: &Serialized<ShardRef>,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let mut request = Request::new(GetShardInputRequest {
            shard_ref: shard_ref.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .get_shard_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("get_shard_input failed: {e:?}")))?;

        Ok(response.into_inner().shard_input)
    }

    async fn get_shard_commit_signature(
        &self,
        peer: NetworkingIndex,
        shard_commit: &Serialized<Signed<ShardCommit>>,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let mut request = Request::new(GetShardCommitSignatureRequest {
            shard_commit: shard_commit.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .get_shard_commit_signature(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("get_shard_commit_signature failed: {e:?}"))
            })?;

        Ok(response.into_inner().shard_commit_signature)
    }

    async fn send_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard_commit_certificate: &Serialized<ShardCertificate<Signed<ShardCommit>>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardCommitCertificateRequest {
            shard_commit_certificate: shard_commit_certificate.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_commit_certificate(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("send_shard_commit_certificate failed: {e:?}"))
            })?;
        Ok(())
    }

    async fn batch_get_shard_commit_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots: &Serialized<ShardSlots>,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>> {
        let mut request = Request::new(BatchGetShardCommitCertificatesRequest {
            shard_slots: shard_slots.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .batch_get_shard_commit_certificates(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!(
                    "batch_get_shard_commit_certificates failed: {e:?}"
                ))
            })?;

        Ok(response.into_inner().shard_commit_certificates)
    }

    async fn get_shard_reveal_signature(
        &self,
        peer: NetworkingIndex,
        shard_reveal: &Serialized<Signed<ShardReveal>>,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let mut request = Request::new(GetShardRevealSignatureRequest {
            shard_reveal: shard_reveal.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .get_shard_reveal_signature(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("get_shard_reveal_signature failed: {e:?}"))
            })?;

        Ok(response.into_inner().shard_reveal_signature)
    }

    async fn send_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard_reveal_certificate: &Serialized<ShardCertificate<Signed<ShardReveal>>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardRevealCertificateRequest {
            shard_reveal_certificate: shard_reveal_certificate.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_reveal_certificate(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("send_shard_reveal_certificate failed: {e:?}"))
            })?;
        Ok(())
    }

    async fn batch_get_shard_reveal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_slots: &Serialized<ShardSlots>,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>> {
        let mut request = Request::new(BatchGetShardRevealCertificatesRequest {
            shard_slots: shard_slots.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .batch_get_shard_reveal_certificates(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!(
                    "batch_get_shard_reveal_certificates failed: {e:?}"
                ))
            })?;

        Ok(response.into_inner().shard_reveal_certificates)
    }

    async fn batch_send_shard_removal_signatures(
        &self,
        peer: NetworkingIndex,
        shard_removal_signatures: &Vec<Serialized<Signed<ShardRemoval>>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let shard_removal_signatures_bytes = shard_removal_signatures
            .iter()
            .map(|x| x.bytes())
            .collect::<Vec<_>>();

        let mut request = Request::new(BatchSendShardRemovalSignaturesRequest {
            shard_removal_signatures: shard_removal_signatures_bytes,
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .batch_send_shard_removal_signatures(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!(
                    "batch_send_shard_removal_signatures failed: {e:?}"
                ))
            })?;
        Ok(())
    }

    async fn batch_send_shard_removal_certificates(
        &self,
        peer: NetworkingIndex,
        shard_removal_certificates: &Vec<Serialized<ShardCertificate<ShardRemoval>>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let shard_removal_certificates_bytes = shard_removal_certificates
            .iter()
            .map(|x| x.bytes())
            .collect::<Vec<_>>();
        let mut request = Request::new(BatchSendShardRemovalCertificatesRequest {
            shard_removal_certificates: shard_removal_certificates_bytes,
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .batch_send_shard_removal_certificates(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!(
                    "batch_send_shard_removal_certificates failed: {e:?}"
                ))
            })?;
        Ok(())
    }

    async fn send_shard_endorsement(
        &self,
        peer: NetworkingIndex,
        shard_endorsement: &Serialized<Signed<ShardEndorsement>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardEndorsementRequest {
            shard_endorsement: shard_endorsement.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_endorsement(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("send_shard_endorsement failed: {e:?}"))
            })?;
        Ok(())
    }
    async fn send_shard_finality_proof(
        &self,
        peer: NetworkingIndex,
        shard_finality_proof: &Serialized<ShardFinalityProof>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardFinalityProofRequest {
            shard_finality_proof: shard_finality_proof.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_finality_proof(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("send_shard_finality_proof failed: {e:?}"))
            })?;
        Ok(())
    }
    async fn send_shard_delivery_proof(
        &self,
        peer: NetworkingIndex,
        shard_delivery_proof: &Serialized<ShardDeliveryProof>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendShardDeliveryProofRequest {
            shard_delivery_proof: shard_delivery_proof.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_shard_delivery_proof(request)
            .await
            .map_err(|e| {
                ShardError::NetworkRequest(format!("send_shard_delivery_proof failed: {e:?}"))
            })?;
        Ok(())
    }
}

/// Proxies Tonic requests to `NetworkService` with actual handler implementation.
struct EncoderTonicServiceProxy<S: EncoderNetworkService> {
    /// Encoder context
    context: Arc<EncoderContext>,
    /// Encoder Network Service - this is typically the same even between different networking stacks. The trait
    /// makes testing easier.
    service: Arc<S>,
}

/// Implements a new method to create an encoder tonic service proxy
impl<S: EncoderNetworkService> EncoderTonicServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(context: Arc<EncoderContext>, service: Arc<S>) -> Self {
        Self { context, service }
    }
}

/// Used to pack the networking index into each request. Using a new type
/// such that this can be extended in the future. May want to version this however?
#[derive(Clone, Debug)]
pub(crate) struct PeerInfo {
    /// networking index, verified using the TLS networking keypair
    pub(crate) network_index: NetworkingIndex,
}

#[async_trait]
impl<S: EncoderNetworkService> EncoderService for EncoderTonicServiceProxy<S> {
    async fn send_shard_input(
        &self,
        request: Request<SendShardInputRequest>,
    ) -> Result<Response<SendShardInputResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_input = request.into_inner().shard_input;

        self.service
            .handle_send_shard_input(peer_index, shard_input)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardInputResponse {}))
    }
    async fn get_shard_input(
        &self,
        request: Request<GetShardInputRequest>,
    ) -> Result<Response<GetShardInputResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_ref = request.into_inner().shard_ref;

        let service_response = self
            .service
            .handle_get_shard_input(peer_index, shard_ref)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(GetShardInputResponse {
            shard_input: service_response.bytes(),
        }))
    }

    async fn get_shard_commit_signature(
        &self,
        request: Request<GetShardCommitSignatureRequest>,
    ) -> Result<Response<GetShardCommitSignatureResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_commit = request.into_inner().shard_commit;

        let shard_commit_signature = self
            .service
            .handle_get_shard_commit_signature(peer_index, shard_commit)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(GetShardCommitSignatureResponse {
            shard_commit_signature: shard_commit_signature.bytes(),
        }))
    }

    async fn send_shard_commit_certificate(
        &self,
        request: Request<SendShardCommitCertificateRequest>,
    ) -> Result<Response<SendShardCommitCertificateResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_commit_certificate = request.into_inner().shard_commit_certificate;

        self.service
            .handle_send_shard_commit_certificate(peer_index, shard_commit_certificate)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardCommitCertificateResponse {}))
    }

    async fn batch_get_shard_commit_certificates(
        &self,
        request: Request<BatchGetShardCommitCertificatesRequest>,
    ) -> Result<Response<BatchGetShardCommitCertificatesResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_slots = request.into_inner().shard_slots;

        let shard_commit_certificates = self
            .service
            .handle_batch_get_shard_commit_certificates(peer_index, shard_slots)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        let shard_commit_certificates = shard_commit_certificates
            .iter()
            .map(|x| x.bytes())
            .collect::<Vec<_>>();

        Ok(Response::new(BatchGetShardCommitCertificatesResponse {
            shard_commit_certificates,
        }))
    }
    async fn get_shard_reveal_signature(
        &self,
        request: Request<GetShardRevealSignatureRequest>,
    ) -> Result<Response<GetShardRevealSignatureResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_reveal = request.into_inner().shard_reveal;

        let shard_reveal_signature = self
            .service
            .handle_get_shard_reveal_signature(peer_index, shard_reveal)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(GetShardRevealSignatureResponse {
            shard_reveal_signature: shard_reveal_signature.bytes(),
        }))
    }
    async fn send_shard_reveal_certificate(
        &self,
        request: Request<SendShardRevealCertificateRequest>,
    ) -> Result<Response<SendShardRevealCertificateResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_reveal_certificate = request.into_inner().shard_reveal_certificate;

        self.service
            .handle_send_shard_reveal_certificate(peer_index, shard_reveal_certificate)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardRevealCertificateResponse {}))
    }

    async fn batch_get_shard_reveal_certificates(
        &self,
        request: Request<BatchGetShardRevealCertificatesRequest>,
    ) -> Result<Response<BatchGetShardRevealCertificatesResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_slots = request.into_inner().shard_slots;

        let shard_reveal_certificates = self
            .service
            .handle_batch_get_shard_reveal_certificates(peer_index, shard_slots)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;
        let shard_reveal_certificates = shard_reveal_certificates
            .iter()
            .map(|x| x.bytes())
            .collect::<Vec<_>>();

        Ok(Response::new(BatchGetShardRevealCertificatesResponse {
            shard_reveal_certificates,
        }))
    }
    async fn batch_send_shard_removal_signatures(
        &self,
        request: Request<BatchSendShardRemovalSignaturesRequest>,
    ) -> Result<Response<BatchSendShardRemovalSignaturesResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_removal_signatures = request.into_inner().shard_removal_signatures;

        self.service
            .handle_batch_send_shard_removal_signatures(peer_index, shard_removal_signatures)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(BatchSendShardRemovalSignaturesResponse {}))
    }
    async fn batch_send_shard_removal_certificates(
        &self,
        request: Request<BatchSendShardRemovalCertificatesRequest>,
    ) -> Result<Response<BatchSendShardRemovalCertificatesResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_removal_certificates = request.into_inner().shard_removal_certificates;

        self.service
            .handle_batch_send_shard_removal_certificates(peer_index, shard_removal_certificates)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(BatchSendShardRemovalCertificatesResponse {}))
    }
    async fn send_shard_endorsement(
        &self,
        request: Request<SendShardEndorsementRequest>,
    ) -> Result<Response<SendShardEndorsementResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_endorsement = request.into_inner().shard_endorsement;

        self.service
            .handle_send_shard_endorsement(peer_index, shard_endorsement)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardEndorsementResponse {}))
    }
    async fn send_shard_finality_proof(
        &self,
        request: Request<SendShardFinalityProofRequest>,
    ) -> Result<Response<SendShardFinalityProofResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_finality_proof = request.into_inner().shard_finality_proof;

        self.service
            .handle_send_shard_finality_proof(peer_index, shard_finality_proof)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardFinalityProofResponse {}))
    }
    async fn send_shard_delivery_proof(
        &self,
        request: Request<SendShardDeliveryProofRequest>,
    ) -> Result<Response<SendShardDeliveryProofResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let shard_delivery_proof = request.into_inner().shard_delivery_proof;

        self.service
            .handle_send_shard_delivery_proof(peer_index, shard_delivery_proof)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendShardDeliveryProofResponse {}))
    }
}

/// Tonic specific manager type that contains a tonic specific client and
/// the oneshot tokio channel to trigger service shutdown.
pub struct EncoderTonicManager {
    context: Arc<EncoderContext>,
    client: Arc<EncoderTonicClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Implementation of the encoder tonic manager that contains a new fn to create the type
// TODO: switch this to type state pattern
impl EncoderTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            context: context.clone(),
            client: Arc::new(EncoderTonicClient::new(context, network_keypair)),
            shutdown_tx: None,
        }
    }
}

impl<S: EncoderNetworkService> EncoderNetworkManager<S> for EncoderTonicManager {
    type Client = EncoderTonicClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self::new(context, network_keypair)
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    /// if the network is running locally, then it uses the localhost address, otherwise
    /// it uses the zero address since it will be used in a hosted context where the service will
    /// be routed to using the IP address. The function starts a gRPC server taking a shutdown channel
    /// to allow the system to trigger shutdown from outside of the spawned tokio task.
    async fn start(&mut self, service: Arc<S>) {
        let network_identity = self
            .context
            .network_committee
            .identity(self.context.own_network_index);
        let own_network_index = self.context.own_network_index;
        // By default, bind to the unspecified address to allow the actual address to be assigned.
        // But bind to localhost if it is requested.
        let own_address = if network_identity.address.is_localhost_ip() {
            network_identity.address.clone()
        } else {
            network_identity.address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();
        let svc =
            EncoderServiceServer::new(EncoderTonicServiceProxy::new(self.context.clone(), service));
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        tokio::spawn(async move {
            // let leader_service = Server::builder().add_service(LeaderShardServiceServer::new(svc));

            let tower_layer = tower::ServiceBuilder::new()
                .layer(AddExtensionLayer::new(PeerInfo {
                    network_index: own_network_index,
                }))
                .into_inner();

            Server::builder()
                .layer(tower_layer)
                .add_service(svc)
                .serve_with_shutdown(own_address, async {
                    rx.await.ok();
                })
                .await
                .unwrap();
        });
        info!("Binding tonic server to address {:?}", own_address);
    }

    async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}
// ////////////////////////////////////////////////////////////////////

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardInputRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_input: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardInputResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct GetShardInputRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_ref: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct GetShardInputResponse {
    #[prost(bytes = "bytes", tag = "1")]
    shard_input: Bytes,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct GetShardCommitSignatureRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_commit: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct GetShardCommitSignatureResponse {
    #[prost(bytes = "bytes", tag = "1")]
    shard_commit_signature: Bytes,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendShardCommitCertificateRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_commit_certificate: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardCommitCertificateResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct BatchGetShardCommitCertificatesRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_slots: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct BatchGetShardCommitCertificatesResponse {
    #[prost(bytes = "bytes", repeated, tag = "1")]
    shard_commit_certificates: Vec<Bytes>,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct GetShardRevealSignatureRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_reveal: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct GetShardRevealSignatureResponse {
    #[prost(bytes = "bytes", tag = "1")]
    shard_reveal_signature: Bytes,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendShardRevealCertificateRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_reveal_certificate: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardRevealCertificateResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct BatchGetShardRevealCertificatesRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_slots: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct BatchGetShardRevealCertificatesResponse {
    #[prost(bytes = "bytes", repeated, tag = "1")]
    shard_reveal_certificates: Vec<Bytes>,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct BatchSendShardRemovalSignaturesRequest {
    #[prost(bytes = "bytes", repeated, tag = "1")]
    shard_removal_signatures: Vec<Bytes>,
}

#[derive(Clone, prost::Message)]
pub(crate) struct BatchSendShardRemovalSignaturesResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct BatchSendShardRemovalCertificatesRequest {
    #[prost(bytes = "bytes", repeated, tag = "1")]
    shard_removal_certificates: Vec<Bytes>,
}

#[derive(Clone, prost::Message)]
pub(crate) struct BatchSendShardRemovalCertificatesResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendShardEndorsementRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_endorsement: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardEndorsementResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendShardFinalityProofRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_finality_proof: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardFinalityProofResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendShardDeliveryProofRequest {
    #[prost(bytes = "bytes", tag = "1")]
    shard_delivery_proof: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendShardDeliveryProofResponse {}
