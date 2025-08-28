use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderPublicKey, PeerKeyPair},
    error::{ShardError, ShardResult},
    parameters::Parameters,
    signed::Signed,
    verified::Verified,
};
use tonic::{codec::CompressionEncoding, Request};

use crate::{
    shard::ShardInput,
    shard_networking::{
        channel_pool::{Channel, ChannelPool},
        generated::encoder_external_tonic_service_client::EncoderExternalTonicServiceClient,
        NetworkingInfo,
    },
};

#[async_trait]
pub trait EncoderExternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

// Implements Tonic RPC client for Encoders.
pub struct EncoderExternalTonicClient {
    pub networking_info: NetworkingInfo,
    own_peer_keypair: PeerKeyPair,
    parameters: Arc<Parameters>,
    channel_pool: Arc<ChannelPool>,
}
impl EncoderExternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub fn new(
        networking_info: NetworkingInfo,
        own_peer_keypair: PeerKeyPair,
        parameters: Arc<Parameters>,
        capacity: usize,
    ) -> Self {
        Self {
            networking_info,
            own_peer_keypair,
            parameters,
            channel_pool: Arc::new(ChannelPool::new(capacity)),
        }
    }

    /// returns an encoder client
    // TODO: re-introduce configuring limits to the client for safety
    pub async fn get_client(
        &self,
        encoder: &EncoderPublicKey,
        timeout: Duration,
    ) -> ShardResult<EncoderExternalTonicServiceClient<Channel>> {
        let config = &self.parameters.tonic;
        if let Some((address, peer_public_key)) = self.networking_info.lookup(encoder) {
            let channel = self
                .channel_pool
                .get_channel(
                    &address,
                    peer_public_key,
                    &self.parameters.tonic,
                    self.own_peer_keypair.clone(),
                    timeout,
                )
                .await?;
            let mut client = EncoderExternalTonicServiceClient::new(channel)
                .max_encoding_message_size(config.message_size_limit)
                .max_decoding_message_size(config.message_size_limit);

            client = client
                .send_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Zstd);
            Ok(client)
        } else {
            Err(ShardError::NetworkClientConnection(
                "failed to get networking info for peer".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EncoderExternalNetworkClient for EncoderExternalTonicClient {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendInputRequest {
            input: input.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(encoder, timeout)
            .await?
            .send_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
}

#[derive(Clone, prost::Message)]
pub struct SendInputRequest {
    #[prost(bytes = "bytes", tag = "1")]
    pub input: Bytes,
}

#[derive(Clone, prost::Message)]
pub struct SendInputResponse {}
