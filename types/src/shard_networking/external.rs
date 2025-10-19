use std::{sync::Arc, time::Duration};

use crate::{
    crypto::NetworkKeyPair,
    error::{ShardError, ShardResult},
    parameters::TonicParameters,
    shard_crypto::keys::EncoderPublicKey,
};
use async_trait::async_trait;
use bytes::Bytes;
use tonic::{codec::CompressionEncoding, Request};

use crate::{
    shard::Input,
    shard_networking::{
        channel_pool::{Channel, ChannelPool},
        generated::encoder_external_tonic_service_client::EncoderExternalTonicServiceClient,
        EncoderNetworkingInfo,
    },
};

#[async_trait]
pub trait EncoderExternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Input,
        timeout: Duration,
    ) -> ShardResult<()>;
}

// Implements Tonic RPC client for Encoders.
pub struct EncoderExternalTonicClient {
    pub networking_info: EncoderNetworkingInfo,
    own_peer_keypair: NetworkKeyPair,
    parameters: Arc<TonicParameters>,
    channel_pool: Arc<ChannelPool>,
}

impl EncoderExternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub fn new(
        networking_info: EncoderNetworkingInfo,
        own_peer_keypair: NetworkKeyPair,
        parameters: Arc<TonicParameters>,
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
        let config = &self.parameters;
        if let Some((peer_public_key, address)) = self.networking_info.encoder_to_tls(encoder) {
            let channel = self
                .channel_pool
                .get_channel(
                    &address,
                    peer_public_key,
                    &self.parameters,
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
        input: &Input,
        timeout: Duration,
    ) -> ShardResult<()> {
        let input_bytes = bcs::to_bytes(input).expect("Could not serialize Input");
        let mut request = Request::new(SendInputRequest {
            input: Bytes::copy_from_slice(&input_bytes),
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
