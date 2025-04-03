use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderKeyPair, PeerPublicKey},
    signed::Signed,
    verified::Verified,
};
use std::sync::Arc;

use crate::{
    core::pipeline_dispatcher::ExternalDispatcher,
    error::{ShardError, ShardResult},
    messaging::EncoderExternalNetworkService,
    storage::datastore::Store,
    types::{
        context::Context,
        shard_input::{ShardInput, ShardInputAPI},
        shard_verifier::ShardVerifier,
    },
};

pub(crate) struct EncoderExternalService<D: ExternalDispatcher> {
    context: Arc<Context>,
    dispatcher: D,
    shard_verifier: ShardVerifier,
    store: Arc<dyn Store>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<D: ExternalDispatcher> EncoderExternalService<D> {
    pub(crate) fn new(
        context: Arc<Context>,
        dispatcher: D,
        shard_verifier: ShardVerifier,
        store: Arc<dyn Store>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
            store,
            encoder_keypair,
        }
    }
}
#[async_trait]
impl<D: ExternalDispatcher> EncoderExternalNetworkService for EncoderExternalService<D> {
    async fn handle_send_input(&self, peer: &PeerPublicKey, input_bytes: Bytes) -> ShardResult<()> {
        let input: Signed<ShardInput, min_sig::BLS12381Signature> =
            bcs::from_bytes(&input_bytes).map_err(ShardError::MalformedType)?;
        let (auth_token_digest, shard) = self
            .shard_verifier
            .verify(&self.context, &self.vdf, input.auth_token())
            .await?;
        let auth_token = input.auth_token().clone();
        let verified_input = Verified::new(input.clone(), input_bytes, |input| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let _ = self
            .dispatcher
            .dispatch_input(peer, auth_token, shard, verified_input)
            .await?;
        Ok(())
    }
}
