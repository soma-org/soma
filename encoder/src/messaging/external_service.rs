use crate::{
    core::pipeline_dispatcher::ExternalDispatcher,
    messaging::EncoderExternalNetworkService,
    types::{
        context::Context,
        input::{Input, InputAPI},
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::PeerPublicKey,
    error::{ShardError, ShardResult},
    signed::Signed,
    verified::Verified,
};
use std::sync::Arc;
use types::shard_verifier::ShardVerifier;

pub(crate) struct EncoderExternalService<D: ExternalDispatcher> {
    context: Arc<Context>,
    dispatcher: D,
    shard_verifier: Arc<ShardVerifier>,
}

impl<D: ExternalDispatcher> EncoderExternalService<D> {
    pub(crate) fn new(
        context: Arc<Context>,
        dispatcher: D,
        shard_verifier: Arc<ShardVerifier>,
    ) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
        }
    }
}
#[async_trait]
impl<D: ExternalDispatcher> EncoderExternalNetworkService for EncoderExternalService<D> {
    async fn handle_send_input(&self, peer: &PeerPublicKey, input_bytes: Bytes) -> ShardResult<()> {
        let input: Signed<Input, min_sig::BLS12381Signature> = match bcs::from_bytes(&input_bytes) {
            Ok(i) => i,
            Err(e) => {
                return Err(ShardError::MalformedType(e));
            }
        };

        let inner_context = self.context.inner();

        tracing::debug!(
            "Getting committees for epoch: {}",
            input.auth_token().epoch()
        );
        let committees = match inner_context.committees(input.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            input.auth_token(),
        )?;

        let verified_input = Verified::new(input.clone(), input_bytes, |_input| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let own_encoder_key = &self.context.own_encoder_key();

        if let Some((own_object_peer, own_object_address)) =
            self.context.inner().object_server(own_encoder_key)
        {
            let _ = self
                .dispatcher
                .dispatch_input(
                    shard,
                    verified_input,
                    own_object_peer,
                    own_object_address,
                    cancellation,
                )
                .await?;
        };
        Ok(())
    }
}
