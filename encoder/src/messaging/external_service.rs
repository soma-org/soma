use crate::{
    core::pipeline_dispatcher::ExternalDispatcher,
    error::{ShardError, ShardResult},
    messaging::EncoderExternalNetworkService,
    types::{
        context::Context,
        shard::ShardRole,
        shard_input::{ShardInput, ShardInputAPI},
        shard_verifier::ShardVerifier,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{crypto::keys::PeerPublicKey, signed::Signed, verified::Verified};
use std::sync::Arc;

pub(crate) struct EncoderExternalService<D: ExternalDispatcher> {
    context: Arc<Context>,
    dispatcher: D,
    shard_verifier: ShardVerifier,
}

impl<D: ExternalDispatcher> EncoderExternalService<D> {
    pub(crate) fn new(context: Arc<Context>, dispatcher: D, shard_verifier: ShardVerifier) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
        }
    }
}
#[async_trait]
impl<D: ExternalDispatcher> EncoderExternalNetworkService for EncoderExternalService<D> {
    async fn handle_send_input(
        &self,
        _peer: &PeerPublicKey,
        input_bytes: Bytes,
    ) -> ShardResult<()> {
        let input: Signed<ShardInput, min_sig::BLS12381Signature> =
            bcs::from_bytes(&input_bytes).map_err(ShardError::MalformedType)?;
        let (own_role, _shard) = self
            .shard_verifier
            .verify(&self.context, input.auth_token())
            .await?;

        match own_role {
            ShardRole::Inference(_own_role) => {
                let _verified_input = Verified::new(input.clone(), input_bytes, |_input| Ok(()))
                    .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

                // TODO: reimplement this
                // let _ = self
                //     .dispatcher
                //     .dispatch_input(peer, auth_token, shard, verified_input)
                //     .await?;
                Ok(())
            }
            _ => Err(ShardError::FailedTypeVerification(
                "send commit should only be sent to evaluation encoders".to_string(),
            )),
        }
    }
}
