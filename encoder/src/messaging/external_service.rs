use crate::{
    core::pipeline_dispatcher::ExternalDispatcher, messaging::EncoderExternalNetworkService,
    types::context::Context,
};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::error;
use types::{
    crypto::NetworkPublicKey,
    error::{ShardError, ShardResult},
    shard::Shard,
    shard_crypto::verified::Verified,
};
use types::{
    shard::{verify_input, Input, InputAPI, ShardAuthToken},
    shard_verifier::ShardVerifier,
};

pub(crate) struct EncoderExternalService<D: ExternalDispatcher> {
    context: Context,
    dispatcher: D,
    shard_verifier: Arc<ShardVerifier>,
}

impl<D: ExternalDispatcher> EncoderExternalService<D> {
    pub(crate) fn new(context: Context, dispatcher: D, shard_verifier: Arc<ShardVerifier>) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
        }
    }

    fn shard_verification(
        &self,
        auth_token: &ShardAuthToken,
    ) -> ShardResult<(Shard, CancellationToken)> {
        // - allowed communication: allower key in tonic service
        // - shard auth token is valid: finality proof, vdf, transaction
        // - own encoder key is in the shard: shard_verification below
        let inner_context = self.context.inner();
        let committees = inner_context.committees(auth_token.epoch())?;

        let (shard, cancellation) = self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            &auth_token,
        )?;

        if !shard.contains(&self.context.own_encoder_key()) {
            return Err(ShardError::UnauthorizedPeer);
        }

        Ok((shard, cancellation))
    }
}
#[async_trait]
impl<D: ExternalDispatcher> EncoderExternalNetworkService for EncoderExternalService<D> {
    async fn handle_send_input(
        &self,
        peer: &NetworkPublicKey,
        input_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let input: Input = bcs::from_bytes(&input_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(input.auth_token())?;

            // - tls key for data matches peer (handled in corresponding type verification fn)
            let verified_input =
                Verified::new(input, input_bytes, |i| verify_input(&i, &shard, peer))
                    .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            // dispatcher handles repeated/conflicting messages from peers
            let _ = self
                .dispatcher
                .dispatch_input(shard, verified_input, cancellation)
                .await?;
            Ok(())
        };

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }
}
