use crate::{
    core::pipeline_dispatcher::ExternalDispatcher, messaging::EncoderExternalNetworkService,
    types::context::Context,
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::PeerPublicKey,
    error::{ShardError, ShardResult},
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use types::{
    shard::{verify_input, Input, InputAPI, ShardAuthToken},
    shard_verifier::ShardVerifier,
};

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
    fn shard_verification(
        &self,
        auth_token: &ShardAuthToken,
    ) -> ShardResult<(Shard, CancellationToken)> {
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
    async fn handle_send_input(&self, peer: &PeerPublicKey, input_bytes: Bytes) -> ShardResult<()> {
        // TODO: need to adjust this to lookup the encoder and do encoder validation on the input
        // expecting a staked full node to speak to the encoders rather than arbitrary peers
        // We must also verify the correct signature and should look up the staked full node
        // and pass that in as the peer.

        let result: ShardResult<()> = {
            let input: Input = bcs::from_bytes(&input_bytes).map_err(ShardError::MalformedType)?;

            // should check that the sender is the correct person to be sending?
            let (shard, cancellation) = self.shard_verification(input.auth_token())?;

            // TODO: fix the verification to actually work
            let verified_input = Verified::new(input, input_bytes, |i| verify_input(&i, &shard))
                .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            let own_encoder_key = &self.context.own_encoder_key();

            // TODO: this is clunky come back and fix perhaps just initializing the input pipeline with its own object peer and address?
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
