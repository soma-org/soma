use crate::{
    core::pipeline_dispatcher::ExternalDispatcher, messaging::EncoderExternalNetworkService,
    types::context::Context,
};
use async_trait::async_trait;
use bytes::Bytes;
use objects::PersistentStore;
use std::{collections::HashMap, sync::Arc};
use tokio_util::sync::CancellationToken;
use tracing::error;
use types::metadata::MetadataAPI;
use types::{
    crypto::NetworkPublicKey,
    error::{ObjectError, ShardError, ShardResult},
    shard::{DownloadLocations, DownloadLocationsV1, GetData, GetDataAPI, Shard},
    shard_crypto::verified::Verified,
};
use types::{
    shard::{verify_input, Input, InputAPI, ShardAuthToken},
    shard_verifier::ShardVerifier,
};

pub(crate) struct EncoderExternalService<D: ExternalDispatcher, P: PersistentStore> {
    context: Context,
    dispatcher: D,
    shard_verifier: Arc<ShardVerifier>,
    persistent_store: P,
}

impl<D: ExternalDispatcher, P: PersistentStore> EncoderExternalService<D, P> {
    pub(crate) fn new(
        context: Context,
        dispatcher: D,
        shard_verifier: Arc<ShardVerifier>,
        persistent_store: P,
    ) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
            persistent_store,
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
impl<D: ExternalDispatcher, P: PersistentStore> EncoderExternalNetworkService
    for EncoderExternalService<D, P>
{
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
    async fn handle_get_data(
        &self,
        peer: &NetworkPublicKey,
        get_data_bytes: Bytes,
    ) -> ShardResult<Bytes> {
        let result: ShardResult<Bytes> = {
            let get_data: GetData =
                bcs::from_bytes(&get_data_bytes).map_err(ShardError::MalformedType)?;

            let mut download_map = HashMap::new();
            for path in get_data.object_paths() {
                let download_metadata =
                    match self.persistent_store.download_metadata(path.clone()).await {
                        Ok(metadata) => Some(metadata),
                        Err(e) => match e {
                            ObjectError::NotFound(_) => None,
                            _ => return Err(ShardError::ObjectError(e)),
                        },
                    };

                match download_metadata {
                    Some(dm) => {
                        download_map.insert(dm.metadata().checksum(), dm);
                    }
                    None => {}
                }
            }

            let download_locations = DownloadLocations::V1(DownloadLocationsV1::new(download_map));
            let download_locations_bytes = bcs::to_bytes(&download_locations)
                .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
            Ok(download_locations_bytes.into())
        };

        match result {
            Ok(download_locations) => Ok(download_locations),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }
}
