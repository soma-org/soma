use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::{authority::AuthorityState, checkpoints::CheckpointStore};
use tonic::Status;
use tracing::{debug, error, info, warn};
use types::encoder_validator::{
    FetchCommitteesRequest, FetchCommitteesResponse, GetLatestEpochRequest, GetLatestEpochResponse,
};

#[derive(Clone)]
pub struct EncoderValidatorService {
    state: Arc<AuthorityState>,
    checkpoint_store: Arc<CheckpointStore>,
}

impl EncoderValidatorService {
    pub fn new(state: Arc<AuthorityState>, checkpoint_store: Arc<CheckpointStore>) -> Self {
        Self {
            state,
            checkpoint_store,
        }
    }
}

#[async_trait]
impl EncoderValidatorApi for EncoderValidatorService {
    async fn fetch_committees(
        &self,
        request: tonic::Request<FetchCommitteesRequest>,
    ) -> Result<tonic::Response<FetchCommitteesResponse>, tonic::Status> {
        let req = request.into_inner();

        if req.start > req.end {
            return Err(Status::invalid_argument("start must be <= end"));
        }

        if req.start == 0 {
            return Err(Status::invalid_argument(
                "epoch 0 must be configured, not fetched",
            ));
        }

        info!(
            "Fetching end-of-epoch checkpoints for epochs {} to {}",
            req.start, req.end
        );

        let mut checkpoints = Vec::new();

        // For each requested epoch N, we need the checkpoint from epoch N-1
        // (since that checkpoint contains the committees for epoch N)
        for target_epoch in req.start..=req.end {
            let checkpoint_epoch = target_epoch - 1;

            match self
                .checkpoint_store
                .get_epoch_last_checkpoint(checkpoint_epoch)
            {
                Ok(Some(summary)) => {
                    if summary.end_of_epoch_data.is_some() {
                        checkpoints.push(summary.into());
                    } else {
                        warn!(
                            "Checkpoint for epoch {} has no end_of_epoch_data",
                            checkpoint_epoch
                        );
                    }
                }
                Ok(None) => {
                    debug!("No checkpoint found for epoch {}", checkpoint_epoch);
                }
                Err(e) => {
                    error!(
                        "Error fetching checkpoint for epoch {}: {}",
                        checkpoint_epoch, e
                    );
                }
            }
        }

        info!("Returning {} certified checkpoints", checkpoints.len());
        Ok(tonic::Response::new(FetchCommitteesResponse {
            epoch_committees: checkpoints,
        }))
    }

    async fn get_latest_epoch(
        &self,
        _request: tonic::Request<GetLatestEpochRequest>,
    ) -> Result<tonic::Response<GetLatestEpochResponse>, tonic::Status> {
        let epoch = self.state.load_epoch_store_one_call_per_task().epoch();
        Ok(tonic::Response::new(GetLatestEpochResponse { epoch }))
    }
}
