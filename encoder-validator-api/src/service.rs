use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::{authority::AuthorityState, checkpoints::CheckpointStore};
use tonic::Status;
use tracing::{debug, error, error_span, info, warn};
use types::{
    consensus::block::BlockAPI,
    encoder_validator::{
        FetchCommitteesRequest, FetchCommitteesResponse, GetLatestEpochRequest,
        GetLatestEpochResponse,
    },
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
    async fn fetch_committees_impl(
        &self,
        request: tonic::Request<FetchCommitteesRequest>,
    ) -> Result<tonic::Response<FetchCommitteesResponse>, tonic::Status> {
        let fetch_committees_request = request.into_inner();
        let start = fetch_committees_request.start;
        let end = fetch_committees_request.end;

        let span = error_span!("fetch_committees", start, end);
        let _guard = span.enter();

        info!(
            "Starting fetch_committees request for epochs {} to {}",
            start, end
        );

        // Validate request parameters
        if start > end {
            error!(
                "Invalid request: start epoch ({}) is greater than end epoch ({})",
                start, end
            );
            return Err(Status::invalid_argument(
                "Start epoch must be less than or equal to end epoch",
            ));
        }

        // Throw an error if epoch 0 is requested
        if start == 0 {
            error!("Invalid request: epoch 0 (genesis) was requested");
            return Err(Status::invalid_argument(
                "Epoch 0 (genesis) committee should not be requested as it must be provided via configuration",
            ));
        }

        let mut response = FetchCommitteesResponse {
            epoch_committees: Vec::new(),
        };

        // We need to look at epochs [start-1, end-1] to get validator sets for epochs [start, end]
        let search_start = start - 1;
        let search_end = end - 1;

        info!(
            "Searching epochs {} to {} for committee data",
            search_start, search_end
        );

        // Iterate through epochs that contain the validator sets for our target epochs
        for epoch in search_start..=search_end {
            debug!("Processing epoch {}", epoch);

            // Get last commit of the epoch
            debug!("Fetching last commit for epoch {}", epoch);
            let last_checkpoint = match self.checkpoint_store.get_epoch_last_checkpoint(epoch) {
                Ok(checkpoint) => checkpoint,
                Err(e) => {
                    error!("Failed to get last commit of epoch {}: {}", epoch, e);
                    return Err(tonic::Status::internal(format!(
                        "Failed to get last commit of epoch {}: {}",
                        epoch, e
                    )));
                }
            };

            if let Some(checkpoint) = last_checkpoint {
                debug!("Found last commit for epoch {}", epoch);

                if checkpoint.end_of_epoch_data.is_some() {
                    debug!("Found end of epoch data for epoch {}", epoch);

                    response.epoch_committees.push(checkpoint.into());
                } else {
                    warn!("No end of epoch data found in block for epoch {}", epoch);
                }
            } else {
                warn!("No last commit found for epoch {}", epoch);
            }
        }

        info!(
            "Completed fetch_committees request: returning data for {} epochs",
            response.epoch_committees.len()
        );
        if response.epoch_committees.is_empty() {
            warn!(
                "No committee data found for any requested epochs ({} to {})",
                start, end
            );
        } else {
            debug!(
                "Returning committee data for epochs: {:?}",
                response
                    .epoch_committees
                    .iter()
                    .map(|c| c.epoch)
                    .collect::<Vec<_>>()
            );
        }

        Ok(tonic::Response::new(response))
    }
}

#[async_trait]
impl EncoderValidatorApi for EncoderValidatorService {
    async fn fetch_committees(
        &self,
        request: tonic::Request<FetchCommitteesRequest>,
    ) -> Result<tonic::Response<FetchCommitteesResponse>, tonic::Status> {
        self.fetch_committees_impl(request).await
    }

    async fn get_latest_epoch(
        &self,
        _request: tonic::Request<GetLatestEpochRequest>,
    ) -> Result<tonic::Response<GetLatestEpochResponse>, tonic::Status> {
        let current_epoch = self.state.load_epoch_store_one_call_per_task().epoch();

        Ok(tonic::Response::new(GetLatestEpochResponse {
            epoch: current_epoch,
        }))
    }
}
