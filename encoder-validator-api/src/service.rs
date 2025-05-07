use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::commit::CommitStore;
use tracing::error_span;
use types::{
    consensus::block::BlockAPI,
    encoder_validator::{EpochCommittee, FetchCommitteesRequest, FetchCommitteesResponse},
};

#[derive(Clone)]
pub struct EncoderValidatorService {
    commit_store: Arc<CommitStore>,
}

impl EncoderValidatorService {
    pub fn new(commit_store: Arc<CommitStore>) -> Self {
        Self { commit_store }
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

        // Validate request parameters
        if start > end {
            return Err(tonic::Status::invalid_argument(
                "Start epoch must be less than or equal to end epoch",
            ));
        }

        // Throw an error if epoch 0 is requested
        if start == 0 {
            return Err(tonic::Status::invalid_argument(
                "Epoch 0 (genesis) committee should not be requested as it must be provided via configuration",
            ));
        }

        let mut response = FetchCommitteesResponse {
            epoch_committees: Vec::new(),
        };

        // We need to look at epochs [start-1, end-1] to get validator sets for epochs [start, end]
        let search_start = start - 1;
        let search_end = end - 1;

        // Iterate through epochs that contain the validator sets for our target epochs
        for epoch in search_start..=search_end {
            // Get last commit of the epoch
            let last_commit = self
                .commit_store
                .get_epoch_last_commit(epoch)
                .map_err(|e| {
                    tonic::Status::internal(format!(
                        "Failed to get last commit of epoch {}: {}",
                        epoch, e
                    ))
                })?;

            if let Some(commit) = last_commit {
                if let Some(end_of_epoch_block) = commit.get_end_of_epoch_block() {
                    if let Some(end_of_epoch_data) = end_of_epoch_block.end_of_epoch_data() {
                        if let (Some(validator_set), Some(aggregate_signature)) = (
                            &end_of_epoch_data.next_validator_set,
                            &end_of_epoch_data.aggregate_signature,
                        ) {
                            let next_epoch = epoch + 1;

                            // Simple collection of signer indices
                            let mut signer_indices = Vec::new();

                            // First, add the block's author if it has a validator_set_signature
                            if end_of_epoch_data.validator_set_signature.is_some() {
                                signer_indices.push(end_of_epoch_block.author().0);
                            }

                            let mut ancestors: Vec<_> = end_of_epoch_block
                                .ancestors()
                                .iter()
                                .map(|b| b.author.0)
                                .collect();

                            signer_indices.append(&mut ancestors);

                            // The validator set from epoch N's last commit is for epoch N+1
                            let validator_set_bytes =
                                bcs::to_bytes(validator_set).map_err(|e| {
                                    tonic::Status::internal(format!(
                                        "Failed to serialize validator set for epoch {}: {}",
                                        next_epoch, e
                                    ))
                                })?;

                            let aggregate_signature_bytes = bcs::to_bytes(aggregate_signature)
                                .map_err(|e| {
                                    tonic::Status::internal(format!(
                                        "Failed to serialize aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    ))
                                })?;

                            response.epoch_committees.push(EpochCommittee {
                                epoch: next_epoch,
                                validator_set: validator_set_bytes.into(),
                                aggregate_signature: aggregate_signature_bytes.into(),
                                next_epoch_start_timestamp_ms: end_of_epoch_data
                                    .next_epoch_start_timestamp_ms,
                                signer_indices,
                            });
                        }
                    }
                }
            }
        }

        // Check if we found all requested epochs
        let found_epochs: std::collections::HashSet<_> = response
            .epoch_committees
            .iter()
            .map(|ec| ec.epoch)
            .collect();

        for epoch in start..=end {
            if !found_epochs.contains(&epoch) {
                return Err(tonic::Status::not_found(format!(
                    "Could not find committee information for epoch {}",
                    epoch
                )));
            }
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
}
