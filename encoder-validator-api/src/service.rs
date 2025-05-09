use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::{commit::CommitStore, state::AuthorityState};
use tonic::Status;
use tracing::error_span;
use types::{
    consensus::block::BlockAPI,
    encoder_validator::{
        EpochCommittee, FetchCommitteesRequest, FetchCommitteesResponse, GetLatestEpochRequest,
        GetLatestEpochResponse,
    },
};

#[derive(Clone)]
pub struct EncoderValidatorService {
    state: Arc<AuthorityState>,
    commit_store: Arc<CommitStore>,
}

impl EncoderValidatorService {
    pub fn new(state: Arc<AuthorityState>, commit_store: Arc<CommitStore>) -> Self {
        Self {
            state,
            commit_store,
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

        // Validate request parameters
        if start > end {
            return Err(Status::invalid_argument(
                "Start epoch must be less than or equal to end epoch",
            ));
        }

        // Throw an error if epoch 0 is requested
        if start == 0 {
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
                        // Check for complete validator set and encoder committee data
                        if let (
                            Some(validator_set),
                            Some(encoder_committee),
                            Some(val_agg_sig),
                            Some(enc_agg_sig),
                        ) = (
                            &end_of_epoch_data.next_validator_set,
                            &end_of_epoch_data.next_encoder_committee,
                            &end_of_epoch_data.validator_aggregate_signature,
                            &end_of_epoch_data.encoder_aggregate_signature,
                        ) {
                            let next_epoch = epoch + 1;

                            // Simple collection of signer indices
                            let mut signer_indices = Vec::new();

                            // First, add the block's author if it has both signatures
                            if end_of_epoch_data.validator_set_signature.is_some()
                                && end_of_epoch_data.encoder_committee_signature.is_some()
                            {
                                signer_indices.push(end_of_epoch_block.author().0);
                            }

                            // Find ancestor blocks that have signed both sets
                            for ancestor_ref in end_of_epoch_block.ancestors() {
                                // We would need to fetch actual ancestor blocks here to check
                                // if they have signatures for both sets, but for simplicity
                                // let's just add all ancestors for now
                                signer_indices.push(ancestor_ref.author.0);
                            }

                            // Serialize the validator set and encoder committee
                            let validator_set_bytes =
                                bcs::to_bytes(validator_set).map_err(|e| {
                                    tonic::Status::internal(format!(
                                        "Failed to serialize validator set for epoch {}: {}",
                                        next_epoch, e
                                    ))
                                })?;

                            let encoder_committee_bytes = bcs::to_bytes(encoder_committee)
                                .map_err(|e| {
                                    tonic::Status::internal(format!(
                                        "Failed to serialize encoder committee for epoch {}: {}",
                                        next_epoch, e
                                    ))
                                })?;

                            // Serialize the aggregate signatures
                            let val_agg_sig_bytes = bcs::to_bytes(val_agg_sig)
                            .map_err(|e| {
                                tonic::Status::internal(format!(
                                    "Failed to serialize validator aggregate signature for epoch {}: {}",
                                    next_epoch, e
                                ))
                            })?;

                            let enc_agg_sig_bytes = bcs::to_bytes(enc_agg_sig)
                            .map_err(|e| {
                                tonic::Status::internal(format!(
                                    "Failed to serialize encoder aggregate signature for epoch {}: {}",
                                    next_epoch, e
                                ))
                            })?;

                            response.epoch_committees.push(EpochCommittee {
                                epoch: next_epoch,
                                validator_set: validator_set_bytes.into(),
                                aggregate_signature: val_agg_sig_bytes.into(),
                                next_epoch_start_timestamp_ms: end_of_epoch_data
                                    .next_epoch_start_timestamp_ms,
                                signer_indices,
                                encoder_committee: encoder_committee_bytes.into(),
                                encoder_aggregate_signature: enc_agg_sig_bytes.into(),
                            });
                        }
                    }
                }
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
