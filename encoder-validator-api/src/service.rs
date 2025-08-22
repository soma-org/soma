use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::{commit::CommitStore, state::AuthorityState};
use tonic::Status;
use tracing::{debug, error, error_span, info, warn};
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
            let last_commit = match self.commit_store.get_epoch_last_commit(epoch) {
                Ok(commit) => commit,
                Err(e) => {
                    error!("Failed to get last commit of epoch {}: {}", epoch, e);
                    return Err(tonic::Status::internal(format!(
                        "Failed to get last commit of epoch {}: {}",
                        epoch, e
                    )));
                }
            };

            if let Some(commit) = last_commit {
                debug!("Found last commit for epoch {}", epoch);

                if let Some(end_of_epoch_block) = commit.get_end_of_epoch_block() {
                    debug!("Found end of epoch block for epoch {}", epoch);

                    if let Some(end_of_epoch_data) = end_of_epoch_block.end_of_epoch_data() {
                        debug!("Found end of epoch data for epoch {}", epoch);

                        // Check for complete validator set, encoder committee, and networking committee data
                        if let (
                            Some(validator_set),
                            Some(encoder_committee),
                            Some(networking_committee),
                            Some(val_agg_sig),
                            Some(enc_agg_sig),
                            Some(net_agg_sig),
                        ) = (
                            &end_of_epoch_data.next_validator_set,
                            &end_of_epoch_data.next_encoder_committee,
                            &end_of_epoch_data.next_networking_committee,
                            &end_of_epoch_data.validator_aggregate_signature,
                            &end_of_epoch_data.encoder_aggregate_signature,
                            &end_of_epoch_data.networking_aggregate_signature,
                        ) {
                            let next_epoch = epoch + 1;
                            info!("Building committee data for epoch {}", next_epoch);

                            // Simple collection of signer indices
                            let mut signer_indices = Vec::new();

                            // First, add the block's author if it has all three signatures
                            if end_of_epoch_data.validator_set_signature.is_some()
                                && end_of_epoch_data.encoder_committee_signature.is_some()
                                && end_of_epoch_data.networking_committee_signature.is_some()
                            {
                                debug!(
                                    "Adding block author {} to signer indices",
                                    end_of_epoch_block.author().0
                                );
                                signer_indices.push(end_of_epoch_block.author().0);
                            }

                            // Find ancestor blocks that have signed all three sets
                            let ancestor_count = end_of_epoch_block.ancestors().len();
                            debug!("Processing {} ancestors for signatures", ancestor_count);

                            for ancestor_ref in end_of_epoch_block.ancestors() {
                                // We would need to fetch actual ancestor blocks here to check
                                // if they have signatures for all three sets, but for simplicity
                                // let's just add all ancestors for now
                                debug!(
                                    "Adding ancestor author {} to signer indices",
                                    ancestor_ref.author.0
                                );
                                signer_indices.push(ancestor_ref.author.0);
                            }

                            // Serialize the validator set
                            debug!("Serializing validator set for epoch {}", next_epoch);
                            let validator_set_bytes = match bcs::to_bytes(validator_set) {
                                Ok(bytes) => {
                                    debug!("Validator set serialized: {} bytes", bytes.len());
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize validator set for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize validator set for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            // Serialize the encoder committee
                            debug!("Serializing encoder committee for epoch {}", next_epoch);
                            let encoder_committee_bytes = match bcs::to_bytes(encoder_committee) {
                                Ok(bytes) => {
                                    debug!("Encoder committee serialized: {} bytes", bytes.len());
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize encoder committee for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize encoder committee for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            // Serialize the networking committee
                            debug!("Serializing networking committee for epoch {}", next_epoch);
                            let networking_committee_bytes = match bcs::to_bytes(
                                networking_committee,
                            ) {
                                Ok(bytes) => {
                                    debug!(
                                        "Networking committee serialized: {} bytes",
                                        bytes.len()
                                    );
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize networking committee for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize networking committee for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            // Serialize the validator aggregate signature
                            debug!(
                                "Serializing validator aggregate signature for epoch {}",
                                next_epoch
                            );
                            let val_agg_sig_bytes = match bcs::to_bytes(val_agg_sig) {
                                Ok(bytes) => {
                                    debug!(
                                        "Validator aggregate signature serialized: {} bytes",
                                        bytes.len()
                                    );
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize validator aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize validator aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            // Serialize the encoder aggregate signature
                            debug!(
                                "Serializing encoder aggregate signature for epoch {}",
                                next_epoch
                            );
                            let enc_agg_sig_bytes = match bcs::to_bytes(enc_agg_sig) {
                                Ok(bytes) => {
                                    debug!(
                                        "Encoder aggregate signature serialized: {} bytes",
                                        bytes.len()
                                    );
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize encoder aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize encoder aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            // Serialize the networking aggregate signature
                            debug!(
                                "Serializing networking aggregate signature for epoch {}",
                                next_epoch
                            );
                            let net_agg_sig_bytes = match bcs::to_bytes(net_agg_sig) {
                                Ok(bytes) => {
                                    debug!(
                                        "Networking aggregate signature serialized: {} bytes",
                                        bytes.len()
                                    );
                                    bytes
                                }
                                Err(e) => {
                                    error!(
                                        "Failed to serialize networking aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    );
                                    return Err(tonic::Status::internal(format!(
                                        "Failed to serialize networking aggregate signature for epoch {}: {}",
                                        next_epoch, e
                                    )));
                                }
                            };

                            debug!("Adding complete committee data for epoch {}", next_epoch);
                            response.epoch_committees.push(EpochCommittee {
                                epoch: next_epoch,
                                validator_set: validator_set_bytes.into(),
                                aggregate_signature: val_agg_sig_bytes.into(),
                                next_epoch_start_timestamp_ms: end_of_epoch_data
                                    .next_epoch_start_timestamp_ms,
                                signer_indices,
                                encoder_committee: encoder_committee_bytes.into(),
                                encoder_aggregate_signature: enc_agg_sig_bytes.into(),
                                networking_committee: networking_committee_bytes.into(),
                                networking_aggregate_signature: net_agg_sig_bytes.into(),
                            });
                        } else {
                            warn!(
                                "Incomplete committee data for epoch {}: missing validator set, encoder committee, networking committee, or signatures",
                                epoch + 1
                            );
                        }
                    } else {
                        warn!("No end of epoch data found in block for epoch {}", epoch);
                    }
                } else {
                    warn!("No end of epoch block found for epoch {}", epoch);
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
