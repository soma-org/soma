use std::sync::Arc;

use crate::tonic_gen::encoder_validator_api_server::EncoderValidatorApi;
use async_trait::async_trait;
use authority::state::AuthorityState;
use tracing::error_span;
use types::encoder_validator::{FetchCommitteesRequest, FetchCommitteesResponse};

#[derive(Clone)]
pub struct EncoderValidatorService {
    state: Arc<AuthorityState>,
}

impl EncoderValidatorService {
    pub fn new(state: Arc<AuthorityState>) -> Self {
        Self { state }
    }

    async fn fetch_committees_impl(
        &self,
        request: tonic::Request<FetchCommitteesRequest>,
    ) -> Result<tonic::Response<FetchCommitteesResponse>, tonic::Status> {
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        let fetch_committees_request = request.into_inner();

        let span = error_span!("fetch_committees");

        // TODO: actually fetch committees from authority state
        // let info = self
        //     .state
        //     .fetch_committees(&epoch_store, fetch_committees_request.clone())
        //     .instrument(span)
        //     .await?;

        let info = FetchCommitteesResponse {};

        Ok(tonic::Response::new(info))
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
