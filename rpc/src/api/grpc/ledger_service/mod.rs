use crate::api::RpcService;
use crate::proto::soma::BatchGetObjectsRequest;
use crate::proto::soma::BatchGetObjectsResponse;
use crate::proto::soma::BatchGetTransactionsRequest;
use crate::proto::soma::BatchGetTransactionsResponse;
use crate::proto::soma::GetCheckpointRequest;
use crate::proto::soma::GetCheckpointResponse;
use crate::proto::soma::GetClaimableEscrowsRequest;
use crate::proto::soma::GetClaimableEscrowsResponse;
use crate::proto::soma::GetClaimableRewardsRequest;
use crate::proto::soma::GetClaimableRewardsResponse;
use crate::proto::soma::GetEpochRequest;
use crate::proto::soma::GetEpochResponse;
use crate::proto::soma::GetObjectRequest;
use crate::proto::soma::GetObjectResponse;
use crate::proto::soma::GetServiceInfoRequest;
use crate::proto::soma::GetServiceInfoResponse;
use crate::proto::soma::GetShardsByEncoderRequest;
use crate::proto::soma::GetShardsByEncoderResponse;
use crate::proto::soma::GetShardsByEpochRequest;
use crate::proto::soma::GetShardsByEpochResponse;
use crate::proto::soma::GetShardsBySubmitterRequest;
use crate::proto::soma::GetShardsBySubmitterResponse;
use crate::proto::soma::GetTransactionRequest;
use crate::proto::soma::GetTransactionResponse;
use crate::proto::soma::GetValidTargetsRequest;
use crate::proto::soma::GetValidTargetsResponse;
use crate::proto::soma::ledger_service_server::LedgerService;

mod get_checkpoint;
mod get_epoch;
mod get_object;
mod get_service_info;
mod get_shard;
mod get_target;
mod get_transaction;

#[tonic::async_trait]
impl LedgerService for RpcService {
    async fn get_service_info(
        &self,
        _request: tonic::Request<GetServiceInfoRequest>,
    ) -> Result<tonic::Response<GetServiceInfoResponse>, tonic::Status> {
        get_service_info::get_service_info(self)
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_object(
        &self,
        request: tonic::Request<GetObjectRequest>,
    ) -> Result<tonic::Response<GetObjectResponse>, tonic::Status> {
        get_object::get_object(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn batch_get_objects(
        &self,
        request: tonic::Request<BatchGetObjectsRequest>,
    ) -> Result<tonic::Response<BatchGetObjectsResponse>, tonic::Status> {
        get_object::batch_get_objects(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_transaction(
        &self,
        request: tonic::Request<GetTransactionRequest>,
    ) -> Result<tonic::Response<GetTransactionResponse>, tonic::Status> {
        get_transaction::get_transaction(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn batch_get_transactions(
        &self,
        request: tonic::Request<BatchGetTransactionsRequest>,
    ) -> Result<tonic::Response<BatchGetTransactionsResponse>, tonic::Status> {
        get_transaction::batch_get_transactions(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_checkpoint(
        &self,
        request: tonic::Request<GetCheckpointRequest>,
    ) -> Result<tonic::Response<GetCheckpointResponse>, tonic::Status> {
        get_checkpoint::get_checkpoint(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_epoch(
        &self,
        request: tonic::Request<GetEpochRequest>,
    ) -> Result<tonic::Response<GetEpochResponse>, tonic::Status> {
        get_epoch::get_epoch(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    // =========================================================================
    // SHARD QUERIES
    // =========================================================================

    async fn get_shards_by_epoch(
        &self,
        request: tonic::Request<GetShardsByEpochRequest>,
    ) -> Result<tonic::Response<GetShardsByEpochResponse>, tonic::Status> {
        get_shard::get_shards_by_epoch(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_shards_by_submitter(
        &self,
        request: tonic::Request<GetShardsBySubmitterRequest>,
    ) -> Result<tonic::Response<GetShardsBySubmitterResponse>, tonic::Status> {
        get_shard::get_shards_by_submitter(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_shards_by_encoder(
        &self,
        request: tonic::Request<GetShardsByEncoderRequest>,
    ) -> Result<tonic::Response<GetShardsByEncoderResponse>, tonic::Status> {
        get_shard::get_shards_by_encoder(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_claimable_escrows(
        &self,
        request: tonic::Request<GetClaimableEscrowsRequest>,
    ) -> Result<tonic::Response<GetClaimableEscrowsResponse>, tonic::Status> {
        get_shard::get_claimable_escrows(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    // =========================================================================
    // TARGET QUERIES
    // =========================================================================

    async fn get_valid_targets(
        &self,
        request: tonic::Request<GetValidTargetsRequest>,
    ) -> Result<tonic::Response<GetValidTargetsResponse>, tonic::Status> {
        get_target::get_valid_targets(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_claimable_rewards(
        &self,
        request: tonic::Request<GetClaimableRewardsRequest>,
    ) -> Result<tonic::Response<GetClaimableRewardsResponse>, tonic::Status> {
        get_target::get_claimable_rewards(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }
}
