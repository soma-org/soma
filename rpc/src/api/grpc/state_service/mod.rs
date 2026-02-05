use crate::api::RpcService;
use crate::proto::soma::GetBalanceRequest;
use crate::proto::soma::GetBalanceResponse;
use crate::proto::soma::GetTargetRequest;
use crate::proto::soma::GetTargetResponse;
use crate::proto::soma::ListOwnedObjectsRequest;
use crate::proto::soma::ListOwnedObjectsResponse;
use crate::proto::soma::ListTargetsRequest;
use crate::proto::soma::ListTargetsResponse;
use crate::proto::soma::state_service_server::StateService;

mod get_balance;
mod get_target;
mod list_owned_objects;
mod list_targets;

#[tonic::async_trait]
impl StateService for RpcService {
    async fn list_owned_objects(
        &self,
        request: tonic::Request<ListOwnedObjectsRequest>,
    ) -> Result<tonic::Response<ListOwnedObjectsResponse>, tonic::Status> {
        list_owned_objects::list_owned_objects(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_balance(
        &self,
        request: tonic::Request<GetBalanceRequest>,
    ) -> Result<tonic::Response<GetBalanceResponse>, tonic::Status> {
        get_balance::get_balance(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn get_target(
        &self,
        request: tonic::Request<GetTargetRequest>,
    ) -> Result<tonic::Response<GetTargetResponse>, tonic::Status> {
        get_target::get_target(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn list_targets(
        &self,
        request: tonic::Request<ListTargetsRequest>,
    ) -> Result<tonic::Response<ListTargetsResponse>, tonic::Status> {
        list_targets::list_targets(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }
}
