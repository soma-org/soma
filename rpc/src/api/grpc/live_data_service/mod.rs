use crate::api::RpcService;
use crate::proto::soma::GetBalanceRequest;
use crate::proto::soma::GetBalanceResponse;
use crate::proto::soma::ListOwnedObjectsRequest;
use crate::proto::soma::ListOwnedObjectsResponse;
use crate::proto::soma::SimulateTransactionRequest;
use crate::proto::soma::SimulateTransactionResponse;
use crate::proto::soma::live_data_service_server::LiveDataService;

mod get_balance;
mod list_owned_objects;
mod simulate;

#[tonic::async_trait]
impl LiveDataService for RpcService {
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

    async fn simulate_transaction(
        &self,
        request: tonic::Request<SimulateTransactionRequest>,
    ) -> Result<tonic::Response<SimulateTransactionResponse>, tonic::Status> {
        simulate::simulate_transaction(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }
}
