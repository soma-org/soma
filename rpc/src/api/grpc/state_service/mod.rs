// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::api::RpcService;
use crate::proto::soma::state_service_server::StateService;
use crate::proto::soma::{
    GetBalanceRequest, GetBalanceResponse,
    GetChallengeRequest, GetChallengeResponse,
    GetTargetRequest, GetTargetResponse,
    ListChallengesRequest, ListChallengesResponse,
    ListOwnedObjectsRequest, ListOwnedObjectsResponse,
    ListTargetsRequest, ListTargetsResponse,
};
mod get_balance;
mod list_owned_objects;

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
        _request: tonic::Request<GetTargetRequest>,
    ) -> Result<tonic::Response<GetTargetResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("get_target not yet implemented"))
    }

    async fn list_targets(
        &self,
        _request: tonic::Request<ListTargetsRequest>,
    ) -> Result<tonic::Response<ListTargetsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("list_targets not yet implemented"))
    }

    async fn get_challenge(
        &self,
        _request: tonic::Request<GetChallengeRequest>,
    ) -> Result<tonic::Response<GetChallengeResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("get_challenge not yet implemented"))
    }

    async fn list_challenges(
        &self,
        _request: tonic::Request<ListChallengesRequest>,
    ) -> Result<tonic::Response<ListChallengesResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("list_challenges not yet implemented"))
    }
}
