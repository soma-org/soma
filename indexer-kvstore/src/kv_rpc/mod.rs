// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Archival gRPC service backed by BigTable.
//!
//! Implements `soma.rpc.LedgerService` by reading historical data from BigTable
//! via [`KeyValueStoreReader`].

use std::convert::Infallible;
use std::sync::Arc;

use rpc::proto::soma::ledger_service_server::LedgerService;
use rpc::proto::soma::{
    BatchGetObjectsRequest, BatchGetObjectsResponse, BatchGetTransactionsRequest,
    BatchGetTransactionsResponse, GetCheckpointRequest, GetCheckpointResponse, GetEpochRequest,
    GetEpochResponse, GetObjectRequest, GetObjectResponse, GetServiceInfoRequest,
    GetServiceInfoResponse, GetTransactionRequest, GetTransactionResponse,
};
use rpc_tonic::server::NamedService;
use tokio::sync::RwLock;

use crate::BigTableClient;

// Re-export tonic 0.13 types used in the LedgerService trait.
use rpc_tonic::{Request, Response, Status};

mod get_checkpoint;
mod get_epoch;
mod get_object;
mod get_service_info;
mod get_transaction;

pub use rpc::proto::soma::ledger_service_server::LedgerServiceServer;

/// Archival gRPC server backed by BigTable.
///
/// Each RPC clones the inner `BigTableClient` (cheap — tonic `Channel` handle)
/// and reads via [`crate::KeyValueStoreReader`].
#[derive(Clone)]
pub struct KvRpcServer {
    client: BigTableClient,
    chain_id: Option<String>,
    server_version: Option<String>,
    cache: Arc<RwLock<Option<GetServiceInfoResponse>>>,
}

impl KvRpcServer {
    pub fn new(client: BigTableClient) -> Self {
        Self { client, chain_id: None, server_version: None, cache: Arc::new(RwLock::new(None)) }
    }

    pub fn with_chain_id(mut self, chain_id: String) -> Self {
        self.chain_id = Some(chain_id);
        self
    }

    pub fn with_server_version(mut self, version: String) -> Self {
        self.server_version = Some(version);
        self
    }

    /// Spawn a background task that refreshes the cached `GetServiceInfoResponse`.
    pub fn spawn_cache_refresh(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let Some(server) = weak.upgrade() else {
                    break;
                };
                if let Ok(response) = get_service_info::build_response(&server).await {
                    *server.cache.write().await = Some(response);
                }
            }
        });
    }

    /// Build an axum Router with LedgerService, health, and reflection services.
    pub async fn into_router(self) -> axum::Router {
        let ledger_service = LedgerServiceServer::new(self);

        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_service_status("soma.rpc.LedgerService", tonic_health::ServingStatus::Serving)
            .await;

        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(rpc::proto::soma::FILE_DESCRIPTOR_SET)
            .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
            .build_v1()
            .expect("failed to build reflection service");

        route_service(axum::Router::new(), ledger_service)
            .merge(route_service(axum::Router::new(), health_service))
            .merge(route_service(axum::Router::new(), reflection))
            .layer(tonic_web::GrpcWebLayer::new())
    }
}

/// Route a tonic 0.13 gRPC service under `/{service_name}/{*rest}`.
fn route_service<S>(router: axum::Router, svc: S) -> axum::Router
where
    S: tower::Service<axum::extract::Request, Error = Infallible>
        + NamedService
        + Clone
        + Send
        + Sync
        + 'static,
    S::Response: axum::response::IntoResponse,
    S::Future: Send + 'static,
{
    router.route_service(&format!("/{}/{{*rest}}", S::NAME), svc)
}

#[rpc_tonic::async_trait]
impl LedgerService for KvRpcServer {
    async fn get_service_info(
        &self,
        _request: Request<GetServiceInfoRequest>,
    ) -> Result<Response<GetServiceInfoResponse>, Status> {
        // Try cached response first
        if let Some(cached) = self.cache.read().await.clone() {
            return Ok(Response::new(cached));
        }

        get_service_info::build_response(self).await.map(Response::new).map_err(Into::into)
    }

    async fn get_checkpoint(
        &self,
        request: Request<GetCheckpointRequest>,
    ) -> Result<Response<GetCheckpointResponse>, Status> {
        get_checkpoint::get_checkpoint(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }

    async fn get_transaction(
        &self,
        request: Request<GetTransactionRequest>,
    ) -> Result<Response<GetTransactionResponse>, Status> {
        get_transaction::get_transaction(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }

    async fn batch_get_transactions(
        &self,
        request: Request<BatchGetTransactionsRequest>,
    ) -> Result<Response<BatchGetTransactionsResponse>, Status> {
        get_transaction::batch_get_transactions(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }

    async fn get_object(
        &self,
        request: Request<GetObjectRequest>,
    ) -> Result<Response<GetObjectResponse>, Status> {
        get_object::get_object(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }

    async fn batch_get_objects(
        &self,
        request: Request<BatchGetObjectsRequest>,
    ) -> Result<Response<BatchGetObjectsResponse>, Status> {
        get_object::batch_get_objects(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }

    async fn get_epoch(
        &self,
        request: Request<GetEpochRequest>,
    ) -> Result<Response<GetEpochResponse>, Status> {
        get_epoch::get_epoch(self, request.into_inner())
            .await
            .map(Response::new)
            .map_err(Into::into)
    }
}
