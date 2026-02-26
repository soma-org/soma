// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "server")]
use std::sync::Arc;

#[cfg(feature = "server")]
use types::{
    config::rpc_config::RpcConfig, storage::read_store::RpcStateReader,
    transaction_executor::TransactionExecutor,
};

#[cfg(feature = "server")]
use crate::api::{reader::StateReader, subscription::SubscriptionServiceHandle};
pub mod client;
pub mod error;
#[cfg(feature = "server")]
mod grpc;
#[cfg(feature = "server")]
mod reader;
#[cfg(feature = "server")]
mod response;
pub(crate) mod rpc_client;
#[cfg(feature = "server")]
pub mod subscription;

#[derive(Clone)]
pub struct ServerVersion {
    pub bin: &'static str,
    pub version: &'static str,
}

impl ServerVersion {
    pub fn new(bin: &'static str, version: &'static str) -> Self {
        Self { bin, version }
    }
}

impl std::fmt::Display for ServerVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.bin)?;
        f.write_str("/")?;
        f.write_str(self.version)
    }
}

#[cfg(feature = "server")]
#[derive(Clone)]
pub struct RpcService {
    reader: StateReader,
    chain_id: types::digests::ChainIdentifier,
    subscription_service_handle: Option<SubscriptionServiceHandle>,

    server_version: Option<ServerVersion>,
    config: RpcConfig,
    executor: Option<Arc<dyn TransactionExecutor>>,
}

#[cfg(feature = "server")]
impl RpcService {
    pub fn new(reader: Arc<dyn RpcStateReader>) -> Self {
        let chain_id = reader.get_chain_identifier().unwrap();
        Self {
            reader: StateReader::new(reader),
            executor: None,
            subscription_service_handle: None,
            chain_id,
            server_version: None,
            config: RpcConfig::default(),
        }
    }

    pub fn with_server_version(&mut self, server_version: ServerVersion) -> &mut Self {
        self.server_version = Some(server_version);
        self
    }

    pub fn with_executor(&mut self, executor: Arc<dyn TransactionExecutor + Send + Sync>) {
        self.executor = Some(executor);
    }

    pub fn with_config(&mut self, config: RpcConfig) {
        self.config = config;
    }

    pub fn with_subscription_service(
        &mut self,
        subscription_service_handle: SubscriptionServiceHandle,
    ) {
        self.subscription_service_handle = Some(subscription_service_handle);
    }

    pub fn chain_id(&self) -> types::digests::ChainIdentifier {
        self.chain_id
    }

    pub fn server_version(&self) -> Option<&ServerVersion> {
        self.server_version.as_ref()
    }

    pub async fn into_router(self) -> axum::Router {
        let router = {
            let ledger_service =
                crate::proto::soma::ledger_service_server::LedgerServiceServer::new(self.clone());
            let transaction_execution_service = crate::proto::soma::transaction_execution_service_server::TransactionExecutionServiceServer::new(self.clone());
            let state_service =
                crate::proto::soma::state_service_server::StateServiceServer::new(self.clone())
                    .send_compressed(tonic::codec::CompressionEncoding::Zstd);

            fn service_name<S: tonic::server::NamedService>(_service: &S) -> &'static str {
                S::NAME
            }

            // gRPC Health Check service
            let (health_reporter, health_service) = tonic_health::server::health_reporter();
            for name in [
                service_name(&ledger_service),
                service_name(&transaction_execution_service),
                service_name(&state_service),
            ] {
                health_reporter
                    .set_service_status(name, tonic_health::ServingStatus::Serving)
                    .await;
            }

            // gRPC Server Reflection (v1 + v1alpha)
            let reflection_v1 = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(
                    crate::proto::google::protobuf::FILE_DESCRIPTOR_SET,
                )
                .register_encoded_file_descriptor_set(
                    crate::proto::google::rpc::FILE_DESCRIPTOR_SET,
                )
                .register_encoded_file_descriptor_set(crate::proto::soma::FILE_DESCRIPTOR_SET)
                .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
                .build_v1()
                .unwrap();

            let reflection_v1alpha = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(
                    crate::proto::google::protobuf::FILE_DESCRIPTOR_SET,
                )
                .register_encoded_file_descriptor_set(
                    crate::proto::google::rpc::FILE_DESCRIPTOR_SET,
                )
                .register_encoded_file_descriptor_set(crate::proto::soma::FILE_DESCRIPTOR_SET)
                .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
                .build_v1alpha()
                .unwrap();

            let mut services = grpc::Services::new()
                .add_service(ledger_service)
                .add_service(transaction_execution_service)
                .add_service(state_service)
                .add_service(health_service)
                .add_service(reflection_v1)
                .add_service(reflection_v1alpha);

            if self.subscription_service_handle.is_some() {
                let subscription_service =
                    crate::proto::soma::subscription_service_server::SubscriptionServiceServer::new(
                        self.clone(),
                    );
                health_reporter
                    .set_service_status(
                        service_name(&subscription_service),
                        tonic_health::ServingStatus::Serving,
                    )
                    .await;

                services = services.add_service(subscription_service);
            }

            services.into_router()
        };

        router.layer(axum::middleware::map_response_with_state(self, response::append_info_headers))
    }

    pub async fn start_service(self, socket_address: std::net::SocketAddr) {
        let listener = tokio::net::TcpListener::bind(socket_address).await.unwrap();
        axum::serve(listener, self.into_router().await).await.unwrap();
    }
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Ascending,
    Descending,
}

impl Direction {
    pub fn is_descending(self) -> bool {
        matches!(self, Self::Descending)
    }
}
