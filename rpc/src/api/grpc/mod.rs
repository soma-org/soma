// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use tonic::server::NamedService;
use tower::Service;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

mod ledger_service;
mod state_service;
mod subscription_service;
mod transaction_execution_service;

#[derive(Default)]
pub struct Services {
    router: axum::Router,
}

impl Services {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new service.
    pub fn add_service<S>(mut self, svc: S) -> Self
    where
        S: Service<
                axum::extract::Request,
                Response: axum::response::IntoResponse,
                Error = Infallible,
            > + NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
        S::Error: Into<BoxError> + Send,
    {
        self.router = self.router.route_service(&format!("/{}/{{*rest}}", S::NAME), svc);
        self
    }

    pub fn into_router(self) -> axum::Router {
        self.router.layer(tonic_web::GrpcWebLayer::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve a router on a random port and return the socket address.
    async fn serve_router(router: axum::Router) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        addr
    }

    #[tokio::test]
    async fn test_health_check_overall() {
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_service_status("test.Service", tonic_health::ServingStatus::Serving)
            .await;

        let router = Services::new().add_service(health_service).into_router();
        let addr = serve_router(router).await;

        let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client = tonic_health::pb::health_client::HealthClient::new(channel);

        // Empty service name checks overall server health
        let response = client
            .check(tonic_health::pb::HealthCheckRequest { service: String::new() })
            .await
            .unwrap();
        assert_eq!(
            response.into_inner().status(),
            tonic_health::pb::health_check_response::ServingStatus::Serving,
        );
    }

    #[tokio::test]
    async fn test_health_check_named_service() {
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_service_status("soma.rpc.LedgerService", tonic_health::ServingStatus::Serving)
            .await;

        let router = Services::new().add_service(health_service).into_router();
        let addr = serve_router(router).await;

        let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client = tonic_health::pb::health_client::HealthClient::new(channel);

        // Known service returns Serving
        let response = client
            .check(tonic_health::pb::HealthCheckRequest {
                service: "soma.rpc.LedgerService".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(
            response.into_inner().status(),
            tonic_health::pb::health_check_response::ServingStatus::Serving,
        );

        // Unknown service returns NotFound
        let err = client
            .check(tonic_health::pb::HealthCheckRequest { service: "nonexistent".to_string() })
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_reflection_lists_services() {
        let (_, health_service) = tonic_health::server::health_reporter();

        let reflection_v1 = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(crate::proto::soma::FILE_DESCRIPTOR_SET)
            .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
            .build_v1()
            .unwrap();

        let router =
            Services::new().add_service(health_service).add_service(reflection_v1).into_router();
        let addr = serve_router(router).await;

        let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();

        let mut client =
            tonic_reflection::pb::v1::server_reflection_client::ServerReflectionClient::new(
                channel,
            );

        // list_services uses a bidirectional stream
        let request = tonic_reflection::pb::v1::ServerReflectionRequest {
            host: String::new(),
            message_request: Some(
                tonic_reflection::pb::v1::server_reflection_request::MessageRequest::ListServices(
                    String::new(),
                ),
            ),
        };

        let response = client.server_reflection_info(tokio_stream::once(request)).await.unwrap();

        let mut stream = response.into_inner();
        let message = tokio_stream::StreamExt::next(&mut stream).await.unwrap().unwrap();

        let services = match message.message_response.unwrap() {
            tonic_reflection::pb::v1::server_reflection_response::MessageResponse::ListServicesResponse(list) => {
                list.service.into_iter().map(|s| s.name).collect::<Vec<_>>()
            }
            other => panic!("expected ListServicesResponse, got {:?}", other),
        };

        assert!(
            services.contains(&"soma.rpc.LedgerService".to_string()),
            "expected soma.rpc.LedgerService in {services:?}"
        );
        assert!(
            services.contains(&"soma.rpc.StateService".to_string()),
            "expected soma.rpc.StateService in {services:?}"
        );
        assert!(
            services.contains(&"grpc.health.v1.Health".to_string()),
            "expected grpc.health.v1.Health in {services:?}"
        );
    }
}
