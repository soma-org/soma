// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::info;

use crate::scoring::ScoringEngine;
use crate::tonic_gen::scoring_server::{Scoring, ScoringServer};
use crate::types::{HealthRequest, HealthResponse, ScoreRequest, ScoreResponse};

pub struct ScoringService {
    engine: Arc<ScoringEngine>,
}

impl ScoringService {
    pub fn new(engine: Arc<ScoringEngine>) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl Scoring for ScoringService {
    async fn score(
        &self,
        request: Request<ScoreRequest>,
    ) -> Result<Response<ScoreResponse>, Status> {
        let request = request.into_inner();
        match self.engine.score(request).await {
            Ok(response) => Ok(Response::new(response)),
            Err(e) => {
                let error_msg = e.to_string();
                let code = if error_msg.contains("must be")
                    || error_msg.contains("required")
                    || error_msg.contains("must not")
                    || error_msg.contains("Checksum")
                    || error_msg.contains("Invalid")
                {
                    tonic::Code::InvalidArgument
                } else {
                    tonic::Code::Internal
                };
                Err(Status::new(code, error_msg))
            }
        }
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { ok: true }))
    }
}

pub async fn start_scoring_server(
    host: &str,
    port: u16,
    engine: Arc<ScoringEngine>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| format!("Invalid scoring service address: {e}"))?;

    let svc = ScoringService::new(engine);

    info!("Scoring service listening on {}", addr);

    tonic::transport::Server::builder().add_service(ScoringServer::new(svc)).serve(addr).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn start_test_server() -> u16 {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = runtime::ModelConfig::new();
        let device = types::config::node_config::DeviceConfig::Cpu;
        let engine = Arc::new(ScoringEngine::new(dir.path(), config, &device).expect("engine"));
        let svc = ScoringService::new(engine);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("local addr").port();

        // Leak the tempdir so it lives long enough for the server
        std::mem::forget(dir);

        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(ScoringServer::new(svc))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .expect("server");
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        port
    }

    #[tokio::test]
    async fn test_health() {
        let port = start_test_server().await;
        let mut client = crate::tonic_gen::scoring_client::ScoringClient::connect(format!(
            "http://127.0.0.1:{port}"
        ))
        .await
        .expect("connect");
        let resp = client.health(HealthRequest {}).await.expect("health");
        assert!(resp.into_inner().ok);
    }

    #[tokio::test]
    async fn test_score_empty_models() {
        let port = start_test_server().await;
        let mut client = crate::tonic_gen::scoring_client::ScoringClient::connect(format!(
            "http://127.0.0.1:{port}"
        ))
        .await
        .expect("connect");
        let request = ScoreRequest {
            data_url: "https://example.com/data.bin".to_string(),
            data_checksum: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_string(),
            data_size: 1024,
            model_manifests: vec![],
            target_embedding: vec![0.1, 0.2],
            seed: 42,
        };
        let resp = client.score(request).await;
        assert!(resp.is_err());
        let status = resp.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("required"));
    }
}
