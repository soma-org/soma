use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::scoring::ScoringEngine;
use crate::types::ScoreRequest;

pub struct AppState {
    pub engine: Arc<ScoringEngine>,
}

pub async fn start_scoring_server(
    host: &str,
    port: u16,
    engine: Arc<ScoringEngine>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| format!("Invalid scoring service address: {e}"))?;

    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let state = Arc::new(AppState { engine });

    let app = Router::new()
        .route("/health", get(health))
        .route("/score", post(score))
        .layer(cors)
        .with_state(state);

    info!("Scoring service listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> &'static str {
    "OK"
}

async fn score(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ScoreRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.engine.score(request).await {
        Ok(response) => match serde_json::to_value(response) {
            Ok(val) => (StatusCode::OK, Json(val)),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Serialization error: {e}") })),
            ),
        },
        Err(e) => {
            let error_msg = e.to_string();
            let status = if error_msg.contains("must be")
                || error_msg.contains("required")
                || error_msg.contains("must not")
                || error_msg.contains("Checksum")
                || error_msg.contains("Invalid")
            {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(serde_json::json!({ "error": error_msg })))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn start_test_server() -> u16 {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = runtime::ModelConfig::new();
        let device = types::config::node_config::DeviceConfig::Cpu;
        let engine = Arc::new(ScoringEngine::new(dir.path(), config, &device).expect("engine"));
        let state = Arc::new(AppState { engine });

        let cors = CorsLayer::new()
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_origin(Any);

        let app = Router::new()
            .route("/health", get(health))
            .route("/score", post(score))
            .layer(cors)
            .with_state(state);

        let listener =
            tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("local addr").port();

        // Leak the tempdir so it lives long enough for the server
        std::mem::forget(dir);

        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server");
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        port
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let port = start_test_server().await;
        let resp =
            reqwest::get(format!("http://127.0.0.1:{port}/health")).await.expect("health request");
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.text().await.expect("body"), "OK");
    }

    #[tokio::test]
    async fn test_score_bad_json() {
        let port = start_test_server().await;
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/score"))
            .json(&serde_json::json!({"invalid": "request"}))
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), 422);
    }

    #[tokio::test]
    async fn test_cors_headers() {
        let port = start_test_server().await;
        let resp = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/health"))
            .header("Origin", "http://example.com")
            .send()
            .await
            .expect("cors request");
        assert_eq!(resp.status(), 200);
        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing CORS header");
        assert_eq!(allow_origin.to_str().expect("header value"), "*");
    }

    #[tokio::test]
    async fn test_score_empty_models() {
        let port = start_test_server().await;
        let body = serde_json::json!({
            "data_url": "https://example.com/data.bin",
            "data_checksum": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "data_size": 1024,
            "model_manifests": [],
            "target_embedding": [0.1, 0.2],
            "seed": 42,
        });
        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/score"))
            .json(&body)
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), 400);
        let body: serde_json::Value = resp.json().await.expect("json");
        assert!(body["error"].as_str().expect("error field").contains("required"));
    }
}
