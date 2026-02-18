// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/server.rs

use crate::app_state::AppState;
use crate::types::{FaucetRequest, FaucetResponse, RequestStatus};
use axum::{Json, Router, extract::State, http::StatusCode, routing::get, routing::post};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

/// Start the faucet HTTP server with the given app state.
pub async fn start_faucet(app_state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{}:{}", app_state.config.host_ip, app_state.config.port)
        .parse()
        .map_err(|e| format!("Invalid faucet address: {e}"))?;

    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let app = Router::new()
        .route("/", get(health))
        .route("/gas", post(request_gas))
        .route("/v1/gas", post(request_gas))
        .route("/v2/gas", post(request_gas))
        .layer(cors)
        .with_state(app_state);

    info!("Faucet server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> &'static str {
    "OK"
}

async fn request_gas(
    State(state): State<Arc<AppState>>,
    Json(request): Json<FaucetRequest>,
) -> (StatusCode, Json<FaucetResponse>) {
    let recipient = match &request {
        FaucetRequest::FixedAmountRequest { recipient } => recipient.clone(),
    };

    let address = match recipient.parse::<types::base::SomaAddress>() {
        Ok(addr) => addr,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(FaucetResponse {
                    status: RequestStatus::Failure(format!("Invalid address: {e}")),
                    coins_sent: None,
                }),
            );
        }
    };

    match state.faucet.local_request_execute_tx(address).await {
        Ok(coins) => (
            StatusCode::OK,
            Json(FaucetResponse {
                status: RequestStatus::Success,
                coins_sent: Some(coins),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(FaucetResponse {
                status: RequestStatus::Failure(e.to_string()),
                coins_sent: None,
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faucet_config::FaucetConfig;
    use crate::local_faucet::LocalFaucet;
    use test_cluster::{TestCluster, TestClusterBuilder};

    /// Setup helper. Returns (port, app_state, cluster).
    /// The cluster MUST be kept alive for the duration of the test so that
    /// the network (validators + fullnode) remains reachable.
    async fn setup_faucet_server() -> (u16, Arc<AppState>, TestCluster) {
        let cluster = TestClusterBuilder::new().build().await;

        // We need to take the wallet out of the cluster while keeping the
        // cluster alive. TestCluster's wallet field is public, so we can
        // reconstruct a WalletContext from the same config path.
        let wallet_config_path = cluster.wallet.config.path().to_path_buf();
        let wallet = sdk::wallet_context::WalletContext::new(&wallet_config_path)
            .expect("Failed to re-open wallet");

        let mut config = FaucetConfig::default();
        config.port = 0; // Let OS pick a random port
        let faucet = LocalFaucet::new(wallet, config.clone())
            .await
            .expect("Failed to create faucet");

        let app_state = Arc::new(AppState {
            faucet: Arc::new(faucet),
            config,
        });

        let cors = CorsLayer::new()
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_origin(Any);

        let app = Router::new()
            .route("/", get(health))
            .route("/gas", post(request_gas))
            .route("/v1/gas", post(request_gas))
            .route("/v2/gas", post(request_gas))
            .layer(cors)
            .with_state(app_state.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind");
        let port = listener.local_addr().expect("No local addr").port();

        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("Server failed");
        });

        // Brief sleep to let the server start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        (port, app_state, cluster)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        let resp = reqwest::get(format!("http://127.0.0.1:{port}/"))
            .await
            .expect("Health request failed");

        assert_eq!(resp.status(), 200);
        let body = resp.text().await.expect("Failed to read body");
        assert_eq!(body, "OK");
    }

    #[tokio::test]
    async fn test_gas_endpoint_success() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        let recipient = types::base::SomaAddress::random();
        let body = serde_json::json!({
            "FixedAmountRequest": { "recipient": recipient.to_string() }
        });

        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/gas"))
            .json(&body)
            .send()
            .await
            .expect("Gas request failed");

        assert_eq!(resp.status(), 200);
        let faucet_resp: FaucetResponse = resp.json().await.expect("Failed to parse response");
        assert!(matches!(faucet_resp.status, RequestStatus::Success));
        assert!(faucet_resp.coins_sent.is_some());
        let coins = faucet_resp.coins_sent.expect("coins_sent was None");
        assert_eq!(coins.len(), 5);
    }

    #[tokio::test]
    async fn test_gas_endpoint_v1_and_v2_aliases() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        let recipient = types::base::SomaAddress::random();
        let body = serde_json::json!({
            "FixedAmountRequest": { "recipient": recipient.to_string() }
        });

        for path in &["/v1/gas", "/v2/gas"] {
            let resp = reqwest::Client::new()
                .post(format!("http://127.0.0.1:{port}{path}"))
                .json(&body)
                .send()
                .await
                .unwrap_or_else(|_| panic!("Request to {path} failed"));

            assert_eq!(resp.status(), 200, "Failed for path {path}");
        }
    }

    #[tokio::test]
    async fn test_gas_endpoint_invalid_json() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        // Missing the FixedAmountRequest wrapper
        let body = serde_json::json!({
            "recipient": "0x0000000000000000000000000000000000000000000000000000000000000001"
        });

        let resp = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/gas"))
            .json(&body)
            .send()
            .await
            .expect("Request failed");

        assert_eq!(resp.status(), 422, "Expected 422 for malformed JSON");
    }

    #[tokio::test]
    async fn test_cors_headers() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        let resp = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/"))
            .header("Origin", "http://example.com")
            .send()
            .await
            .expect("CORS request failed");

        assert_eq!(resp.status(), 200);
        // CORS layer should include access-control-allow-origin header
        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("Missing access-control-allow-origin header");
        assert_eq!(allow_origin.to_str().unwrap(), "*");
    }

    /// §8.6: SDK faucet client test — calls request_from_faucet via the SDK
    #[tokio::test]
    async fn test_sdk_faucet_client() {
        let (port, _state, _cluster) = setup_faucet_server().await;

        let address = types::base::SomaAddress::random();
        let url = format!("http://127.0.0.1:{port}/v2/gas");

        let response = sdk::faucet_client::request_from_faucet(address, &url)
            .await
            .expect("SDK faucet request failed");

        assert!(
            matches!(response.status, sdk::faucet_client::RequestStatus::Success),
            "Expected Success, got {:?}",
            response.status
        );
        let coins = response.coins_sent.expect("coins_sent was None");
        assert_eq!(coins.len(), 5);
        for coin in &coins {
            assert_eq!(coin.amount, 200_000_000_000);
            assert!(!coin.id.is_empty());
            assert!(!coin.transfer_tx_digest.is_empty());
        }
    }

    /// §8.6: SDK faucet client with bad URL returns error, not panic
    #[tokio::test]
    async fn test_sdk_faucet_bad_url() {
        let address = types::base::SomaAddress::random();
        let result =
            sdk::faucet_client::request_from_faucet(address, "http://127.0.0.1:1/v2/gas").await;
        assert!(result.is_err(), "Expected connection error for bad URL");
    }
}
