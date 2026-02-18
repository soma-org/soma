// Faucet request protocol follows the convention established by MystenLabs/sui (Apache-2.0)
// See: https://github.com/MystenLabs/sui/blob/main/crates/sui-faucet/src/types.rs

use serde::{Deserialize, Serialize};
use types::base::SomaAddress;

pub const DEFAULT_FAUCET_URL: &str = "http://127.0.0.1:9123/v2/gas";

#[derive(Serialize)]
enum FaucetRequest {
    FixedAmountRequest { recipient: String },
}

#[derive(Deserialize, Debug, Clone)]
pub struct FaucetResponse {
    pub status: RequestStatus,
    pub coins_sent: Option<Vec<CoinInfo>>,
}

#[derive(Deserialize, Debug, Clone)]
pub enum RequestStatus {
    Success,
    Failure(String),
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoinInfo {
    pub amount: u64,
    pub id: String,
    pub transfer_tx_digest: String,
}

/// Request test tokens from a faucet server.
///
/// Sends a `FixedAmountRequest` to the given faucet URL and returns the
/// response containing the coins sent.
pub async fn request_from_faucet(
    address: SomaAddress,
    faucet_url: &str,
) -> anyhow::Result<FaucetResponse> {
    let body = FaucetRequest::FixedAmountRequest { recipient: address.to_string() };

    let resp = reqwest::Client::new()
        .post(faucet_url)
        .json(&body)
        .send()
        .await?;

    let status_code = resp.status();

    if status_code == reqwest::StatusCode::TOO_MANY_REQUESTS {
        anyhow::bail!("Faucet rate limit exceeded (429). Please wait and try again.");
    }

    if status_code == reqwest::StatusCode::SERVICE_UNAVAILABLE {
        anyhow::bail!("Faucet service is temporarily unavailable (503). Please try again later.");
    }

    if status_code.is_client_error() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Faucet request failed with status {status_code}: {body}");
    }

    let faucet_resp: FaucetResponse = resp.json().await?;
    Ok(faucet_resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sdk_faucet_bad_url() {
        let address = SomaAddress::ZERO;
        let result = request_from_faucet(address, "http://127.0.0.1:1/v2/gas").await;
        assert!(result.is_err(), "Expected connection error for bad URL");
    }
}
