// Faucet client logic derived from MystenLabs/sui (Apache-2.0)
// See: https://github.com/MystenLabs/sui/blob/main/crates/sui/src/client_commands.rs

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use faucet::app_state::AppState;
use faucet::faucet_config::{FaucetConfig, DEFAULT_AMOUNT, DEFAULT_FAUCET_PORT, DEFAULT_NUM_COINS};
use faucet::faucet_gen::faucet_client::FaucetClient;
use faucet::faucet_types::GasRequest;
use faucet::local_faucet::LocalFaucet;
use faucet::server::start_faucet;
use sdk::wallet_context::{WalletContext, create_wallet_context};
use soma_keys::key_identity::KeyIdentity;
use tracing::info;
use types::config::soma_config_dir;

use crate::response::format_soma_public;

pub async fn execute_request(
    context: &mut WalletContext,
    address: Option<KeyIdentity>,
    url: Option<String>,
) -> Result<()> {
    let address = context.get_identity_address(address)?;
    let url = match url {
        Some(url) => url,
        None => find_faucet_url(context)?,
    };

    println!("Requesting tokens from faucet at {url} for address {address}...");

    let mut client = FaucetClient::connect(url).await?;
    let response =
        client.request_gas(GasRequest { recipient: address.to_string() }).await?.into_inner();

    if response.status == "Success" {
        println!("Successfully received {} coins:", response.coins_sent.len());
        for coin in &response.coins_sent {
            println!(
                "  - {} (id: {}, tx: {})",
                format_soma_public(coin.amount as u128),
                coin.id,
                coin.transfer_tx_digest
            );
        }
    } else {
        anyhow::bail!("Faucet request failed: {}", response.status);
    }

    Ok(())
}

pub async fn execute_start(
    port: u16,
    host: String,
    amount: u64,
    num_coins: usize,
    config_dir: Option<PathBuf>,
) -> Result<()> {
    let config = FaucetConfig {
        port,
        host_ip: host.clone(),
        amount,
        num_coins,
        wallet_client_timeout_secs: 60,
        config_dir: config_dir.clone(),
    };

    let resolved_config_dir = match config_dir {
        Some(dir) => dir,
        None => soma_config_dir()?,
    };

    info!("Starting faucet with config dir: {:?}", resolved_config_dir);

    let wallet = create_wallet_context(config.wallet_client_timeout_secs, resolved_config_dir)?;

    let faucet = LocalFaucet::new(wallet, config.clone())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize faucet: {e}"))?;

    let app_state = Arc::new(AppState { faucet: Arc::new(faucet), config });

    let display_host = if host == "0.0.0.0" { "127.0.0.1" } else { &host };
    println!("Faucet gRPC server listening on {display_host}:{port}");

    start_faucet(app_state).await.map_err(|e| anyhow::anyhow!("Faucet server error: {e}"))?;

    Ok(())
}

/// Map the active network environment's RPC URL to a faucet URL.
fn find_faucet_url(context: &WalletContext) -> Result<String> {
    let env = context.get_active_env().map_err(|_| {
        anyhow::anyhow!(
            "No active network environment configured. \
             Please set one with `soma env` or provide --url explicitly."
        )
    })?;

    faucet_url_for_rpc(&env.rpc, &env.alias)
}

/// Derive the faucet gRPC URL from a known network RPC URL.
fn faucet_url_for_rpc(rpc: &str, alias: &str) -> Result<String> {
    // Local network
    if rpc.contains("127.0.0.1") || rpc.contains("0.0.0.0") || rpc.contains("localhost") {
        return Ok("http://127.0.0.1:9123".to_string());
    }

    // Testnet
    if rpc.contains("testnet") {
        return Ok("https://faucet.testnet.soma.network".to_string());
    }

    anyhow::bail!(
        "Could not determine faucet URL for network environment '{alias}' (RPC: {rpc}). \
         Please provide a faucet URL explicitly with --url."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_faucet_url_localnet() {
        let result = faucet_url_for_rpc("http://127.0.0.1:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123");

        let result = faucet_url_for_rpc("http://0.0.0.0:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123");

        let result = faucet_url_for_rpc("http://localhost:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123");
    }

    #[test]
    fn test_find_faucet_url_testnet() {
        let result =
            faucet_url_for_rpc("https://fullnode.testnet.soma.org:443", "testnet").unwrap();
        assert_eq!(result, "https://faucet.testnet.soma.network");
    }
}
