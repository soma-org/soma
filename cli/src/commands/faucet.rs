// Faucet client logic derived from MystenLabs/sui (Apache-2.0)
// See: https://github.com/MystenLabs/sui/blob/main/crates/sui/src/client_commands.rs

use anyhow::{Result, bail};
use sdk::faucet_client::{self, RequestStatus};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::base::SomaAddress;

use crate::response::format_soma_public;

/// Execute the `soma client faucet` command.
///
/// Requests test tokens from a faucet server for the given address.
/// If no address is provided, uses the active address. If no URL is
/// provided, derives it from the active network environment.
pub async fn execute(
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

    let response = faucet_client::request_from_faucet(address, &url).await?;

    match response.status {
        RequestStatus::Success => {
            if let Some(coins) = response.coins_sent {
                println!("Successfully received {} coins:", coins.len());
                for coin in &coins {
                    println!(
                        "  - {} (id: {}, tx: {})",
                        format_soma_public(coin.amount as u128),
                        coin.id,
                        coin.transfer_tx_digest
                    );
                }
            } else {
                println!("Request succeeded (no coin details returned).");
            }
        }
        RequestStatus::Failure(msg) => {
            bail!("Faucet request failed: {msg}");
        }
    }

    Ok(())
}

/// Map the active network environment's RPC URL to a faucet URL.
fn find_faucet_url(context: &WalletContext) -> Result<String> {
    let env = context.get_active_env().map_err(|_| {
        anyhow::anyhow!(
            "No active network environment configured. \
             Please set one with `soma client env` or provide --url explicitly."
        )
    })?;

    faucet_url_for_rpc(&env.rpc, &env.alias)
}

/// Derive the faucet URL from a known network RPC URL.
fn faucet_url_for_rpc(rpc: &str, alias: &str) -> Result<String> {
    // Local network
    if rpc.contains("127.0.0.1") || rpc.contains("0.0.0.0") || rpc.contains("localhost") {
        return Ok("http://127.0.0.1:9123/gas".to_string());
    }

    // Devnet
    if rpc.contains("devnet") {
        return Ok("https://faucet.devnet.soma.network/gas".to_string());
    }

    // Testnet
    if rpc.contains("testnet") {
        return Ok("https://faucet.testnet.soma.network/gas".to_string());
    }

    bail!(
        "Could not determine faucet URL for network environment '{alias}' (RPC: {rpc}). \
         Please provide a faucet URL explicitly with --url."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.5: Test find_faucet_url mapping for localnet
    #[test]
    fn test_find_faucet_url_localnet() {
        let result = faucet_url_for_rpc("http://127.0.0.1:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123/gas");

        let result = faucet_url_for_rpc("http://0.0.0.0:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123/gas");

        let result = faucet_url_for_rpc("http://localhost:9000", "local").unwrap();
        assert_eq!(result, "http://127.0.0.1:9123/gas");
    }

    /// §8.5: Test find_faucet_url mapping for devnet
    #[test]
    fn test_find_faucet_url_devnet() {
        let result =
            faucet_url_for_rpc("https://fullnode.devnet.soma.org:443", "devnet").unwrap();
        assert_eq!(result, "https://faucet.devnet.soma.network/gas");
    }

    /// §8.5: Test find_faucet_url mapping for testnet
    #[test]
    fn test_find_faucet_url_testnet() {
        let result =
            faucet_url_for_rpc("https://fullnode.testnet.soma.org:443", "testnet").unwrap();
        assert_eq!(result, "https://faucet.testnet.soma.network/gas");
    }

    /// §8.5: Test that mainnet/unknown RPC returns error
    #[test]
    fn test_find_faucet_url_unknown() {
        let result = faucet_url_for_rpc("https://fullnode.mainnet.soma.org:443", "mainnet");
        assert!(result.is_err(), "Expected error for mainnet faucet URL");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Could not determine faucet URL"), "Unexpected error: {err}");
    }
}
