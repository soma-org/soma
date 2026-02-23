// Faucet client logic derived from MystenLabs/sui (Apache-2.0)
// See: https://github.com/MystenLabs/sui/blob/main/crates/sui/src/client_commands.rs

use anyhow::Result;
use faucet::faucet_gen::faucet_client::FaucetClient;
use faucet::faucet_types::GasRequest;
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;

use crate::response::format_soma_public;

/// Execute the `soma client faucet` command.
///
/// Requests test tokens from a faucet gRPC server for the given address.
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

    // Devnet
    if rpc.contains("devnet") {
        return Ok("https://faucet.devnet.soma.network".to_string());
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
    fn test_find_faucet_url_devnet() {
        let result = faucet_url_for_rpc("https://fullnode.devnet.soma.org:443", "devnet").unwrap();
        assert_eq!(result, "https://faucet.devnet.soma.network");
    }

    #[test]
    fn test_find_faucet_url_testnet() {
        let result =
            faucet_url_for_rpc("https://fullnode.testnet.soma.org:443", "testnet").unwrap();
        assert_eq!(result, "https://faucet.testnet.soma.network");
    }

    #[test]
    fn test_find_faucet_url_unknown() {
        let result = faucet_url_for_rpc("https://fullnode.mainnet.soma.org:443", "mainnet");
        assert!(result.is_err(), "Expected error for mainnet faucet URL");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Could not determine faucet URL"), "Unexpected error: {err}");
    }
}
