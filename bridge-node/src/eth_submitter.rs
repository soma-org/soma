//! End-to-end Eth-side tx submitter — wallet + provider + send.
//!
//! Owns the operator's signing wallet and an `alloy` HTTP provider
//! that combines them. Given a [`crate::outbound_relayer::OutboundWithdrawal`],
//! the submitter:
//!   1. Builds the ABI calldata via [`crate::eth_abi::encode_release_calldata`].
//!   2. Wraps it in an EIP-1559 `TransactionRequest` (chain id, nonce,
//!      gas — all picked up automatically by alloy when fields are
//!      omitted).
//!   3. Signs with the operator's wallet (`PrivateKeySigner`).
//!   4. Submits via `eth_sendRawTransaction` on the configured RPC
//!      endpoint.
//!   5. (Optionally) waits for the receipt + asserts execution success.
//!
//! Sui-parity note: Sui's `crates/sui-bridge/src/eth_transaction_builder.rs`
//! does the same with the same alloy version (1.x). The provider/wallet
//! plumbing here is the same pattern; what differs is the calldata is
//! Soma-specific (`transferBridgedTokensWithSignatures` on `SomaBridge`,
//! not `transferAttestedBridgedTokens` on `SuiBridge`).

use std::sync::Arc;

use alloy::network::{EthereumWallet, TransactionBuilder};
use alloy::primitives::{Address, B256, Bytes, TxHash, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::rpc::types::TransactionRequest;
use tracing::{info, instrument, warn};

use crate::error::{BridgeError, BridgeResult};
use crate::eth_abi;
use crate::eth_wallet::EthWallet;
use crate::outbound_relayer::OutboundWithdrawal;

/// Tx submitter: holds a signed-provider that can both build/sign EIP-1559
/// txs (using the wallet) and submit them (via `eth_sendRawTransaction`).
///
/// Construction is async because `ProviderBuilder::connect_http` reads
/// the chain id from the endpoint to pick a `Network` impl; we capture
/// it once at startup so we don't issue an extra RPC per submission.
#[derive(Clone)]
pub struct EthSubmitter {
    /// Address of the `SomaBridge` contract — the `to` of every
    /// release tx.
    bridge_contract: Address,
    /// The signer + provider, ready to `send_transaction`. `Arc` so
    /// cloning the submitter doesn't reconnect the HTTP pool.
    provider: Arc<dyn Provider + Send + Sync>,
    /// Wallet's own address — used for logging + nonce fetch.
    wallet_address: Address,
}

impl std::fmt::Debug for EthSubmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EthSubmitter")
            .field("bridge_contract", &self.bridge_contract)
            .field("wallet_address", &self.wallet_address)
            .finish()
    }
}

impl EthSubmitter {
    /// Build a submitter from an RPC URL, a bridge-contract address,
    /// and an operator wallet. The wallet's address is captured up
    /// front for logging; the actual signing happens inside alloy at
    /// every `send_transaction`.
    pub fn new(rpc_url: &str, bridge_contract: Address, wallet: EthWallet) -> BridgeResult<Self> {
        let wallet_address = wallet.address();
        let alloy_wallet: EthereumWallet = wallet.into_alloy_wallet();
        let url = rpc_url
            .parse()
            .map_err(|e| BridgeError::ConfigError(format!("Bad Eth RPC URL: {e}")))?;
        let provider = ProviderBuilder::new().wallet(alloy_wallet).connect_http(url);
        Ok(Self { bridge_contract, provider: Arc::new(provider), wallet_address })
    }

    /// Submit a fully signed `transferBridgedTokensWithSignatures` tx
    /// for the given withdrawal cert. Returns the tx hash; the caller
    /// is free to poll for the receipt or fire-and-forget.
    ///
    /// alloy's `send_transaction` does the heavy lifting:
    ///   - fills missing fields (nonce, gas, EIP-1559 fee params)
    ///     from the connected provider
    ///   - signs with the wallet attached at provider construction
    ///   - submits via `eth_sendRawTransaction`
    #[instrument(skip(self, withdrawal), fields(
        nonce = withdrawal.nonce,
        recipient = ?withdrawal.recipient_eth_address,
        amount = withdrawal.amount,
    ))]
    pub async fn submit_withdrawal(&self, withdrawal: &OutboundWithdrawal) -> BridgeResult<TxHash> {
        // CRITICAL: use the message bytes the committee actually
        // signed — DON'T reconstruct from the wrapper fields. Any
        // field difference (sender, timestamp_ms, ...) makes the
        // on-chain `ecrecover` recover the wrong address and the
        // contract rejects the cert as below-threshold.
        let calldata =
            eth_abi::encode_release_calldata(&withdrawal.message_bytes, &withdrawal.certificate)?;

        // EIP-1559 request — alloy fills nonce/fees/gasLimit from the
        // connected provider. `value = 0` because the release function
        // is non-payable.
        let tx = TransactionRequest::default()
            .with_from(self.wallet_address)
            .with_to(self.bridge_contract)
            .with_value(U256::ZERO)
            .with_input(Bytes::from(calldata));

        info!(
            from = ?self.wallet_address,
            to = ?self.bridge_contract,
            "submitting Eth-side withdrawal release tx"
        );

        let pending = self
            .provider
            .send_transaction(tx)
            .await
            .map_err(|e| BridgeError::ProviderError(format!("send_transaction: {e}")))?;
        let tx_hash = *pending.tx_hash();
        info!(?tx_hash, "release tx broadcasted");
        Ok(tx_hash)
    }

    /// Wait for the tx receipt + assert `status == 1`. Useful for
    /// tests and operator-grade waits; production may prefer to
    /// fire-and-forget and observe via event monitoring.
    pub async fn wait_for_success(&self, tx_hash: TxHash) -> BridgeResult<B256> {
        let receipt = self
            .provider
            .get_transaction_receipt(tx_hash)
            .await
            .map_err(|e| BridgeError::ProviderError(format!("get_receipt: {e}")))?
            .ok_or_else(|| BridgeError::ProviderError("receipt not yet available".into()))?;
        if !receipt.status() {
            warn!(?tx_hash, "release tx reverted on chain");
            return Err(BridgeError::Internal(format!("tx {tx_hash:?} reverted on chain")));
        }
        Ok(receipt.transaction_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::node_bindings::Anvil;

    /// Submitter constructs cleanly against a real (local) RPC and
    /// captures the wallet address. We don't deploy any contracts —
    /// the integration test in `tests/anvil_e2e.rs` covers the full
    /// deploy + submit path.
    #[tokio::test]
    async fn submitter_constructs_against_local_rpc() {
        let anvil = match Anvil::new().try_spawn() {
            Ok(a) => a,
            Err(e) => {
                eprintln!(
                    "skipping submitter_constructs_against_local_rpc: anvil unavailable ({e})"
                );
                return;
            }
        };
        let wallet = EthWallet::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        .unwrap();
        let s =
            EthSubmitter::new(&anvil.endpoint(), Address::ZERO, wallet).expect("EthSubmitter::new");
        assert_eq!(format!("{:?}", s.wallet_address), "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266");
    }
}
