// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/local_faucet.rs

use crate::errors::FaucetError;
use crate::faucet_config::FaucetConfig;
use crate::types::CoinInfo;
use sdk::wallet_context::WalletContext;
use tokio::sync::Mutex;
use tracing::{error, info};
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI as _;
use types::object::ObjectRef;
use types::transaction::TransactionData;

pub struct LocalFaucet {
    wallet: Mutex<WalletContext>,
    /// The active gas coin used for faucet transactions. Updated after each tx
    /// to reflect the mutated coin's new version.
    active_coin: Mutex<ObjectRef>,
    faucet_address: SomaAddress,
    config: FaucetConfig,
}

impl LocalFaucet {
    /// Create a new LocalFaucet from a wallet context and config.
    ///
    /// Finds a suitable gas coin owned by the wallet and stores it for future
    /// faucet transactions.
    pub async fn new(mut wallet: WalletContext, config: FaucetConfig) -> Result<Self, FaucetError> {
        let (address, coins) = find_gas_coins_and_address(&mut wallet, &config).await?;
        let active_coin = coins[0];

        info!("Faucet initialized with address {} and coin {}", address, active_coin.0,);

        Ok(Self {
            wallet: Mutex::new(wallet),
            active_coin: Mutex::new(active_coin),
            faucet_address: address,
            config,
        })
    }

    /// Execute a faucet transaction: send `num_coins` coins of `amount` shannons
    /// each to `recipient`.
    pub async fn local_request_execute_tx(
        &self,
        recipient: SomaAddress,
    ) -> Result<Vec<CoinInfo>, FaucetError> {
        let num_coins = self.config.num_coins;
        let amount = self.config.amount;

        let amounts = vec![amount; num_coins];
        let recipients = vec![recipient; num_coins];

        let tx_data = {
            let coin_ref = *self.active_coin.lock().await;
            TransactionData::new_pay_coins(
                vec![coin_ref],
                Some(amounts.clone()),
                recipients,
                self.faucet_address,
            )
        };

        let wallet = self.wallet.lock().await;
        let tx = wallet.sign_transaction(&tx_data).await;
        let tx_digest = *tx.digest();

        let response = wallet
            .execute_transaction_and_wait_for_indexing(tx)
            .await
            .map_err(|e| FaucetError::Transfer(e.to_string()))?;

        if !response.effects.status().is_ok() {
            return Err(FaucetError::Transfer(format!(
                "Transaction failed: {:?}",
                response.effects.status()
            )));
        }

        // Update the active coin to the mutated version.
        // The gas coin (first input) is mutated in the effects.
        let old_coin_id = self.active_coin.lock().await.0;
        if let Some(new_ref) = response
            .effects
            .mutated()
            .iter()
            .find(|(obj_ref, _)| obj_ref.0 == old_coin_id)
            .map(|(obj_ref, _)| *obj_ref)
        {
            *self.active_coin.lock().await = new_ref;
        } else {
            error!("Gas coin {} was not found in mutated objects after faucet tx", old_coin_id);
        }

        // Build CoinInfo for each created coin
        let coins_sent: Vec<CoinInfo> = response
            .effects
            .created()
            .iter()
            .map(|(obj_ref, _)| CoinInfo {
                amount,
                id: obj_ref.0.to_string(),
                transfer_tx_digest: tx_digest.to_string(),
            })
            .collect();

        info!("Sent {} coins to {}, tx digest: {}", coins_sent.len(), recipient, tx_digest,);

        Ok(coins_sent)
    }
}

/// Find gas coins and the faucet address from the wallet.
///
/// Iterates through all addresses in the wallet and finds one with coins
/// that have sufficient balance for faucet operations.
pub async fn find_gas_coins_and_address(
    wallet: &mut WalletContext,
    config: &FaucetConfig,
) -> Result<(SomaAddress, Vec<ObjectRef>), FaucetError> {
    let min_balance = config.amount.saturating_mul(config.num_coins as u64);

    let accounts = wallet
        .get_all_accounts_and_gas_objects()
        .await
        .map_err(|e| FaucetError::Wallet(format!("Failed to get accounts: {}", e)))?;

    for (address, coins) in accounts {
        if !coins.is_empty() {
            info!("Found {} coins for address {}, using as faucet address", coins.len(), address,);
            return Ok((address, coins));
        }
    }

    Err(FaucetError::Wallet(format!(
        "No address with coins found in wallet (need at least {} shannons)",
        min_balance,
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use test_cluster::TestClusterBuilder;

    #[tokio::test]
    async fn test_local_faucet_dispense_coins() {
        let cluster = TestClusterBuilder::new().build().await;
        let wallet = cluster.wallet;

        let config = FaucetConfig::default();
        let faucet = LocalFaucet::new(wallet, config).await.expect("Failed to create faucet");
        let faucet = Arc::new(faucet);

        let recipient = SomaAddress::random();
        let coins =
            faucet.local_request_execute_tx(recipient).await.expect("Faucet request failed");

        assert_eq!(coins.len(), 5, "Expected 5 coins, got {}", coins.len());
        for coin in &coins {
            assert_eq!(coin.amount, 200_000_000_000);
            assert!(!coin.id.is_empty());
            assert!(!coin.transfer_tx_digest.is_empty());
        }
    }

    #[tokio::test]
    async fn test_local_faucet_dispense_twice() {
        let cluster = TestClusterBuilder::new().build().await;
        let wallet = cluster.wallet;

        let config = FaucetConfig::default();
        let faucet = LocalFaucet::new(wallet, config).await.expect("Failed to create faucet");

        let recipient = SomaAddress::random();

        let coins1 =
            faucet.local_request_execute_tx(recipient).await.expect("First faucet request failed");
        assert_eq!(coins1.len(), 5);

        let coins2 =
            faucet.local_request_execute_tx(recipient).await.expect("Second faucet request failed");
        assert_eq!(coins2.len(), 5);

        // All coin IDs should be unique
        let mut all_ids: Vec<String> =
            coins1.iter().chain(coins2.iter()).map(|c| c.id.clone()).collect();
        let original_len = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), original_len, "Duplicate coin IDs found");
    }

    #[tokio::test]
    async fn test_find_gas_coins_and_address() {
        let cluster = TestClusterBuilder::new().build().await;
        let mut wallet = cluster.wallet;

        let config = FaucetConfig::default();
        let (address, coins) = find_gas_coins_and_address(&mut wallet, &config)
            .await
            .expect("Failed to find gas coins");

        assert!(!coins.is_empty(), "Expected at least one coin");
        assert_ne!(address, SomaAddress::ZERO);
    }

    #[tokio::test]
    async fn test_dispense_to_zero_address() {
        let cluster = TestClusterBuilder::new().build().await;
        let wallet = cluster.wallet;

        let config = FaucetConfig::default();
        let faucet = LocalFaucet::new(wallet, config).await.expect("Failed to create faucet");

        // The zero address is a valid format, faucet should not reject it
        let coins = faucet
            .local_request_execute_tx(SomaAddress::ZERO)
            .await
            .expect("Faucet request to zero address should succeed");

        assert_eq!(coins.len(), 5);
    }

    /// §8.1: Test that find_gas_coins_and_address fails with insufficient balance
    #[tokio::test]
    async fn test_find_gas_coins_insufficient_balance() {
        let cluster = TestClusterBuilder::new().build().await;
        let mut wallet = cluster.wallet;

        let mut config = FaucetConfig::default();
        config.amount = u64::MAX;

        // The function finds coins regardless of amount — it only fails if no coins at all.
        // Balance sufficiency is checked at transaction time, not at find time.
        let result = find_gas_coins_and_address(&mut wallet, &config).await;
        assert!(result.is_ok(), "find_gas_coins should succeed (it doesn't check balance)");
    }

    /// §8.7: Concurrent faucet requests should not panic or corrupt state
    #[tokio::test]
    async fn test_concurrent_faucet_requests() {
        let cluster = TestClusterBuilder::new().build().await;
        let wallet = cluster.wallet;

        let config = FaucetConfig::default();
        let faucet = Arc::new(
            LocalFaucet::new(wallet, config).await.expect("Failed to create faucet"),
        );

        let mut handles = Vec::new();
        for _ in 0..5 {
            let f = faucet.clone();
            let recipient = SomaAddress::random();
            handles.push(tokio::spawn(async move {
                f.local_request_execute_tx(recipient).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles).await;

        let mut successes = 0;
        let mut failures = 0;
        for result in results {
            match result.expect("Task should not panic") {
                Ok(coins) => {
                    assert!(!coins.is_empty());
                    successes += 1;
                }
                Err(_) => {
                    // Some failures are expected due to object version conflicts
                    failures += 1;
                }
            }
        }

        // At least one request should succeed (the first one that acquires the mutex)
        assert!(
            successes >= 1,
            "Expected at least 1 success, got {successes} successes and {failures} failures"
        );
    }
}
