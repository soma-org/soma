// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore as _;
use types::object::CoinType;

use crate::response::BalanceOutput;

/// Stage 13c: read the address's USDC and SOMA accumulator balances.
/// Coin objects are gone (Stage 13a), so the previous "list owned
/// Coin objects and sum them" path always returned 0.
///
/// `_with_coins` is accepted for backwards-compatible CLI flag
/// parsing but no longer means anything — there are no per-coin
/// objects to enumerate.
pub async fn execute(
    context: &WalletContext,
    address: Option<KeyIdentity>,
    _with_coins: bool,
) -> Result<BalanceOutput> {
    let address = match address {
        Some(key_id) => context.config.keystore.get_by_identity(&key_id)?,
        None => {
            context.config.active_address.ok_or_else(|| anyhow::anyhow!("No active address set"))?
        }
    };

    let client = context.get_client().await?;
    let usdc = client.get_balance_by_coin_type(&address, CoinType::Usdc).await?;
    let soma = client.get_balance_by_coin_type(&address, CoinType::Soma).await?;

    Ok(BalanceOutput { address, usdc, soma })
}
