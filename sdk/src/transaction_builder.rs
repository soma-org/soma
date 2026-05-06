// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use fastcrypto::encoding::Encoding as _;
use types::base::SomaAddress;
use types::digests::{CheckpointDigest, ChainIdentifier};
use types::transaction::{Transaction, TransactionData, TransactionExpiration, TransactionKind};

use crate::wallet_context::WalletContext;

/// Per-process nonce so two builds for the same `kind`+`sender`
/// during the same epoch still produce distinct tx digests. Doesn't
/// affect protocol semantics — only the tx digest cache cares.
static STATELESS_NONCE: AtomicU32 = AtomicU32::new(0);

/// A builder for constructing and optionally executing transactions.
///
/// Stage 13c: gas is balance-mode for non-system txs — the
/// builder always emits `gas_payment = vec![]` and lets the
/// authority's `prepare_gas` debit the sender's USDC accumulator.
///
/// Stateless txs (those with no owned input objects: BalanceTransfer,
/// AddStake/WithdrawStake, BridgeWithdraw, OpenChannel, Settle,
/// RequestClose, WithdrawAfterTimeout, TopUp) require a
/// `TransactionExpiration::ValidDuring` declaration — the validator
/// rejects stateless txs with `None`. Use [`build_transaction_async`]
/// (or [`build_transaction_data_async`]) and the builder will look
/// up the current epoch + chain id and emit `ValidDuring` for a
/// 2-epoch window automatically.
pub struct TransactionBuilder<'a> {
    context: &'a WalletContext,
}

impl<'a> TransactionBuilder<'a> {
    pub fn new(context: &'a WalletContext) -> Self {
        Self { context }
    }

    /// Build unsigned transaction data for a given kind.
    ///
    /// Always emits `TransactionExpiration::None`. For stateless tx
    /// kinds (no owned inputs), prefer [`build_transaction_data_async`]
    /// — the validator rejects stateless txs whose expiration is
    /// `None`, and this method is kept only for back-compat with
    /// existing callers that already declare expiration explicitly.
    pub fn build_transaction_data(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<TransactionData> {
        Ok(TransactionData::new(kind, sender, Vec::new()))
    }

    /// Build unsigned transaction data, auto-declaring
    /// `TransactionExpiration::ValidDuring` for stateless txs.
    ///
    /// Looks up the current epoch + chain id via the wallet's RPC
    /// client; the resulting `ValidDuring` window is `[epoch,
    /// epoch+1]` (the maximum allowed by the protocol). For
    /// non-stateless kinds (TransferObjects), emits `None` — those
    /// are replay-protected by version-bump on the consumed objects.
    pub async fn build_transaction_data_async(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<TransactionData> {
        let expiration = self.expiration_for_kind(&kind).await?;
        Ok(TransactionData::new_with_expiration(kind, sender, Vec::new(), expiration))
    }

    /// Build a signed transaction ready for execution.
    ///
    /// Equivalent to [`build_transaction_data`] + sign — keeps
    /// `TransactionExpiration::None`. Prefer
    /// [`build_transaction_async`] for stateless tx kinds.
    pub async fn build_transaction(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<Transaction> {
        let tx_data = self.build_transaction_data(sender, kind)?;
        let tx = self.context.sign_transaction(&tx_data).await;
        Ok(tx)
    }

    /// Async variant: builds with auto-declared `ValidDuring` for
    /// stateless txs, then signs. The right default for any tx kind
    /// post-Stage-6.
    pub async fn build_transaction_async(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<Transaction> {
        let tx_data = self.build_transaction_data_async(sender, kind).await?;
        let tx = self.context.sign_transaction(&tx_data).await;
        Ok(tx)
    }

    /// Build and serialize unsigned transaction data as base64.
    /// Useful for offline signing workflows.
    pub fn build_serialized_unsigned(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<String> {
        let tx_data = self.build_transaction_data(sender, kind)?;
        let bytes = bcs::to_bytes(&tx_data)?;
        Ok(fastcrypto::encoding::Base64::encode(&bytes))
    }

    /// Compute the appropriate `TransactionExpiration` for a tx of
    /// the given kind:
    ///   * system kinds → `None` (they have their own replay
    ///     protections)
    ///   * `TransferObjects` (only kind that touches owned inputs in
    ///     balance-mode) → `None` — the owned-input version-bump
    ///     catches replays
    ///   * everything else → `ValidDuring { now, now+1, chain, fresh_nonce }`
    ///     because Stage 13c gas is `vec![]` and the validator
    ///     considers any tx with empty gas "stateless" for replay
    ///     purposes regardless of shared inputs (see
    ///     `authority::handle_transaction_*` Stage 5.5c check).
    async fn expiration_for_kind(
        &self,
        kind: &TransactionKind,
    ) -> Result<TransactionExpiration> {
        // System txs handle replay protection internally — pass-through.
        if kind.is_system_tx() {
            return Ok(TransactionExpiration::None);
        }

        // The only user-tx kind that has owned inputs in balance-mode
        // is `TransferObjects` — its version-bump on the consumed
        // objects catches replays.
        if matches!(kind, TransactionKind::TransferObjects { .. }) {
            return Ok(TransactionExpiration::None);
        }

        // Everything else is "stateless" from the validator's PoV
        // (empty `gas` after Stage 13c). Declare a 2-epoch ValidDuring.
        let client = self.context.get_client().await?;
        let chain_id_str = client.get_chain_identifier().await?;
        let chain = parse_chain_identifier(&chain_id_str)?;

        let epoch_resp = client.get_epoch(None).await?;
        let current_epoch = epoch_resp
            .epoch
            .and_then(|e| e.epoch)
            .ok_or_else(|| anyhow::anyhow!("epoch not found in get_epoch response"))?;

        let nonce = STATELESS_NONCE.fetch_add(1, Ordering::Relaxed);

        Ok(TransactionExpiration::ValidDuring {
            min_epoch: Some(current_epoch),
            max_epoch: Some(current_epoch.saturating_add(1)),
            chain,
            nonce,
        })
    }
}

/// Parse the wire `chain_id` (base58 32-byte string emitted by
/// `get_service_info`) into a `ChainIdentifier`.
fn parse_chain_identifier(s: &str) -> Result<ChainIdentifier> {
    use std::str::FromStr as _;
    let digest = rpc::types::Digest::from_str(s)
        .map_err(|e| anyhow::anyhow!("invalid chain_id digest {s:?}: {e}"))?;
    let bytes: [u8; 32] = (*<rpc::types::Digest as AsRef<[u8; rpc::types::Digest::LENGTH]>>::as_ref(&digest)).into();
    let cp_digest = CheckpointDigest::new(bytes);
    Ok(ChainIdentifier::from(cp_digest))
}

/// Options for transaction execution behavior. Stage 13c: gas is
/// balance-mode and not selectable; the only knob left is whether
/// to serialize-instead-of-execute.
#[derive(Debug, Clone, Default)]
pub struct ExecutionOptions {
    /// If true, serialize the unsigned transaction instead of executing
    pub serialize_unsigned: bool,
}

impl ExecutionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn serialize_unsigned(mut self) -> Self {
        self.serialize_unsigned = true;
        self
    }
}
