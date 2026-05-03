// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use fastcrypto::encoding::Encoding as _;
use types::base::SomaAddress;
use types::transaction::{Transaction, TransactionData, TransactionKind};

use crate::wallet_context::WalletContext;

/// A builder for constructing and optionally executing transactions.
///
/// Stage 13c: gas is balance-mode for non-system txs — the
/// builder always emits `gas_payment = vec![]` and lets the
/// authority's `prepare_gas` debit the sender's USDC accumulator.
pub struct TransactionBuilder<'a> {
    context: &'a WalletContext,
}

impl<'a> TransactionBuilder<'a> {
    pub fn new(context: &'a WalletContext) -> Self {
        Self { context }
    }

    /// Build unsigned transaction data for a given kind.
    pub fn build_transaction_data(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<TransactionData> {
        Ok(TransactionData::new(kind, sender, Vec::new()))
    }

    /// Build a signed transaction ready for execution.
    pub async fn build_transaction(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> Result<Transaction> {
        let tx_data = self.build_transaction_data(sender, kind)?;
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
