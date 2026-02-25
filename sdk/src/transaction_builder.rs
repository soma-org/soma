// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use fastcrypto::encoding::Encoding as _;
use types::base::SomaAddress;
use types::crypto::SomaKeyPair;
use types::object::ObjectRef;
use types::transaction::{Transaction, TransactionData, TransactionKind};

use crate::wallet_context::WalletContext;

/// A builder for constructing and optionally executing transactions.
///
/// This provides a clean interface for CLI commands to build transactions
/// without dealing with gas selection, signing, and execution details.
pub struct TransactionBuilder<'a> {
    context: &'a WalletContext,
}

impl<'a> TransactionBuilder<'a> {
    pub fn new(context: &'a WalletContext) -> Self {
        Self { context }
    }

    /// Build unsigned transaction data for a given kind.
    /// Returns the TransactionData which can be serialized or signed.
    pub async fn build_transaction_data(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
        gas: Option<ObjectRef>,
    ) -> Result<TransactionData> {
        let gas_payment = match gas {
            Some(gas_ref) => vec![gas_ref],
            None => {
                let gas_ref =
                    self.context.get_one_gas_object_owned_by_address(sender).await?.ok_or_else(
                        || {
                            anyhow!(
                                "No gas object found for address {}. \
                             Please ensure the address has coins.",
                                sender
                            )
                        },
                    )?;
                vec![gas_ref]
            }
        };

        Ok(TransactionData::new(kind, sender, gas_payment))
    }

    /// Build a signed transaction ready for execution.
    pub async fn build_transaction(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
        gas: Option<ObjectRef>,
    ) -> Result<Transaction> {
        let tx_data = self.build_transaction_data(sender, kind, gas).await?;
        let tx = self.context.sign_transaction(&tx_data).await;
        Ok(tx)
    }

    /// Build and serialize unsigned transaction data as base64.
    /// Useful for offline signing workflows.
    pub async fn build_serialized_unsigned(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
        gas: Option<ObjectRef>,
    ) -> Result<String> {
        let tx_data = self.build_transaction_data(sender, kind, gas).await?;
        let bytes = bcs::to_bytes(&tx_data)?;
        Ok(fastcrypto::encoding::Base64::encode(&bytes))
    }
}

/// Options for transaction execution behavior
#[derive(Debug, Clone, Default)]
pub struct ExecutionOptions {
    /// If true, serialize the unsigned transaction instead of executing
    pub serialize_unsigned: bool,
    /// Optional gas object to use (otherwise auto-selected)
    pub gas: Option<ObjectRef>,
}

impl ExecutionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn serialize_unsigned(mut self) -> Self {
        self.serialize_unsigned = true;
        self
    }

    pub fn with_gas(mut self, gas: ObjectRef) -> Self {
        self.gas = Some(gas);
        self
    }
}
