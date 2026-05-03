// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::effects::TransactionEffects;
use crate::full_checkpoint_content::ObjectSet;
use crate::object::Object;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BalanceChange {
    /// Owner of the balance change
    pub address: SomaAddress,

    /// The amount indicate the balance value changes.
    ///
    /// A negative amount means spending coin value and positive means receiving coin value.
    pub amount: i128,
}

/// Stage 13k: BalanceChange derivation used to walk Coin objects in
/// the input/output set. Stage 13a stopped materializing Coin
/// objects; the equivalent post-13c signal is the `BalanceEvent`
/// stream emitted by executors and aggregated by the per-commit
/// Settlement transaction. Until those events get plumbed into
/// `TransactionInfo.balance_changes` directly, derivation returns
/// empty. The struct + function signatures stay so existing
/// gRPC/RPC callers don't need to change.
pub fn derive_balance_changes(
    _effects: &TransactionEffects,
    _input_objects: &[Object],
    _output_objects: &[Object],
) -> Vec<BalanceChange> {
    Vec::new()
}

pub fn derive_balance_changes_2(
    _effects: &TransactionEffects,
    _objects: &ObjectSet,
) -> Vec<BalanceChange> {
    Vec::new()
}
