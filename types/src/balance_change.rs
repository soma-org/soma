// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-tx balance attribution derived from `TransactionEffects.balance_events`.
//!
//! Stage 13m made `TransactionEffects.balance_events` the signed,
//! checkpoint-shipped record of every per-tx accumulator change. This
//! module is the canonical reader: walk the events and fold them into a
//! deterministic list of `(address, coin_type, signed_amount)` rows.
//!
//! Indexers, the gRPC `derive_balance_changes` field, the CLI tx
//! summary, and any downstream consumer that wants per-tx attribution
//! should call into here rather than re-execute or scrape the object
//! store.

use std::collections::BTreeMap;

use crate::balance::BalanceEvent;
use crate::base::SomaAddress;
use crate::effects::{TransactionEffects, TransactionEffectsAPI};
use crate::full_checkpoint_content::ObjectSet;
use crate::object::{CoinType, Object};

/// A net per-tx balance change for a single `(address, coin_type)`.
///
/// `amount` is signed: negative means the address paid out, positive
/// means it received. Magnitudes use `i128` so any realistic
/// aggregation of `u64` deposits/withdrawals fits without overflow.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BalanceChange {
    /// Owner of the balance change.
    pub address: SomaAddress,

    /// Coin denomination this delta applies to. Soma's accumulator is
    /// keyed per `(owner, coin_type)`, so a single tx can produce
    /// multiple `BalanceChange` entries — one per coin type the tx
    /// touched for any given address.
    pub coin_type: CoinType,

    /// Signed delta. Negative for outflows (e.g., transfer sender, gas
    /// payer), positive for inflows (e.g., transfer recipient, mint).
    pub amount: i128,
}

/// Derive per-tx balance changes from the signed effects.
///
/// Walks `effects.balance_events()` and aggregates `(address,
/// coin_type)` deltas into a deterministic, zero-filtered list. Output
/// ordering matches `BTreeMap` iteration over `(address, coin_type)`,
/// so two callers given the same effects produce byte-identical
/// results — important for cross-validator and cross-replica
/// consistency in indexers.
///
/// `_input_objects` and `_output_objects` are vestigial parameters
/// retained for source-compat with the pre-13m signature. They are
/// ignored — every signal needed is already on the effects struct.
pub fn derive_balance_changes(
    effects: &TransactionEffects,
    _input_objects: &[Object],
    _output_objects: &[Object],
) -> Vec<BalanceChange> {
    aggregate(effects.balance_events())
}

/// Same as [`derive_balance_changes`] but with the indexer-side
/// `ObjectSet` parameter (also vestigial post-13m).
pub fn derive_balance_changes_2(
    effects: &TransactionEffects,
    _objects: &ObjectSet,
) -> Vec<BalanceChange> {
    aggregate(effects.balance_events())
}

fn aggregate(events: &[BalanceEvent]) -> Vec<BalanceChange> {
    let mut net: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
    for ev in events {
        *net.entry((ev.owner(), ev.coin_type())).or_insert(0) += ev.signed_delta();
    }
    net.into_iter()
        .filter_map(|((address, coin_type), amount)| {
            (amount != 0).then_some(BalanceChange { address, coin_type, amount })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance::BalanceEvent;
    use crate::base::SomaAddress;
    use crate::digests::TransactionDigest;
    use crate::effects::{ExecutionStatus, TransactionEffects, TransactionEffectsV1};
    use crate::object::{CoinType, Version};
    use crate::tx_fee::TransactionFee;

    fn effects_with(events: Vec<BalanceEvent>) -> TransactionEffects {
        TransactionEffects::V1(TransactionEffectsV1 {
            status: ExecutionStatus::Success,
            executed_epoch: 0,
            transaction_digest: TransactionDigest::default(),
            version: Version::default(),
            changed_objects: vec![],
            dependencies: vec![],
            unchanged_shared_objects: vec![],
            transaction_fee: TransactionFee::default(),
            gas_object_index: None,
            balance_events: events,
            delegation_events: vec![],
        })
    }

    #[test]
    fn empty_effects_yield_no_changes() {
        let effects = effects_with(vec![]);
        assert!(derive_balance_changes(&effects, &[], &[]).is_empty());
    }

    #[test]
    fn netting_collapses_offsetting_events() {
        // A withdraw and equal deposit on the same (address, coin_type)
        // net to zero and drop out of the result entirely.
        let alice = SomaAddress::random();
        let effects = effects_with(vec![
            BalanceEvent::deposit(alice, CoinType::Usdc, 50),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 50),
        ]);
        assert!(derive_balance_changes(&effects, &[], &[]).is_empty());
    }

    #[test]
    fn signed_amounts_reflect_direction() {
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let effects = effects_with(vec![
            BalanceEvent::withdraw(alice, CoinType::Soma, 100),
            BalanceEvent::deposit(bob, CoinType::Soma, 100),
        ]);
        let changes = derive_balance_changes(&effects, &[], &[]);
        assert_eq!(changes.len(), 2);
        let by_addr: BTreeMap<_, _> =
            changes.iter().map(|c| ((c.address, c.coin_type), c.amount)).collect();
        assert_eq!(by_addr[&(alice, CoinType::Soma)], -100);
        assert_eq!(by_addr[&(bob, CoinType::Soma)], 100);
    }

    #[test]
    fn distinct_coin_types_are_separate_rows() {
        // One address transacting in both SOMA and USDC produces two
        // independent rows — the coin_type discriminates.
        let alice = SomaAddress::random();
        let effects = effects_with(vec![
            BalanceEvent::deposit(alice, CoinType::Soma, 10),
            BalanceEvent::deposit(alice, CoinType::Usdc, 20),
        ]);
        let changes = derive_balance_changes(&effects, &[], &[]);
        assert_eq!(changes.len(), 2);
    }

    #[test]
    fn output_ordering_is_deterministic() {
        // Same events emitted in different orders must produce the
        // same Vec<BalanceChange>. This is what makes the indexer
        // deterministic across replicas.
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let a = effects_with(vec![
            BalanceEvent::deposit(alice, CoinType::Soma, 10),
            BalanceEvent::deposit(bob, CoinType::Usdc, 20),
        ]);
        let b = effects_with(vec![
            BalanceEvent::deposit(bob, CoinType::Usdc, 20),
            BalanceEvent::deposit(alice, CoinType::Soma, 10),
        ]);
        assert_eq!(
            derive_balance_changes(&a, &[], &[]),
            derive_balance_changes(&b, &[], &[]),
        );
    }
}
