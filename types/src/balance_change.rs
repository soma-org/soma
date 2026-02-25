// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::effects::{TransactionEffects, TransactionEffectsAPI as _};
use crate::full_checkpoint_content::ObjectSet;
use crate::object::{Object, Owner};
use crate::storage::ObjectKey;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BalanceChange {
    /// Owner of the balance change
    pub address: SomaAddress,

    /// The amount indicate the balance value changes.
    ///
    /// A negative amount means spending coin value and positive means receiving coin value.
    pub amount: i128,
}

fn coins(objects: &[Object]) -> impl Iterator<Item = (&SomaAddress, u64)> + '_ {
    objects.iter().filter_map(|object| {
        let address = match object.owner() {
            Owner::AddressOwner(soma_address) => soma_address,
            Owner::Shared { .. } | Owner::Immutable => return None,
        };
        let balance = object.as_coin()?;
        Some((address, balance))
    })
}

pub fn derive_balance_changes(
    _effects: &TransactionEffects,
    input_objects: &[Object],
    output_objects: &[Object],
) -> Vec<BalanceChange> {
    // 1. subtract all input coins
    let balances = coins(input_objects).fold(
        std::collections::BTreeMap::<_, i128>::new(),
        |mut acc, (address, balance)| {
            *acc.entry(address).or_default() -= balance as i128;
            acc
        },
    );

    // 2. add all mutated/output coins
    let balances = coins(output_objects).fold(balances, |mut acc, (address, balance)| {
        *acc.entry(address).or_default() += balance as i128;
        acc
    });

    balances
        .into_iter()
        .filter_map(|(address, amount)| {
            if amount == 0 {
                return None;
            }

            Some(BalanceChange { address: *address, amount })
        })
        .collect()
}

pub fn derive_balance_changes_2(
    effects: &TransactionEffects,
    objects: &ObjectSet,
) -> Vec<BalanceChange> {
    let input_objects = effects
        .modified_at_versions()
        .into_iter()
        .filter_map(|(object_id, version)| objects.get(&ObjectKey(object_id, version)).cloned())
        .collect::<Vec<_>>();
    let output_objects = effects
        .all_changed_objects()
        .into_iter()
        .filter_map(|(object_ref, _owner, _kind)| objects.get(&object_ref.into()).cloned())
        .collect::<Vec<_>>();

    // 1. subtract all input coins
    let balances = coins(&input_objects).fold(
        std::collections::BTreeMap::<_, i128>::new(),
        |mut acc, (address, balance)| {
            *acc.entry(address).or_default() -= balance as i128;
            acc
        },
    );

    // 2. add all mutated/output coins
    let balances = coins(&output_objects).fold(balances, |mut acc, (address, balance)| {
        *acc.entry(address).or_default() += balance as i128;
        acc
    });

    balances
        .into_iter()
        .filter_map(|(address, amount)| {
            if amount == 0 {
                return None;
            }

            Some(BalanceChange { address: *address, amount })
        })
        .collect()
}
