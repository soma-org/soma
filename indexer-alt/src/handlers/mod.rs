// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::HashSet;

use anyhow::Context;
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::object::{Object, ObjectID, Owner};

pub mod coin_balance_buckets;
pub mod cp_sequence_numbers;
pub mod kv_checkpoints;
pub mod kv_epoch_ends;
pub mod kv_epoch_starts;
pub mod kv_objects;
pub mod kv_transactions;
pub mod obj_info;
pub mod obj_versions;
pub mod soma_asks;
pub mod soma_bids;
pub mod soma_epoch_state;
pub mod soma_settlements;
pub mod soma_staked_soma;
pub mod soma_tx_details;
pub mod soma_validators;
pub mod soma_vaults;
pub mod tx_affected_addresses;
pub mod tx_affected_objects;
pub mod tx_balance_changes;
pub mod tx_calls;
pub mod tx_digests;
pub mod tx_kinds;

/// Returns the first appearance of all objects that were used as inputs to the transactions in the
/// checkpoint. These are objects that existed prior to the checkpoint, and excludes objects that
/// were created or unwrapped within the checkpoint.
pub fn checkpoint_input_objects(
    checkpoint: &Checkpoint,
) -> anyhow::Result<BTreeMap<ObjectID, &Object>> {
    let mut output_objects_seen = HashSet::new();
    let mut checkpoint_input_objects = BTreeMap::new();

    for tx in checkpoint.transactions.iter() {
        for change in tx.effects.object_changes() {
            let id = change.id;

            let Some(version) = change.input_version else {
                continue;
            };

            // This object was previously modified, created, or unwrapped in the checkpoint, so
            // this version is not a checkpoint input.
            if output_objects_seen.contains(&id) {
                continue;
            }

            // Make sure this object has not already been recorded as an input.
            if checkpoint_input_objects.contains_key(&id) {
                continue;
            }

            let input_obj = tx
                .input_objects(&checkpoint.object_set)
                .find(|obj| obj.id() == id && obj.version() == version)
                .with_context(|| {
                    format!(
                        "Object {id} at version {:?} referenced in effects not found in input_objects",
                        version,
                    )
                })?;

            checkpoint_input_objects.insert(id, input_obj);
        }

        for change in tx.effects.object_changes() {
            if change.output_version.is_some() {
                output_objects_seen.insert(change.id);
            }
        }
    }

    Ok(checkpoint_input_objects)
}

/// The recipient addresses from changed objects in a transaction's effects.
///
/// Returns addresses from `AddressOwner` owners, skipping other owner types.
pub fn affected_addresses(
    effects: &types::effects::TransactionEffects,
) -> impl Iterator<Item = SomaAddress> {
    effects.all_changed_objects().into_iter().filter_map(|(_, owner, _)| match owner {
        Owner::AddressOwner(address) => Some(address),
        _ => None,
    })
}
