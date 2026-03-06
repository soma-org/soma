// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::object::{Object, ObjectType, Owner};
use indexer_alt_schema::objects::{StoredCoinBalanceBucket, StoredCoinOwnerKind};
use indexer_alt_schema::schema::coin_balance_buckets;

use crate::handlers::checkpoint_input_objects;

pub struct CoinBalanceBuckets;

/// Compute the balance bucket using log10 of the coin balance.
fn get_coin_balance_bucket(balance: u64) -> i16 {
    if balance == 0 {
        0
    } else {
        balance.ilog10() as i16
    }
}

/// Determine the CoinOwnerKind for a Soma Owner.
/// In Soma, address-owned coins are "Fastpath" and shared coins are "Consensus".
fn coin_owner_kind(owner: &Owner) -> Option<StoredCoinOwnerKind> {
    match owner {
        Owner::AddressOwner(_) => Some(StoredCoinOwnerKind::Fastpath),
        Owner::Shared { .. } => Some(StoredCoinOwnerKind::Consensus),
        Owner::Immutable => None,
    }
}

/// Extract the coin balance bucket data from an object, if it's an address-owned coin.
fn coin_data(obj: &Object) -> Option<(StoredCoinOwnerKind, Vec<u8>, u64)> {
    let owner_kind = coin_owner_kind(obj.owner())?;
    if *obj.type_() != ObjectType::Coin {
        return None;
    }
    let balance = obj.as_coin()?;
    let owner_id = match obj.owner() {
        Owner::AddressOwner(addr) => addr.to_vec(),
        Owner::Shared { .. } => vec![],
        Owner::Immutable => return None,
    };
    Some((owner_kind, owner_id, balance))
}

#[async_trait]
impl Processor for CoinBalanceBuckets {
    const NAME: &'static str = "coin_balance_buckets";

    type Value = StoredCoinBalanceBucket;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            transactions,
            summary,
            object_set,
            ..
        } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;
        let checkpoint_input_objects = checkpoint_input_objects(checkpoint)?;

        let mut values = Vec::new();

        for tx in transactions {
            // Handle deleted/wrapped objects -- emit deletion entries for coins
            for change in tx.effects.object_changes() {
                if change.output_version.is_some() {
                    continue;
                }

                // Object was removed. Check if it was a coin in the input.
                if let Some(input_obj) = checkpoint_input_objects.get(&change.id) {
                    if input_obj.as_coin().is_some() {
                        values.push(StoredCoinBalanceBucket {
                            object_id: change.id.to_vec(),
                            cp_sequence_number,
                            owner_kind: None,
                            owner_id: None,
                            coin_type: None,
                            coin_balance_bucket: None,
                        });
                    }
                }
            }

            // Handle output objects
            for output_obj in tx.output_objects(object_set) {
                if let Some((owner_kind, owner_id, balance)) = coin_data(output_obj) {
                    // Check if the bucket actually changed vs input
                    let new_bucket = get_coin_balance_bucket(balance);

                    let changed = checkpoint_input_objects
                        .get(&output_obj.id())
                        .and_then(|old| {
                            let old_balance = old.as_coin()?;
                            let old_bucket = get_coin_balance_bucket(old_balance);
                            let (old_kind, old_id, _) = coin_data(old)?;
                            Some(old_bucket != new_bucket || old_kind != owner_kind || old_id != owner_id)
                        })
                        .unwrap_or(true); // If no old object, it's new

                    if changed {
                        values.push(StoredCoinBalanceBucket {
                            object_id: output_obj.id().to_vec(),
                            cp_sequence_number,
                            owner_kind: Some(owner_kind),
                            owner_id: Some(owner_id),
                            coin_type: Some(b"SOMA".to_vec()),
                            coin_balance_bucket: Some(new_bucket),
                        });
                    }
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for CoinBalanceBuckets {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(coin_balance_buckets::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
