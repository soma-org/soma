// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! DataLoader implementations for batching nested resolver queries.
//!
//! These loaders implement `async_graphql::dataloader::Loader` to batch multiple
//! per-parent database queries into a single query, preventing N+1 query storms.

use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::Arc;

use async_graphql::dataloader::Loader;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;

use crate::api::types::reward::{Reward, RewardBalance};
use crate::db::PgReader;

// ---------------------------------------------------------------------------
// TargetReportersLoader
// ---------------------------------------------------------------------------

/// Batch-loading key for target reporters: (target_id, cp_sequence_number).
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct TargetReporterKey {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
}

/// Batches `soma_target_reports` lookups across multiple targets.
pub struct TargetReportersLoader {
    pub pg: Arc<PgReader>,
}

impl Loader<TargetReporterKey> for TargetReportersLoader {
    type Value = Vec<Vec<u8>>;
    type Error = Arc<anyhow::Error>;

    async fn load(
        &self,
        keys: &[TargetReporterKey],
    ) -> Result<HashMap<TargetReporterKey, Self::Value>, Self::Error> {
        use indexer_alt_schema::schema::soma_target_reports;

        let mut conn = self.pg.connect().await.map_err(|e| Arc::new(e.into()))?;

        let all_target_ids: Vec<&[u8]> =
            keys.iter().map(|k| k.target_id.as_slice()).collect();

        let rows: Vec<(Vec<u8>, i64, Vec<u8>)> = soma_target_reports::table
            .select((
                soma_target_reports::target_id,
                soma_target_reports::cp_sequence_number,
                soma_target_reports::reporter,
            ))
            .filter(soma_target_reports::target_id.eq_any(all_target_ids))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Arc::new(e.into()))?;

        let mut map: HashMap<TargetReporterKey, Vec<Vec<u8>>> = HashMap::new();
        for (tid, cp, reporter) in rows {
            let key = TargetReporterKey {
                target_id: tid,
                cp_sequence_number: cp,
            };
            // Only include rows matching requested keys (filter out cross-matches)
            if keys.contains(&key) {
                map.entry(key).or_default().push(reporter);
            }
        }
        Ok(map)
    }
}

// ---------------------------------------------------------------------------
// TargetRewardLoader
// ---------------------------------------------------------------------------

/// Batches `soma_rewards` + `soma_reward_balances` lookups across multiple targets.
pub struct TargetRewardLoader {
    pub pg: Arc<PgReader>,
}

impl Loader<Vec<u8>> for TargetRewardLoader {
    type Value = Reward;
    type Error = Arc<anyhow::Error>;

    async fn load(
        &self,
        keys: &[Vec<u8>],
    ) -> Result<HashMap<Vec<u8>, Self::Value>, Self::Error> {
        use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

        let mut conn = self.pg.connect().await.map_err(|e| Arc::new(e.into()))?;

        let target_id_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

        // Batch-load all matching rewards
        type RewardRow = (Vec<u8>, i64, i64, Vec<u8>);
        let reward_rows: Vec<RewardRow> = soma_rewards::table
            .select((
                soma_rewards::target_id,
                soma_rewards::cp_sequence_number,
                soma_rewards::epoch,
                soma_rewards::tx_digest,
            ))
            .filter(soma_rewards::target_id.eq_any(&target_id_refs))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Arc::new(e.into()))?;

        // Batch-load all matching balances
        type BalanceRow = (Vec<u8>, Vec<u8>, i64);
        let balance_rows: Vec<BalanceRow> = soma_reward_balances::table
            .select((
                soma_reward_balances::target_id,
                soma_reward_balances::recipient,
                soma_reward_balances::amount,
            ))
            .filter(soma_reward_balances::target_id.eq_any(&target_id_refs))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Arc::new(e.into()))?;

        // Group balances by target_id
        let mut balances_map: HashMap<Vec<u8>, Vec<RewardBalance>> = HashMap::new();
        for (tid, recipient, amount) in balance_rows {
            balances_map
                .entry(tid)
                .or_default()
                .push(RewardBalance { recipient, amount });
        }

        // Build reward map
        let mut map: HashMap<Vec<u8>, Reward> = HashMap::new();
        for (target_id, cp_sequence_number, epoch, tx_digest) in reward_rows {
            let balances = balances_map.remove(&target_id).unwrap_or_default();
            map.insert(
                target_id.clone(),
                Reward {
                    target_id,
                    cp_sequence_number,
                    epoch,
                    tx_digest,
                    balances,
                },
            );
        }

        Ok(map)
    }
}
