// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, SomaAddress};

/// A target on the Soma network — represents an inference task.
pub struct Target {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub status: String,
    pub submitter: Option<Vec<u8>>,
    pub winning_model_id: Option<Vec<u8>>,
    pub reward_pool: i64,
    pub bond_amount: i64,
    pub report_count: i32,
    pub winning_distance_score: Option<f64>,
    pub winning_loss_score: Option<f64>,
    pub winning_model_owner: Option<Vec<u8>>,
    pub fill_epoch: Option<i64>,
    pub distance_threshold: f64,
    pub model_ids_json: String,
    pub winning_data_url: Option<String>,
    pub winning_data_checksum: Option<Vec<u8>>,
    pub winning_data_size: Option<i64>,
}

#[Object]
impl Target {
    /// The target's object ID.
    async fn target_id(&self) -> SomaAddress {
        SomaAddress(self.target_id.clone())
    }

    /// The checkpoint where this version of the target was written.
    async fn checkpoint_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// The epoch this target was created in.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// The target's status: open, filled, or claimed.
    async fn status(&self) -> &str {
        &self.status
    }

    /// The address that filled this target (if any).
    async fn submitter(&self) -> Option<SomaAddress> {
        self.submitter.as_ref().map(|s| SomaAddress(s.clone()))
    }

    /// The model used in the winning submission (if any).
    async fn winning_model_id(&self) -> Option<SomaAddress> {
        self.winning_model_id
            .as_ref()
            .map(|id| SomaAddress(id.clone()))
    }

    /// The pre-allocated reward amount (shannons).
    async fn reward_pool(&self) -> BigInt {
        BigInt(self.reward_pool)
    }

    /// The submission bond held (shannons).
    async fn bond_amount(&self) -> BigInt {
        BigInt(self.bond_amount)
    }

    /// Number of fraud reports filed against this target.
    async fn report_count(&self) -> i32 {
        self.report_count
    }

    /// Cosine distance score from the winning submission (scalar).
    async fn winning_distance_score(&self) -> Option<f64> {
        self.winning_distance_score
    }

    /// Mean loss score from the winning submission's model inference.
    async fn winning_loss_score(&self) -> Option<f64> {
        self.winning_loss_score
    }

    /// The owner of the winning model at fill time.
    async fn winning_model_owner(&self) -> Option<SomaAddress> {
        self.winning_model_owner
            .as_ref()
            .map(|a| SomaAddress(a.clone()))
    }

    /// The epoch in which this target was filled (if filled or claimed).
    async fn fill_epoch(&self) -> Option<BigInt> {
        self.fill_epoch.map(BigInt)
    }

    /// Cosine distance threshold — submitter must report distance <= this value.
    async fn distance_threshold(&self) -> f64 {
        self.distance_threshold
    }

    /// Model IDs assigned to this target (hex addresses).
    async fn model_ids(&self) -> Vec<SomaAddress> {
        let ids: Vec<String> = serde_json::from_str(&self.model_ids_json).unwrap_or_default();
        ids.into_iter()
            .filter_map(|hex_str| {
                let hex = hex_str.strip_prefix("0x").unwrap_or(&hex_str);
                hex::decode(hex).ok().map(SomaAddress)
            })
            .collect()
    }

    /// URL where the winning submission's data can be downloaded.
    async fn winning_data_url(&self) -> Option<&str> {
        self.winning_data_url.as_deref()
    }

    /// Checksum of the winning submission's data.
    async fn winning_data_checksum(&self) -> Option<Base64> {
        self.winning_data_checksum
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// Size of the winning submission's data in bytes.
    async fn winning_data_size(&self) -> Option<BigInt> {
        self.winning_data_size.map(BigInt)
    }

    /// Addresses that have filed fraud reports against this target.
    async fn reporters(&self, ctx: &Context<'_>) -> Result<Vec<SomaAddress>> {
        use std::ops::DerefMut;
        use std::sync::Arc;

        use diesel::ExpressionMethods;
        use diesel::QueryDsl;
        use diesel_async::RunQueryDsl;

        use crate::db::PgReader;

        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_target_reports;

        // Get reporters at the latest checkpoint for this target
        let reporters: Vec<Vec<u8>> = soma_target_reports::table
            .select(soma_target_reports::reporter)
            .filter(soma_target_reports::target_id.eq(&self.target_id))
            .filter(soma_target_reports::cp_sequence_number.eq(self.cp_sequence_number))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(reporters.into_iter().map(SomaAddress).collect())
    }

    /// The reward claim for this target (if claimed).
    async fn reward(&self, ctx: &Context<'_>) -> Result<Option<super::reward::Reward>> {
        use std::ops::DerefMut;
        use std::sync::Arc;

        use diesel::ExpressionMethods;
        use diesel::OptionalExtension;
        use diesel::QueryDsl;
        use diesel_async::RunQueryDsl;

        use crate::db::PgReader;

        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

        type RewardRow = (Vec<u8>, i64, i64, Vec<u8>);

        let reward_row: Option<RewardRow> = soma_rewards::table
            .select((
                soma_rewards::target_id,
                soma_rewards::cp_sequence_number,
                soma_rewards::epoch,
                soma_rewards::tx_digest,
            ))
            .filter(soma_rewards::target_id.eq(&self.target_id))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let Some(row) = reward_row else {
            return Ok(None);
        };

        type BalanceRow = (Vec<u8>, i64);

        let balance_rows: Vec<BalanceRow> = soma_reward_balances::table
            .select((
                soma_reward_balances::recipient,
                soma_reward_balances::amount,
            ))
            .filter(soma_reward_balances::target_id.eq(&self.target_id))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let balances = balance_rows
            .into_iter()
            .map(|(recipient, amount)| super::reward::RewardBalance { recipient, amount })
            .collect();

        Ok(Some(super::reward::Reward {
            target_id: row.0,
            cp_sequence_number: row.1,
            epoch: row.2,
            tx_digest: row.3,
            balances,
        }))
    }
}

/// Filter for querying targets.
#[derive(InputObject, Default)]
pub struct TargetFilter {
    /// Filter by status (open, filled, claimed).
    pub status: Option<String>,
    /// Filter by generation epoch.
    pub epoch: Option<i64>,
    /// Filter by submitter address (hex with optional 0x prefix).
    pub submitter: Option<String>,
    /// Filter by winning model ID (hex with optional 0x prefix).
    pub winning_model_id: Option<String>,
    /// Filter by fill epoch.
    pub fill_epoch: Option<i64>,
    /// Filter by winning model owner (hex with optional 0x prefix).
    pub winning_model_owner: Option<String>,
    /// Filter by assigned model ID — returns targets that include this model (hex with optional 0x prefix).
    pub model_id: Option<String>,
}
