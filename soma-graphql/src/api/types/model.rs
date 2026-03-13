// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, SomaAddress};

/// A model registered in the Soma network's model registry.
pub struct Model {
    pub model_id: Vec<u8>,
    pub epoch: i64,
    pub status: String,
    pub owner: Vec<u8>,
    pub architecture_version: i64,
    pub commit_epoch: i64,
    pub stake: i64,
    pub commission_rate: i64,
    pub has_embedding: bool,
    pub next_epoch_commission_rate: i64,
    pub staking_pool_id: Vec<u8>,
    pub activation_epoch: Option<i64>,
    pub deactivation_epoch: Option<i64>,
    pub rewards_pool: i64,
    pub pool_token_balance: i64,
    pub pending_stake: i64,
    pub pending_total_soma_withdraw: i64,
    pub pending_pool_token_withdraw: i64,
    pub exchange_rates_json: String,
    pub manifest_url: Option<String>,
    pub manifest_checksum: Option<Vec<u8>>,
    pub manifest_size: Option<i64>,
    pub weights_commitment: Option<Vec<u8>>,
    pub embedding_commitment: Option<Vec<u8>>,
    pub decryption_key_commitment: Option<Vec<u8>>,
    pub decryption_key: Option<Vec<u8>>,
    pub has_pending_update: bool,
    pub pending_manifest_url: Option<String>,
    pub pending_manifest_checksum: Option<Vec<u8>>,
    pub pending_manifest_size: Option<i64>,
    pub pending_weights_commitment: Option<Vec<u8>>,
    pub pending_embedding_commitment: Option<Vec<u8>>,
    pub pending_decryption_key_commitment: Option<Vec<u8>>,
    pub pending_commit_epoch: Option<i64>,
}

#[Object]
impl Model {
    /// The model's ID.
    async fn model_id(&self) -> SomaAddress {
        SomaAddress(self.model_id.clone())
    }

    /// The epoch this snapshot was taken at.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// Status: created, pending, active, or inactive.
    async fn status(&self) -> &str {
        &self.status
    }

    /// The model owner's address.
    async fn owner(&self) -> SomaAddress {
        SomaAddress(self.owner.clone())
    }

    /// Protocol-versioned model architecture version.
    async fn architecture_version(&self) -> BigInt {
        BigInt(self.architecture_version)
    }

    /// The epoch when this model was committed.
    async fn commit_epoch(&self) -> BigInt {
        BigInt(self.commit_epoch)
    }

    /// The model's staked SOMA balance.
    async fn stake(&self) -> BigInt {
        BigInt(self.stake)
    }

    /// Commission rate in basis points.
    async fn commission_rate(&self) -> BigInt {
        BigInt(self.commission_rate)
    }

    /// Whether the model has revealed its embedding.
    async fn has_embedding(&self) -> bool {
        self.has_embedding
    }

    /// Next epoch's commission rate in basis points.
    async fn next_epoch_commission_rate(&self) -> BigInt {
        BigInt(self.next_epoch_commission_rate)
    }

    /// The staking pool's object ID.
    async fn staking_pool_id(&self) -> SomaAddress {
        SomaAddress(self.staking_pool_id.clone())
    }

    /// The epoch when this model's staking pool was activated.
    async fn activation_epoch(&self) -> Option<BigInt> {
        self.activation_epoch.map(BigInt)
    }

    /// The epoch when this model's staking pool was deactivated.
    async fn deactivation_epoch(&self) -> Option<BigInt> {
        self.deactivation_epoch.map(BigInt)
    }

    /// Total rewards accumulated in the staking pool.
    async fn rewards_pool(&self) -> BigInt {
        BigInt(self.rewards_pool)
    }

    /// Total pool tokens issued.
    async fn pool_token_balance(&self) -> BigInt {
        BigInt(self.pool_token_balance)
    }

    /// Pending stake awaiting processing.
    async fn pending_stake(&self) -> BigInt {
        BigInt(self.pending_stake)
    }

    /// Pending SOMA withdrawal amounts.
    async fn pending_total_soma_withdraw(&self) -> BigInt {
        BigInt(self.pending_total_soma_withdraw)
    }

    /// Pending pool token withdrawal amounts.
    async fn pending_pool_token_withdraw(&self) -> BigInt {
        BigInt(self.pending_pool_token_withdraw)
    }

    /// Historical exchange rates by epoch (JSON).
    async fn exchange_rates_json(&self) -> &str {
        &self.exchange_rates_json
    }

    /// URL where the model weights can be downloaded.
    async fn manifest_url(&self) -> Option<&str> {
        self.manifest_url.as_deref()
    }

    /// Checksum of the model weights file.
    async fn manifest_checksum(&self) -> Option<Base64> {
        self.manifest_checksum.as_ref().map(|c| Base64(c.clone()))
    }

    /// Size of the model weights file in bytes.
    async fn manifest_size(&self) -> Option<BigInt> {
        self.manifest_size.map(BigInt)
    }

    /// Blake2b commitment of the model weights.
    async fn weights_commitment(&self) -> Option<Base64> {
        self.weights_commitment.as_ref().map(|c| Base64(c.clone()))
    }

    /// Blake2b commitment of the model embedding.
    async fn embedding_commitment(&self) -> Option<Base64> {
        self.embedding_commitment
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// Blake2b commitment of the decryption key.
    async fn decryption_key_commitment(&self) -> Option<Base64> {
        self.decryption_key_commitment
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// AES-256 decryption key for encrypted weights (only revealed models).
    async fn decryption_key(&self) -> Option<Base64> {
        self.decryption_key.as_ref().map(|k| Base64(k.clone()))
    }

    /// Whether this model has a pending weight update.
    async fn has_pending_update(&self) -> bool {
        self.has_pending_update
    }

    /// URL for the pending weight update's manifest.
    async fn pending_manifest_url(&self) -> Option<&str> {
        self.pending_manifest_url.as_deref()
    }

    /// Checksum of the pending weight update file.
    async fn pending_manifest_checksum(&self) -> Option<Base64> {
        self.pending_manifest_checksum
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// Size of the pending weight update file in bytes.
    async fn pending_manifest_size(&self) -> Option<BigInt> {
        self.pending_manifest_size.map(BigInt)
    }

    /// Weights commitment for the pending update.
    async fn pending_weights_commitment(&self) -> Option<Base64> {
        self.pending_weights_commitment
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// Embedding commitment for the pending update.
    async fn pending_embedding_commitment(&self) -> Option<Base64> {
        self.pending_embedding_commitment
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// Decryption key commitment for the pending update.
    async fn pending_decryption_key_commitment(&self) -> Option<Base64> {
        self.pending_decryption_key_commitment
            .as_ref()
            .map(|c| Base64(c.clone()))
    }

    /// The epoch when the pending update was committed.
    async fn pending_commit_epoch(&self) -> Option<BigInt> {
        self.pending_commit_epoch.map(BigInt)
    }

    /// Targets assigned to this model.
    async fn targets(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
    ) -> Result<async_graphql::connection::Connection<String, super::target::Target>> {
        use std::ops::DerefMut;
        use std::sync::Arc;

        use diesel::ExpressionMethods;
        use diesel::QueryDsl;
        use diesel_async::RunQueryDsl;

        use crate::config::GraphQlConfig;
        use crate::db::PgReader;

        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first
            .unwrap_or(config.default_page_size)
            .min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{soma_target_models, soma_targets};

        // Get target IDs assigned to this model
        let target_ids: Vec<Vec<u8>> = soma_target_models::table
            .select(soma_target_models::target_id)
            .filter(soma_target_models::model_id.eq(&self.model_id))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        if target_ids.is_empty() {
            return Ok(async_graphql::connection::Connection::new(false, false));
        }

        type RowA = (
            Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>,
            i64, i64, i32,
        );
        type RowB = (
            Option<f64>, Option<f64>, Option<Vec<u8>>, Option<i64>,
            f64, String, Option<String>, Option<Vec<u8>>, Option<i64>,
        );

        let after_cp: Option<i64> = match &after {
            Some(cursor) => {
                let parts: Vec<&str> = cursor.splitn(2, ':').collect();
                if parts.len() != 2 {
                    return Err(Error::new("Invalid cursor format"));
                }
                Some(parts[1].parse().map_err(|e| Error::new(format!("Invalid cursor: {e}")))?)
            }
            None => None,
        };

        let mut query = soma_targets::table
            .select((
                (
                    soma_targets::target_id,
                    soma_targets::cp_sequence_number,
                    soma_targets::epoch,
                    soma_targets::status,
                    soma_targets::submitter,
                    soma_targets::winning_model_id,
                    soma_targets::reward_pool,
                    soma_targets::bond_amount,
                    soma_targets::report_count,
                ),
                (
                    soma_targets::winning_distance_score,
                    soma_targets::winning_loss_score,
                    soma_targets::winning_model_owner,
                    soma_targets::fill_epoch,
                    soma_targets::distance_threshold,
                    soma_targets::model_ids_json,
                    soma_targets::winning_data_url,
                    soma_targets::winning_data_checksum,
                    soma_targets::winning_data_size,
                ),
            ))
            .filter(soma_targets::target_id.eq_any(&target_ids))
            .order(soma_targets::cp_sequence_number.desc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(acp) = after_cp {
            query = query.filter(soma_targets::cp_sequence_number.lt(acp));
        }

        let results: Vec<(RowA, RowB)> = query
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = async_graphql::connection::Connection::new(has_previous, has_next);
        for (a, b) in nodes {
            let cursor = format!("{}:{}", hex::encode(&a.0), a.1);
            connection.edges.push(async_graphql::connection::Edge::new(
                cursor,
                super::target::Target {
                    target_id: a.0,
                    cp_sequence_number: a.1,
                    epoch: a.2,
                    status: a.3,
                    submitter: a.4,
                    winning_model_id: a.5,
                    reward_pool: a.6,
                    bond_amount: a.7,
                    report_count: a.8,
                    winning_distance_score: b.0,
                    winning_loss_score: b.1,
                    winning_model_owner: b.2,
                    fill_epoch: b.3,
                    distance_threshold: b.4,
                    model_ids_json: b.5,
                    winning_data_url: b.6,
                    winning_data_checksum: b.7,
                    winning_data_size: b.8,
                },
            ));
        }

        Ok(connection)
    }
}

/// Filter for querying models.
#[derive(InputObject, Default)]
pub struct ModelFilter {
    /// Filter by status (created, pending, active, inactive).
    pub status: Option<String>,
    /// Filter by owner address (hex with optional 0x prefix).
    pub owner: Option<String>,
    /// Filter by whether the model has an embedding.
    pub has_embedding: Option<bool>,
    /// Filter by minimum stake (inclusive).
    pub min_stake: Option<i64>,
    /// Filter by maximum stake (inclusive).
    pub max_stake: Option<i64>,
}
