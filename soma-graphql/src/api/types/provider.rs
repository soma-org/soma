// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::DerefMut;
use std::sync::Arc;

use async_graphql::*;
use diesel::ExpressionMethods;
use diesel::OptionalExtension;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;

use crate::api::scalars::{BigInt, SomaAddress};
use crate::db::PgReader;

/// Off-chain projection of a provider record — sourced from the
/// `soma_providers` indexer table mirroring the on-chain `Provider`
/// shared object.
#[derive(Clone, Debug)]
pub struct Provider {
    pub address: Vec<u8>,
    pub endpoint: String,
    pub last_update_cp: i64,
}

/// Reputation aggregates derived from the on-chain channel history.
/// Indexer-published, observation-only — nothing on-chain is keyed
/// on these. Buyers consume them as one routing input among many.
#[derive(Clone, Debug, SimpleObject)]
pub struct ProviderReputation {
    /// Sum of `Settle` deltas to this provider over the last 30 days
    /// (chain-time).
    pub volume_settled_30d: BigInt,
    /// Distinct payer addresses that settled to this provider in the
    /// last 30 days.
    pub distinct_buyers_30d: BigInt,
    /// `count(TopUp) / count(OpenChannel)` for this provider, all-time.
    /// > 0 means buyers chose to refund vs. walk away.
    pub channel_renewal_rate: f64,
    /// Average `(last_update_cp - opened_at_cp)` across this
    /// provider's channels (excluding withdrawn). Loose proxy for
    /// "how long buyers stick around."
    pub mean_channel_span_cps: BigInt,
    /// Volume-weighted fraction of recently-rated channels this
    /// provider's buyers flagged negative. `None` when no in-window
    /// ratings on positive-volume channels (consumers should treat
    /// "no signal" rather than `0.0`).
    pub negative_rate_30d: Option<f64>,
    /// Number of distinct channels rated in the 30d window.
    /// Confidence proxy alongside `negative_rate_30d`.
    pub rating_count_30d: BigInt,
    /// Versioned with the formula. Consumers compare to detect
    /// signal-definition changes without reading source.
    pub signal_version: i32,
}

#[Object]
impl Provider {
    /// Provider's address.
    async fn address(&self) -> SomaAddress {
        SomaAddress(self.address.clone())
    }

    /// HTTP(S) endpoint for `/soma/info` and `/v1/...` APIs.
    async fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Most recent checkpoint at which the provider record changed.
    async fn last_update_cp(&self) -> BigInt {
        BigInt(self.last_update_cp)
    }

    /// Reputation aggregates — derived from on-chain channel history
    /// by the `provider_reputation` SQL view. `None` if the provider
    /// has no channels yet.
    async fn reputation(&self, ctx: &Context<'_>) -> Result<Option<ProviderReputation>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::provider_reputation;

        type Row = (Vec<u8>, i64, i64, f64, i64, Option<f64>, i64, i32);
        let row: Option<Row> = provider_reputation::table
            .select((
                provider_reputation::address,
                provider_reputation::volume_settled_30d,
                provider_reputation::distinct_buyers_30d,
                provider_reputation::channel_renewal_rate,
                provider_reputation::mean_channel_span_cps,
                provider_reputation::negative_rate_30d,
                provider_reputation::rating_count_30d,
                provider_reputation::signal_version,
            ))
            .filter(provider_reputation::address.eq(&self.address))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(row.map(|r| ProviderReputation {
            volume_settled_30d: BigInt(r.1),
            distinct_buyers_30d: BigInt(r.2),
            channel_renewal_rate: r.3,
            mean_channel_span_cps: BigInt(r.4),
            negative_rate_30d: r.5,
            rating_count_30d: BigInt(r.6),
            signal_version: r.7,
        }))
    }
}
