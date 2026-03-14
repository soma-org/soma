// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::DerefMut;
use std::sync::Arc;

use async_graphql::connection::{Connection, Edge};
use async_graphql::{Context, Error, Object, Result};
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;

use crate::api::scalars::{BigInt, Digest, SomaAddress};
use crate::db::PgReader;

/// An address on the Soma network. Used to look up transactions involving this address.
pub struct Address {
    pub address: Vec<u8>,
}

#[Object]
impl Address {
    /// The address (hex with 0x prefix).
    async fn address(&self) -> SomaAddress {
        SomaAddress(self.address.clone())
    }

    /// Transactions that affected this address, paginated by sequence number.
    async fn transactions(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
    ) -> Result<Connection<String, TransactionRef>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let limit = first.unwrap_or(20).min(50) as i64;
        let after_seq: i64 = after
            .as_deref()
            .map(|s| s.parse::<i64>())
            .transpose()
            .map_err(|e| Error::new(format!("Invalid cursor: {e}")))?
            .unwrap_or(-1);

        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::tx_affected_addresses;
        use indexer_alt_schema::schema::tx_digests;

        // Step 1: Get tx sequence numbers for this address
        let seq_nums: Vec<i64> = tx_affected_addresses::table
            .select(tx_affected_addresses::tx_sequence_number)
            .filter(tx_affected_addresses::affected.eq(&self.address))
            .filter(tx_affected_addresses::tx_sequence_number.gt(after_seq))
            .order(tx_affected_addresses::tx_sequence_number.asc())
            .limit(limit + 1)
            .load::<i64>(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let has_next = seq_nums.len() as i64 > limit;
        let seq_nums: Vec<i64> = seq_nums.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        // Step 2: Batch-fetch digests for those sequence numbers
        let digests: Vec<(i64, Vec<u8>)> = if seq_nums.is_empty() {
            vec![]
        } else {
            tx_digests::table
                .select((tx_digests::tx_sequence_number, tx_digests::tx_digest))
                .filter(tx_digests::tx_sequence_number.eq_any(&seq_nums))
                .load::<(i64, Vec<u8>)>(conn.deref_mut())
                .await
                .map_err(|e| Error::new(e.to_string()))?
        };

        // Build a map for fast lookup
        let digest_map: std::collections::HashMap<i64, Vec<u8>> = digests.into_iter().collect();

        let mut connection = Connection::new(has_previous, has_next);
        for seq in seq_nums {
            if let Some(digest) = digest_map.get(&seq) {
                connection.edges.push(Edge::new(
                    seq.to_string(),
                    TransactionRef { tx_sequence_number: seq, tx_digest: digest.clone() },
                ));
            }
        }

        Ok(connection)
    }
}

/// A lightweight reference to a transaction (digest + sequence number).
pub struct TransactionRef {
    pub tx_sequence_number: i64,
    pub tx_digest: Vec<u8>,
}

#[Object]
impl TransactionRef {
    async fn digest(&self) -> Digest {
        Digest(self.tx_digest.clone())
    }

    async fn sequence_number(&self) -> BigInt {
        BigInt(self.tx_sequence_number)
    }
}
