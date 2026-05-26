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
use crate::api::types::usage::ChannelUsage;
use crate::db::PgReader;

/// Off-chain projection of a payment channel — sourced from the
/// `soma_channels` indexer table, which mirrors the on-chain
/// `Channel` shared object.
#[derive(Clone, Debug)]
pub struct Channel {
    pub channel_id: Vec<u8>,
    pub payer: Vec<u8>,
    pub payee: Vec<u8>,
    pub authorized_signer: Vec<u8>,
    pub token: String,
    pub deposit: i64,
    pub settled_amount: i64,
    pub close_requested_at_ms: Option<i64>,
    pub status: i16,
    pub opened_at_cp: i64,
    pub opened_tx_digest: Vec<u8>,
    pub last_update_cp: i64,
    /// On-chain `Channel.model_id`, frozen at `OpenChannel` time from
    /// the matching offering. Channels are scoped per
    /// `(payer, payee, model_id)` — the relay rejects requests whose
    /// payload `model` doesn't match.
    pub model_id: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Enum)]
pub enum ChannelStatus {
    Open,
    Closing,
    Withdrawn,
}

impl ChannelStatus {
    pub fn from_i16(v: i16) -> Self {
        match v {
            0 => Self::Open,
            1 => Self::Closing,
            _ => Self::Withdrawn,
        }
    }
    pub fn to_i16(self) -> i16 {
        match self {
            Self::Open => 0,
            Self::Closing => 1,
            Self::Withdrawn => 2,
        }
    }
}

#[Object]
impl Channel {
    /// On-chain `Channel` object id.
    async fn id(&self) -> SomaAddress {
        SomaAddress(self.channel_id.clone())
    }

    /// Address that opened the channel and owns the deposit.
    async fn payer(&self) -> SomaAddress {
        SomaAddress(self.payer.clone())
    }

    /// Address that receives settlements.
    async fn payee(&self) -> SomaAddress {
        SomaAddress(self.payee.clone())
    }

    /// Address whose key signs vouchers (typically equal to payer).
    async fn authorized_signer(&self) -> SomaAddress {
        SomaAddress(self.authorized_signer.clone())
    }

    /// Coin denomination — "USDC" or "SOMA".
    async fn token(&self) -> &str {
        &self.token
    }

    /// Live escrow balance. Decreases on `Settle`, increases on `TopUp`.
    /// 0 once the channel is `withdrawn`.
    async fn deposit(&self) -> BigInt {
        BigInt(self.deposit)
    }

    /// Highest cumulative voucher amount paid out so far.
    async fn settled_amount(&self) -> BigInt {
        BigInt(self.settled_amount)
    }

    /// Set when `RequestClose` was called; cleared by `TopUp`.
    async fn close_requested_at_ms(&self) -> Option<BigInt> {
        self.close_requested_at_ms.map(BigInt)
    }

    async fn status(&self) -> ChannelStatus {
        ChannelStatus::from_i16(self.status)
    }

    /// First checkpoint at which this channel appeared.
    async fn opened_at_cp(&self) -> BigInt {
        BigInt(self.opened_at_cp)
    }

    /// `OpenChannel` tx digest. Lets clients derive the channel id
    /// via `predicted_channel_id(open_tx_digest)`.
    async fn opened_tx_digest(&self) -> String {
        format!("0x{}", hex::encode(&self.opened_tx_digest))
    }

    /// Most recent checkpoint at which the channel changed.
    async fn last_update_cp(&self) -> BigInt {
        BigInt(self.last_update_cp)
    }

    /// Canonical `model_id` from the protocol-config `ModelRegistry`
    /// that this channel was opened for. Channels are per-model — a
    /// single (payer, payee) pair has one channel per offering.
    async fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Voucher-side token usage for this channel, sourced from the
    /// most recent `Settle` row in `soma_inference_settlements`.
    /// `None` when the channel has never been settled (newly opened
    /// or only topped up). Cumulative_* are monotonic per channel, so
    /// the latest row carries the channel's lifetime totals.
    async fn usage(&self, ctx: &Context<'_>) -> Result<Option<ChannelUsage>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_inference_settlements as s;

        type Row = (i64, i64, i64, i64, i64, i64, i64);

        let row: Option<Row> = s::table
            .select((
                s::cumulative_amount,
                s::cumulative_prompt_tokens,
                s::cumulative_completion_tokens,
                s::cumulative_cache_read_tokens,
                s::cumulative_cache_write_tokens,
                s::cumulative_requests,
                s::timestamp_ms,
            ))
            .filter(s::channel_id.eq(&self.channel_id))
            .order(s::tx_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(row.map(|r| ChannelUsage {
            cumulative_amount: BigInt(r.0),
            cumulative_prompt_tokens: BigInt(r.1),
            cumulative_completion_tokens: BigInt(r.2),
            cumulative_cache_read_tokens: BigInt(r.3),
            cumulative_cache_write_tokens: BigInt(r.4),
            cumulative_requests: BigInt(r.5),
            last_settlement_ms: BigInt(r.6),
        }))
    }
}
