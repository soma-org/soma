// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, DateTime, SomaAddress};

/// A bridge deposit minted on Soma in response to a Base-side
/// `TokensDeposited` event. One row per `BridgeDeposit` tx —
/// see `soma_bridge_deposits` table + handler.
pub struct BridgeDeposit {
    pub tx_sequence_number: i64,
    pub cp_sequence_number: i64,
    pub recipient: Vec<u8>,
    pub amount: i64,
    pub nonce: i64,
    pub eth_tx_hash: Vec<u8>,
    pub timestamp_ms: i64,
}

#[Object]
impl BridgeDeposit {
    /// Tx sequence number on Soma (monotonic, unique per tx).
    async fn tx_sequence_number(&self) -> BigInt {
        BigInt(self.tx_sequence_number)
    }

    /// Checkpoint sequence number that included the mint tx.
    async fn cp_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// 32-byte Soma recipient that received the minted USDC.
    async fn recipient(&self) -> SomaAddress {
        SomaAddress(self.recipient.clone())
    }

    /// USDC amount minted, in micros (6-decimal base units).
    async fn amount(&self) -> BigInt {
        BigInt(self.amount)
    }

    /// L1 nonce assigned by the Base bridge contract. Monotonic per
    /// (source bridge contract, message type).
    async fn nonce(&self) -> BigInt {
        BigInt(self.nonce)
    }

    /// Base-side originating tx hash, `0x`-prefixed.
    async fn eth_tx_hash(&self) -> String {
        format!("0x{}", hex::encode(&self.eth_tx_hash))
    }

    /// Soma checkpoint timestamp of the mint.
    async fn timestamp(&self) -> DateTime {
        DateTime(self.timestamp_ms)
    }
}
