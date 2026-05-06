// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Chain-backed [`super::ChannelSurface`] — opens / reads / settles
//! `types::channel::Channel` objects via the SDK's `WalletContext`.
//!
//! Reads go through the gRPC fullnode (`get_object`); writes go
//! through `sdk::channel::{open_channel, settle, top_up,
//! request_close, withdraw_after_timeout}`.

use std::sync::Arc;

use async_trait::async_trait;
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::channel::{Channel, Voucher};
use types::crypto::GenericSignature;
use types::object::{CoinType, ObjectID};

use super::{ChainError, ChannelSurface};

pub struct ChainChannelSurface {
    ctx: Arc<WalletContext>,
    /// Address that signs both the on-chain ops *and* the off-chain
    /// HTTP vouchers. For the demo it's the same address that opens
    /// the channel (i.e. the payer).
    signer: SomaAddress,
}

impl ChainChannelSurface {
    pub fn new(ctx: Arc<WalletContext>, signer: SomaAddress) -> Self {
        Self { ctx, signer }
    }
}

#[async_trait]
impl ChannelSurface for ChainChannelSurface {
    async fn open(
        &self,
        payee: SomaAddress,
        coin_type: CoinType,
        deposit_amount: u64,
    ) -> Result<ObjectID, ChainError> {
        sdk::channel::open_channel(
            &self.ctx,
            self.signer,
            payee,
            self.signer, // authorized_signer == payer for the demo
            coin_type,
            deposit_amount,
        )
        .await
        .map_err(|e| ChainError::Tx(format!("open_channel: {e}")))
    }

    async fn get(&self, id: ObjectID) -> Result<Channel, ChainError> {
        let client = self
            .ctx
            .get_client()
            .await
            .map_err(|e| ChainError::Rpc(format!("get_client: {e}")))?;
        let obj = client.get_object(id).await.map_err(|e| {
            // Status::Display formats as "status: <Code>, message: ...".
            // Match on the code via a string contains rather than the
            // tonic enum to avoid pulling tonic into this crate's deps.
            let s = format!("{e}");
            if s.contains("NotFound") {
                ChainError::NotFound
            } else {
                ChainError::Rpc(format!("get_object {id}: {e}"))
            }
        })?;
        obj.as_channel()
            .ok_or_else(|| ChainError::Invalid(format!("{id} is not a Channel object")))
    }

    async fn settle(
        &self,
        voucher: Voucher,
        sig: GenericSignature,
    ) -> Result<(), ChainError> {
        sdk::channel::settle(&self.ctx, self.signer, voucher, sig)
            .await
            .map_err(|e| ChainError::Tx(format!("settle: {e}")))
    }

    async fn top_up(
        &self,
        id: ObjectID,
        coin_type: CoinType,
        amount: u64,
    ) -> Result<(), ChainError> {
        sdk::channel::top_up(&self.ctx, self.signer, id, coin_type, amount)
            .await
            .map_err(|e| ChainError::Tx(format!("top_up: {e}")))
    }

    async fn request_close(&self, id: ObjectID) -> Result<(), ChainError> {
        sdk::channel::request_close(&self.ctx, self.signer, id)
            .await
            .map_err(|e| ChainError::Tx(format!("request_close: {e}")))
    }

    async fn withdraw_after_timeout(&self, id: ObjectID) -> Result<(), ChainError> {
        sdk::channel::withdraw_after_timeout(&self.ctx, self.signer, id)
            .await
            .map_err(|e| ChainError::Tx(format!("withdraw_after_timeout: {e}")))
    }

    fn signer_address(&self) -> SomaAddress {
        self.signer
    }
}
