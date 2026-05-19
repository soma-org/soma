// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! SDK helpers for the on-chain provider registry.
//!
//! Two ops:
//!   - [`register`] — first-time registration. Hard-fails on chain if
//!     a Provider object already exists for the signer.
//!   - [`update`] — change the advertised endpoint.
//!
//! Plus [`get`] for a one-shot read of the provider record by
//! address. Servers call [`register_or_update`] on boot to be
//! agnostic about whether they've registered before.

use std::sync::Arc;

use types::base::SomaAddress;
use types::object::ObjectID;
use types::provider::Provider;
use types::transaction::{RegisterProviderArgs, Transaction, TransactionKind, UpdateProviderArgs};

use crate::transaction_builder::TransactionBuilder;
use crate::wallet_context::WalletContext;

/// Submit a `RegisterProvider` tx. Fails if the signer already has a
/// Provider object on-chain (call [`update`] instead in that case).
pub async fn register(
    ctx: &WalletContext,
    sender: SomaAddress,
    endpoint: String,
) -> anyhow::Result<ObjectID> {
    let kind = TransactionKind::RegisterProvider(RegisterProviderArgs { endpoint });
    let tx = build_signed(ctx, sender, kind).await?;
    ctx.execute_transaction_require_success(tx).await?;
    Ok(Provider::derive_id(sender))
}

/// Submit an `UpdateProvider` tx. Changes the advertised `endpoint`.
/// Submitting the same endpoint succeeds but has no on-chain effect —
/// there's no liveness signal on-chain to refresh.
pub async fn update(
    ctx: &WalletContext,
    sender: SomaAddress,
    endpoint: String,
) -> anyhow::Result<()> {
    let provider_id = Provider::derive_id(sender);
    let kind = TransactionKind::UpdateProvider(UpdateProviderArgs { provider_id, endpoint });
    let tx = build_signed(ctx, sender, kind).await?;
    ctx.execute_transaction_require_success(tx).await?;
    Ok(())
}

/// Read the on-chain `Provider` record for `address`, if one has
/// been registered. Returns `Ok(None)` when no Provider object
/// exists at the canonical id.
pub async fn get(ctx: &WalletContext, address: SomaAddress) -> anyhow::Result<Option<Provider>> {
    let id = Provider::derive_id(address);
    let client = ctx.get_client().await?;
    match client.get_object(id).await {
        Ok(obj) => Ok(obj.as_provider()),
        Err(e) => {
            // gRPC NotFound for an unregistered address — surfaces as
            // a Status string rather than a typed enum. Match by
            // substring rather than pulling tonic into this crate.
            let s = format!("{e}");
            if s.contains("NotFound") {
                Ok(None)
            } else {
                Err(anyhow::anyhow!("get_object {id}: {e}"))
            }
        }
    }
}

/// Idempotent register-or-update: the path a server takes on boot.
/// If no Provider object exists for `sender`, register; otherwise
/// update. The on-chain layer enforces the right tx kind for the
/// current state, so this is just a convenience wrapper that picks
/// the correct one in advance to skip the duplicate-register error.
pub async fn register_or_update(
    ctx: &WalletContext,
    sender: SomaAddress,
    endpoint: String,
) -> anyhow::Result<()> {
    if get(ctx, sender).await?.is_some() {
        update(ctx, sender, endpoint).await
    } else {
        register(ctx, sender, endpoint).await.map(|_| ())
    }
}

/// Convenience wrapper holding the wallet context + sender so callers
/// (server boot, CLI, tests) don't have to thread them through every
/// call.
pub struct ProviderClient {
    ctx: Arc<WalletContext>,
    sender: SomaAddress,
}

impl ProviderClient {
    pub fn new(ctx: Arc<WalletContext>, sender: SomaAddress) -> Self {
        Self { ctx, sender }
    }

    pub async fn register(&self, endpoint: String) -> anyhow::Result<ObjectID> {
        register(&self.ctx, self.sender, endpoint).await
    }

    pub async fn update(&self, endpoint: String) -> anyhow::Result<()> {
        update(&self.ctx, self.sender, endpoint).await
    }

    pub async fn get(&self) -> anyhow::Result<Option<Provider>> {
        get(&self.ctx, self.sender).await
    }

    pub async fn register_or_update(&self, endpoint: String) -> anyhow::Result<()> {
        register_or_update(&self.ctx, self.sender, endpoint).await
    }
}

async fn build_signed(
    ctx: &WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
) -> anyhow::Result<Transaction> {
    TransactionBuilder::new(ctx).build_transaction_async(sender, kind).await
}
