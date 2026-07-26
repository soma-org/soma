// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stable SDK surface for the per-(provider, model) on-chain `Offering`
//! shared object. Mirrors `sdk::provider` shape:
//!
//! - [`register`]           — submit `RegisterOffering`.
//! - [`update`]             — submit `UpdateOffering`.
//! - [`register_or_update`] — idempotent boot path (register else update).
//! - [`deactivate`]         — submit `DeactivateOffering` (one-way for v1).
//! - [`get`]                — read the on-chain `Offering` record.
//!
//! All four go through the same `build_signed → execute_must_succeed`
//! path as every other SDK helper.

use types::base::SomaAddress;
use types::object::ObjectID;
use types::offering::Offering;
use types::transaction::{
    DeactivateOfferingArgs, RegisterOfferingArgs, TransactionKind, UpdateOfferingArgs,
};

use crate::wallet_context::WalletContext;

/// All fields the executor cares about for a `RegisterOffering` /
/// `UpdateOffering` tx, named so callers don't have to remember the
/// arg-struct field order.
#[derive(Debug, Clone)]
pub struct OfferingPrices {
    pub prompt_micros_per_1k: u64,
    pub completion_micros_per_1k: u64,
    pub cache_read_micros_per_1k: u64,
    pub cache_write_micros_per_1k: u64,
    pub request_micros: u64,
    pub ttft_bound_ms: u32,
    pub ttot_bound_ms: u32,
}

/// Submit a `RegisterOffering` tx. `model_id` must be present in the
/// protocol-config `ModelRegistry` at the current protocol version;
/// otherwise the executor rejects with `OfferingUnknownModel`. Fails if
/// the signer already has an Offering for this `model_id` — call
/// [`update`] instead.
pub async fn register(
    ctx: &WalletContext,
    sender: SomaAddress,
    model_id: String,
    p: OfferingPrices,
) -> anyhow::Result<ObjectID> {
    let kind = TransactionKind::RegisterOffering(RegisterOfferingArgs {
        model_id: model_id.clone(),
        prompt_micros_per_1k: p.prompt_micros_per_1k,
        completion_micros_per_1k: p.completion_micros_per_1k,
        cache_read_micros_per_1k: p.cache_read_micros_per_1k,
        cache_write_micros_per_1k: p.cache_write_micros_per_1k,
        request_micros: p.request_micros,
        ttft_bound_ms: p.ttft_bound_ms,
        ttot_bound_ms: p.ttot_bound_ms,
    });
    let tx = crate::transaction_builder::TransactionBuilder::new(ctx)
        .build_transaction_async(sender, kind)
        .await?;
    ctx.execute_transaction_require_success(tx).await?;
    Ok(Offering::derive_id(sender, &model_id))
}

/// Submit an `UpdateOffering` tx. Changes prices / SLA bounds on the
/// existing `(signer, model_id)` Offering. Re-activates a previously
/// deactivated row (deactivation is one-way only for the *current*
/// state, not the row itself).
pub async fn update(
    ctx: &WalletContext,
    sender: SomaAddress,
    model_id: String,
    p: OfferingPrices,
) -> anyhow::Result<()> {
    let offering_id = Offering::derive_id(sender, &model_id);
    let kind = TransactionKind::UpdateOffering(UpdateOfferingArgs {
        offering_id,
        model_id,
        prompt_micros_per_1k: p.prompt_micros_per_1k,
        completion_micros_per_1k: p.completion_micros_per_1k,
        cache_read_micros_per_1k: p.cache_read_micros_per_1k,
        cache_write_micros_per_1k: p.cache_write_micros_per_1k,
        request_micros: p.request_micros,
        ttft_bound_ms: p.ttft_bound_ms,
        ttot_bound_ms: p.ttot_bound_ms,
    });
    let tx = crate::transaction_builder::TransactionBuilder::new(ctx)
        .build_transaction_async(sender, kind)
        .await?;
    ctx.execute_transaction_require_success(tx).await?;
    Ok(())
}

/// Idempotent register-or-update for the `(sender, model_id)` Offering —
/// the path a provider server takes on boot. Registers if no Offering
/// object exists yet, otherwise updates prices / SLA bounds in place.
/// Mirrors [`crate::provider::register_or_update`].
pub async fn register_or_update(
    ctx: &WalletContext,
    sender: SomaAddress,
    model_id: String,
    p: OfferingPrices,
) -> anyhow::Result<()> {
    if get(ctx, sender, &model_id).await?.is_some() {
        update(ctx, sender, model_id, p).await
    } else {
        register(ctx, sender, model_id, p).await.map(|_| ())
    }
}

/// Same as [`register_or_update`] but waits only for finality (effects),
/// not for checkpoint inclusion. The right choice on a provider's boot
/// path — the on-chain offering is best-effort, no downstream caller in
/// the boot sequence reads it back, and the heartbeat re-stamps the
/// provider record later. Skips the up-to-30s `wait_for_checkpoint`
/// window.
pub async fn register_or_update_no_wait(
    ctx: &WalletContext,
    sender: SomaAddress,
    model_id: String,
    p: OfferingPrices,
) -> anyhow::Result<()> {
    let kind = if get(ctx, sender, &model_id).await?.is_some() {
        let offering_id = Offering::derive_id(sender, &model_id);
        TransactionKind::UpdateOffering(UpdateOfferingArgs {
            offering_id,
            model_id,
            prompt_micros_per_1k: p.prompt_micros_per_1k,
            completion_micros_per_1k: p.completion_micros_per_1k,
            cache_read_micros_per_1k: p.cache_read_micros_per_1k,
            cache_write_micros_per_1k: p.cache_write_micros_per_1k,
            request_micros: p.request_micros,
            ttft_bound_ms: p.ttft_bound_ms,
            ttot_bound_ms: p.ttot_bound_ms,
        })
    } else {
        TransactionKind::RegisterOffering(RegisterOfferingArgs {
            model_id,
            prompt_micros_per_1k: p.prompt_micros_per_1k,
            completion_micros_per_1k: p.completion_micros_per_1k,
            cache_read_micros_per_1k: p.cache_read_micros_per_1k,
            cache_write_micros_per_1k: p.cache_write_micros_per_1k,
            request_micros: p.request_micros,
            ttft_bound_ms: p.ttft_bound_ms,
            ttot_bound_ms: p.ttot_bound_ms,
        })
    };
    let tx = crate::transaction_builder::TransactionBuilder::new(ctx)
        .build_transaction_async(sender, kind)
        .await?;
    ctx.execute_transaction_finality_only_require_success(tx).await?;
    Ok(())
}

/// Submit a `DeactivateOffering` tx. Flips `active=false` on the row;
/// the row stays on chain for audit. Channels opened against an
/// inactive offering are rejected with `ChannelOfferingMissing`.
pub async fn deactivate(
    ctx: &WalletContext,
    sender: SomaAddress,
    model_id: String,
) -> anyhow::Result<()> {
    let offering_id = Offering::derive_id(sender, &model_id);
    let kind =
        TransactionKind::DeactivateOffering(DeactivateOfferingArgs { offering_id, model_id });
    let tx = crate::transaction_builder::TransactionBuilder::new(ctx)
        .build_transaction_async(sender, kind)
        .await?;
    ctx.execute_transaction_require_success(tx).await?;
    Ok(())
}

/// Read the on-chain `Offering` for `(provider, model_id)`, if any.
/// Returns `Ok(None)` when no Offering object exists at the canonical
/// id; otherwise returns the deserialized value.
pub async fn get(
    ctx: &WalletContext,
    provider: SomaAddress,
    model_id: &str,
) -> anyhow::Result<Option<Offering>> {
    let id = Offering::derive_id(provider, model_id);
    let client = ctx.get_client().await?;
    match client.get_object(id).await {
        Ok(obj) => Ok(obj.as_offering()),
        Err(e) => {
            let s = format!("{e}");
            if s.contains("NotFound") {
                Ok(None)
            } else {
                Err(anyhow::anyhow!("get_object {id}: {e}"))
            }
        }
    }
}
