// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-memory proxy-side channel state.
//!
//! The proxy is **stateless on disk** — there is no `~/.soma/.../proxy/`
//! tree any more. On cold start the router rehydrates channel
//! pointers from the indexer (`channels(payer = me, status = open)`)
//! and per-channel `cumulative_authorized` from the provider's
//! `/soma/channel/{id}` endpoint. The chain's `Channel.settled_amount`
//! is the floor; the provider's last-held cumulative is the floor for
//! the proxy's next signed voucher (so we never issue a non-monotonic
//! voucher after a restart).
//!
//! Channels are scoped per `(payer, provider, model_id)` — the on-chain
//! `Channel.model_id` is frozen at OpenChannel time and the provider
//! relay rejects mismatched requests, so the proxy must keep distinct
//! pointers per model.
//!
//! See `proxy::router::Router::hydrate_open_channels` for the
//! cold-start flow.

use std::collections::HashMap;
use std::sync::Arc;

use ::types::base::SomaAddress;
use ::types::object::ObjectID;
use tokio::sync::Mutex;

use crate::channel::running_tab::TabClientState;

/// Cache key for the per-channel pointer: `(provider_address, model_id)`.
/// Matches the on-chain channel scope.
pub type ProviderModelKey = (SomaAddress, String);

#[derive(Clone, Default)]
pub struct ClientStore {
    /// Per-channel slot, keyed by `Channel` object id. Single source
    /// of truth for the proxy's view of cumulative authorization on
    /// each channel.
    slots: Arc<tokio::sync::RwLock<HashMap<ObjectID, Arc<Mutex<ChannelSlot>>>>>,
    /// Most-recent channel id used per `(provider, model)`. Lets the
    /// router skip a fresh OpenChannel when an existing channel for
    /// the same model still has headroom.
    pointer: Arc<tokio::sync::RwLock<HashMap<ProviderModelKey, ObjectID>>>,
}

pub struct ChannelSlot {
    pub state: TabClientState,
}

impl ClientStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn read_pointer(&self, addr: &SomaAddress, model_id: &str) -> Option<ObjectID> {
        self.pointer.read().await.get(&(*addr, model_id.to_string())).copied()
    }

    pub async fn write_pointer(&self, addr: &SomaAddress, model_id: &str, id: &ObjectID) {
        self.pointer.write().await.insert((*addr, model_id.to_string()), *id);
    }

    pub async fn slot(&self, id: &ObjectID) -> Option<Arc<Mutex<ChannelSlot>>> {
        self.slots.read().await.get(id).cloned()
    }

    /// Snapshot every installed slot's `(deposit - cumulative_authorized)`
    /// headroom. The `/health` endpoint reduces this to
    /// `(channels_open, balance_micros)`. Saturating-sub guards against
    /// the (impossible-in-practice) post-reconcile case where the
    /// cumulative briefly exceeds the deposit during a slot update.
    pub async fn channel_headrooms(&self) -> Vec<u64> {
        let slots = self.slots.read().await;
        let mut out = Vec::with_capacity(slots.len());
        for slot in slots.values() {
            let g = slot.lock().await;
            out.push(g.state.deposit_micros.saturating_sub(g.state.cumulative_authorized_micros));
        }
        out
    }

    /// Insert (or overwrite) a slot for `channel_id`. Used both by
    /// the lazy-open path (after a fresh OpenChannel) and by the
    /// indexer-backed cold-start hydration.
    pub async fn install_slot(
        &self,
        channel_id: ObjectID,
        provider_address: SomaAddress,
        provider_endpoint: String,
        model_id: String,
        deposit_micros: u64,
        cumulative_authorized_micros: u64,
    ) -> Arc<Mutex<ChannelSlot>> {
        let mut state = TabClientState::new(
            channel_id,
            provider_address,
            provider_endpoint,
            model_id.clone(),
            deposit_micros,
        );
        state.cumulative_authorized_micros = cumulative_authorized_micros;
        let slot = Arc::new(Mutex::new(ChannelSlot { state }));
        self.slots.write().await.insert(channel_id, slot.clone());
        self.pointer.write().await.insert((provider_address, model_id), channel_id);
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr_from_byte(b: u8) -> SomaAddress {
        let mut bytes = [0u8; 32];
        bytes[31] = b;
        SomaAddress::from(bytes)
    }
    fn id_from_byte(b: u8) -> ObjectID {
        let mut bytes = [0u8; 32];
        bytes[31] = b;
        ObjectID::new(bytes)
    }

    #[tokio::test]
    async fn pointer_scope_is_per_provider_and_model() {
        let store = ClientStore::new();
        let provider = addr_from_byte(0xAB);
        let other_provider = addr_from_byte(0xCD);
        let chan_haiku = id_from_byte(0x01);
        let chan_opus = id_from_byte(0x02);
        let chan_other = id_from_byte(0x03);

        store
            .install_slot(
                chan_haiku,
                provider,
                "http://prov:8444".to_string(),
                "anthropic/claude-haiku-4.5".to_string(),
                5_000_000,
                0,
            )
            .await;
        store
            .install_slot(
                chan_opus,
                provider,
                "http://prov:8444".to_string(),
                "anthropic/claude-opus-4.7".to_string(),
                5_000_000,
                0,
            )
            .await;
        store
            .install_slot(
                chan_other,
                other_provider,
                "http://other:8444".to_string(),
                "anthropic/claude-haiku-4.5".to_string(),
                5_000_000,
                0,
            )
            .await;

        // Same provider, different models — distinct channels.
        assert_eq!(
            store.read_pointer(&provider, "anthropic/claude-haiku-4.5").await,
            Some(chan_haiku),
        );
        assert_eq!(
            store.read_pointer(&provider, "anthropic/claude-opus-4.7").await,
            Some(chan_opus),
        );
        // Same model, different provider — distinct channels.
        assert_eq!(
            store.read_pointer(&other_provider, "anthropic/claude-haiku-4.5").await,
            Some(chan_other),
        );
        // Unknown (provider, model) — no pointer.
        assert_eq!(store.read_pointer(&provider, "openai/gpt-5.4-nano").await, None);
        assert_eq!(
            store.read_pointer(&addr_from_byte(0xFF), "anthropic/claude-haiku-4.5").await,
            None,
        );
    }

    #[tokio::test]
    async fn installed_slot_records_bound_model_id() {
        let store = ClientStore::new();
        let provider = addr_from_byte(0xAB);
        let chan = id_from_byte(0x01);
        let slot = store
            .install_slot(
                chan,
                provider,
                "http://prov:8444".to_string(),
                "google/gemini-3.1-flash-lite".to_string(),
                1_000_000,
                42,
            )
            .await;
        let g = slot.lock().await;
        assert_eq!(g.state.model_id, "google/gemini-3.1-flash-lite");
        assert_eq!(g.state.cumulative_authorized_micros, 42);
    }
}
