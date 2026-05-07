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
//! See `proxy::router::Router::hydrate_open_channels` for the
//! cold-start flow.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use ::types::base::SomaAddress;
use ::types::object::ObjectID;

use crate::channel::running_tab::TabClientState;

#[derive(Clone, Default)]
pub struct ClientStore {
    /// Per-channel slot, keyed by `Channel` object id. Single source
    /// of truth for the proxy's view of cumulative authorization on
    /// each channel.
    slots: Arc<tokio::sync::RwLock<HashMap<ObjectID, Arc<Mutex<ChannelSlot>>>>>,
    /// Most-recent channel id used per provider address. Lets the
    /// router skip a fresh OpenChannel when an existing channel still
    /// has headroom.
    pointer: Arc<tokio::sync::RwLock<HashMap<SomaAddress, ObjectID>>>,
}

pub struct ChannelSlot {
    pub state: TabClientState,
}

impl ClientStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn read_pointer(&self, addr: &SomaAddress) -> Option<ObjectID> {
        self.pointer.read().await.get(addr).copied()
    }

    pub async fn write_pointer(&self, addr: &SomaAddress, id: &ObjectID) {
        self.pointer.write().await.insert(*addr, *id);
    }

    pub async fn slot(&self, id: &ObjectID) -> Option<Arc<Mutex<ChannelSlot>>> {
        self.slots.read().await.get(id).cloned()
    }

    /// Insert (or overwrite) a slot for `channel_id`. Used both by
    /// the lazy-open path (after a fresh OpenChannel) and by the
    /// indexer-backed cold-start hydration.
    pub async fn install_slot(
        &self,
        channel_id: ObjectID,
        provider_address: SomaAddress,
        provider_endpoint: String,
        deposit_micros: u64,
        cumulative_authorized_micros: u64,
    ) -> Arc<Mutex<ChannelSlot>> {
        let mut state = TabClientState::new(
            channel_id,
            provider_address,
            provider_endpoint,
            deposit_micros,
        );
        state.cumulative_authorized_micros = cumulative_authorized_micros;
        let slot = Arc::new(Mutex::new(ChannelSlot { state }));
        self.slots.write().await.insert(channel_id, slot.clone());
        self.pointer.write().await.insert(provider_address, channel_id);
        slot
    }
}
