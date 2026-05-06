// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider-side per-channel ledger. Stores enough to verify the next
//! voucher and to call `sdk::channel::settle` on shutdown — i.e. the
//! current cumulative + the most recent on-chain `Voucher` signature.
//!
//! The ledger does not duplicate `types::channel::Channel` state —
//! that's read fresh from chain when needed. It only persists the
//! pieces that are *off-chain* (the latest signature held by the
//! provider) so a restart can reload its IOU.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use ::types::channel::Channel;
use ::types::object::ObjectID;
use tokio::sync::Mutex;

use crate::channel::running_tab::TabProviderState;
use crate::persist::{read_json, write_json};

#[derive(Clone)]
pub struct Ledger {
    base: PathBuf,
    cache: Arc<tokio::sync::RwLock<HashMap<ObjectID, Arc<Mutex<Slot>>>>>,
}

pub struct Slot {
    pub state: TabProviderState,
    pub path: PathBuf,
}

impl Ledger {
    pub fn new(soma_home: PathBuf) -> Self {
        let base = soma_home.join("provider").join("channels");
        let _ = std::fs::create_dir_all(&base);
        Self {
            base,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    fn path_for(&self, id: &ObjectID) -> PathBuf {
        self.base.join(format!("{}.json", id))
    }

    pub async fn slot(&self, id: &ObjectID) -> Option<Arc<Mutex<Slot>>> {
        if let Some(s) = self.cache.read().await.get(id) {
            return Some(s.clone());
        }
        let path = self.path_for(id);
        let state: Option<TabProviderState> = read_json(&path).ok().flatten();
        if let Some(state) = state {
            let slot = Arc::new(Mutex::new(Slot { state, path }));
            self.cache.write().await.insert(*id, slot.clone());
            Some(slot)
        } else {
            None
        }
    }

    /// Initialize a slot from a fresh on-chain `Channel` read. Called
    /// the first time the provider sees a request for this channel.
    pub async fn init_slot(
        &self,
        id: ObjectID,
        chan: &Channel,
    ) -> anyhow::Result<Arc<Mutex<Slot>>> {
        let path = self.path_for(&id);
        let state = TabProviderState::new(id, chan);
        write_json(&path, &state).context("write provider channel state")?;
        let slot = Arc::new(Mutex::new(Slot { state, path }));
        self.cache.write().await.insert(id, slot.clone());
        Ok(slot)
    }

    pub async fn persist(&self, slot: &mut Slot) -> anyhow::Result<()> {
        write_json(&slot.path, &slot.state)?;
        Ok(())
    }

    /// Snapshot every loaded slot — used by the SIGTERM hook to
    /// settle each open channel before exit.
    pub async fn snapshot(&self) -> Vec<(ObjectID, TabProviderState)> {
        let g = self.cache.read().await;
        let mut out = Vec::with_capacity(g.len());
        for (id, slot) in g.iter() {
            let s = slot.lock().await;
            out.push((*id, s.state.clone()));
        }
        out
    }
}
