// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::Mutex;

use crate::chain::types::ChannelHandle;
use crate::channel::running_tab::TabProviderState;
use crate::persist::{read_json, write_json};

#[derive(Clone)]
pub struct Ledger {
    base: PathBuf,
    cache: Arc<tokio::sync::RwLock<HashMap<ChannelHandle, Arc<Mutex<Slot>>>>>,
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

    fn path_for(&self, h: &ChannelHandle) -> PathBuf {
        self.base.join(format!("{}.json", h.0))
    }

    pub async fn slot(&self, h: &ChannelHandle) -> Option<Arc<Mutex<Slot>>> {
        if let Some(s) = self.cache.read().await.get(h) {
            return Some(s.clone());
        }
        let path = self.path_for(h);
        let state: Option<TabProviderState> = read_json(&path).ok().flatten();
        if let Some(state) = state {
            let slot = Arc::new(Mutex::new(Slot { state, path }));
            self.cache.write().await.insert(h.clone(), slot.clone());
            Some(slot)
        } else {
            None
        }
    }

    pub async fn init_slot(
        &self,
        ch: &crate::chain::types::ChannelState,
    ) -> anyhow::Result<Arc<Mutex<Slot>>> {
        let path = self.path_for(&ch.handle);
        let state = TabProviderState {
            handle: ch.handle.clone(),
            client_address: ch.client,
            client_pubkey_hex: ch.client_pubkey_hex.clone(),
            deposit_micros: ch.deposit_micros,
            cumulative_authorized_micros: 0,
            total_consumed_micros: 0,
            last_signature_b64: String::new(),
            last_request_id: None,
        };
        write_json(&path, &state).context("write provider channel state")?;
        let slot = Arc::new(Mutex::new(Slot { state, path }));
        self.cache.write().await.insert(ch.handle.clone(), slot.clone());
        Ok(slot)
    }

    pub async fn persist(&self, slot: &mut Slot) -> anyhow::Result<()> {
        write_json(&slot.path, &slot.state)?;
        Ok(())
    }
}
