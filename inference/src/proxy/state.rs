// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::Mutex;
use types::base::SomaAddress;

use crate::chain::types::{ChannelHandle, ChannelState};
use crate::channel::running_tab::TabClientState;
use crate::persist::{read_json, write_json};

#[derive(Clone)]
pub struct ClientStore {
    base: PathBuf,
    cache: Arc<tokio::sync::RwLock<HashMap<ChannelHandle, Arc<Mutex<ChannelSlot>>>>>,
}

pub struct ChannelSlot {
    pub state: TabClientState,
    pub path: PathBuf,
}

impl ClientStore {
    pub fn new(soma_home: PathBuf) -> Self {
        let base = soma_home.join("client");
        let _ = std::fs::create_dir_all(base.join("channels"));
        let _ = std::fs::create_dir_all(base.join("channels-by-provider"));
        Self {
            base,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    fn channel_path(&self, h: &ChannelHandle) -> PathBuf {
        self.base.join("channels").join(format!("{}.json", h.0))
    }

    fn pointer_path(&self, addr: &SomaAddress) -> PathBuf {
        self.base
            .join("channels-by-provider")
            .join(format!("{addr}.txt"))
    }

    pub fn read_pointer(&self, addr: &SomaAddress) -> Option<ChannelHandle> {
        let path = self.pointer_path(addr);
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else {
                    Some(ChannelHandle(s.to_string()))
                }
            }
            Err(_) => None,
        }
    }

    pub fn write_pointer(&self, addr: &SomaAddress, h: &ChannelHandle) -> std::io::Result<()> {
        let path = self.pointer_path(addr);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, h.0.as_bytes())
    }

    pub async fn slot(&self, h: &ChannelHandle) -> Option<Arc<Mutex<ChannelSlot>>> {
        if let Some(s) = self.cache.read().await.get(h) {
            return Some(s.clone());
        }
        let path = self.channel_path(h);
        let state: Option<TabClientState> = read_json(&path).ok().flatten();
        if let Some(state) = state {
            let slot = Arc::new(Mutex::new(ChannelSlot { state, path }));
            self.cache.write().await.insert(h.clone(), slot.clone());
            Some(slot)
        } else {
            None
        }
    }

    pub async fn init_slot(
        &self,
        chan: &ChannelState,
        provider_pubkey_hex: &str,
        provider_endpoint: &str,
    ) -> anyhow::Result<Arc<Mutex<ChannelSlot>>> {
        let path = self.channel_path(&chan.handle);
        let state = TabClientState {
            handle: chan.handle.clone(),
            provider_address: chan.provider,
            provider_pubkey_hex: provider_pubkey_hex.to_string(),
            provider_endpoint: provider_endpoint.to_string(),
            deposit_micros: chan.deposit_micros,
            cumulative_authorized_micros: 0,
            last_authorized: None,
            realized: HashMap::new(),
        };
        write_json(&path, &state).context("write client channel state")?;
        let slot = Arc::new(Mutex::new(ChannelSlot { state, path }));
        self.cache
            .write()
            .await
            .insert(chan.handle.clone(), slot.clone());
        Ok(slot)
    }
}
