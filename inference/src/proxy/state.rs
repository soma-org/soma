// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::Mutex;
use ::types::base::SomaAddress;
use ::types::object::ObjectID;

use crate::channel::running_tab::TabClientState;
use crate::persist::{read_json, write_json};

#[derive(Clone)]
pub struct ClientStore {
    base: PathBuf,
    cache: Arc<tokio::sync::RwLock<HashMap<ObjectID, Arc<Mutex<ChannelSlot>>>>>,
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

    fn channel_path(&self, id: &ObjectID) -> PathBuf {
        self.base.join("channels").join(format!("{}.json", id))
    }

    fn pointer_path(&self, addr: &SomaAddress) -> PathBuf {
        self.base
            .join("channels-by-provider")
            .join(format!("{addr}.txt"))
    }

    pub fn read_pointer(&self, addr: &SomaAddress) -> Option<ObjectID> {
        let path = self.pointer_path(addr);
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else {
                    s.parse::<ObjectID>().ok()
                }
            }
            Err(_) => None,
        }
    }

    pub fn write_pointer(&self, addr: &SomaAddress, id: &ObjectID) -> std::io::Result<()> {
        let path = self.pointer_path(addr);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, id.to_string().as_bytes())
    }

    pub async fn slot(&self, id: &ObjectID) -> Option<Arc<Mutex<ChannelSlot>>> {
        if let Some(s) = self.cache.read().await.get(id) {
            return Some(s.clone());
        }
        let path = self.channel_path(id);
        let state: Option<TabClientState> = read_json(&path).ok().flatten();
        if let Some(state) = state {
            let slot = Arc::new(Mutex::new(ChannelSlot { state, path }));
            self.cache.write().await.insert(*id, slot.clone());
            Some(slot)
        } else {
            None
        }
    }

    pub async fn init_slot(
        &self,
        channel_id: ObjectID,
        provider_address: SomaAddress,
        provider_endpoint: &str,
        deposit_micros: u64,
    ) -> anyhow::Result<Arc<Mutex<ChannelSlot>>> {
        let path = self.channel_path(&channel_id);
        let state = TabClientState::new(
            channel_id,
            provider_address,
            provider_endpoint.to_string(),
            deposit_micros,
        );
        write_json(&path, &state).context("write client channel state")?;
        let slot = Arc::new(Mutex::new(ChannelSlot { state, path }));
        self.cache.write().await.insert(channel_id, slot.clone());
        Ok(slot)
    }
}
