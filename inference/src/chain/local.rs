// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Filesystem-backed [`Discovery`] for development. Replaces with on-chain
//! `Offering` query + `RegisterProvider` / `Heartbeat` transactions later.

use std::path::PathBuf;

use async_trait::async_trait;
use types::base::SomaAddress;

use crate::chain::types::*;
use crate::chain::Discovery;
use crate::now_ms;
use crate::persist::{read_json, write_json, DirLock};

pub struct LocalDiscovery {
    base: PathBuf,
}

impl LocalDiscovery {
    pub fn new(base: impl Into<PathBuf>) -> std::io::Result<Self> {
        let base = base.into();
        std::fs::create_dir_all(base.join("providers"))?;
        std::fs::create_dir_all(base.join("channels"))?;
        Ok(Self { base })
    }

    fn lock(&self) -> std::io::Result<DirLock> {
        DirLock::acquire(&self.base)
    }

    fn provider_path(&self, addr: &SomaAddress) -> PathBuf {
        self.base.join("providers").join(format!("{addr}.json"))
    }

    fn channel_path(&self, handle: &ChannelHandle) -> PathBuf {
        self.base.join("channels").join(format!("{}.json", handle.0))
    }
}

#[async_trait]
impl Discovery for LocalDiscovery {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError> {
        let dir = self.base.join("providers");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            if let Some(rec) = read_json::<ProviderRecord>(&entry.path())? {
                out.push(rec);
            }
        }
        Ok(out)
    }

    async fn register_provider(&self, record: ProviderRecord) -> Result<(), ChainError> {
        let path = self.provider_path(&record.address);
        let _g = self.lock()?;
        write_json(&path, &record)?;
        Ok(())
    }

    async fn open_channel(&self, params: OpenChannelParams) -> Result<ChannelHandle, ChainError> {
        let _g = self.lock()?;
        let handle = ChannelHandle(uuid::Uuid::new_v4().simple().to_string());
        let state = ChannelState {
            handle: handle.clone(),
            client: params.client,
            client_pubkey_hex: params.client_pubkey_hex,
            provider: params.provider,
            deposit_micros: params.deposit_micros,
            status: ChannelStatus::Open,
            opened_ms: now_ms(),
            expires_ms: params.expires_ms,
        };
        write_json(&self.channel_path(&handle), &state)?;
        Ok(handle)
    }

    async fn channel(&self, handle: &ChannelHandle) -> Result<ChannelState, ChainError> {
        let path = self.channel_path(handle);
        let mut state: ChannelState = read_json(&path)?.ok_or(ChainError::NotFound)?;
        if state.status == ChannelStatus::Open && now_ms() > state.expires_ms {
            state.status = ChannelStatus::Expired;
        }
        Ok(state)
    }
}
