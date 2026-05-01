// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-process [`Discovery`] for tests.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use types::base::SomaAddress;

use crate::chain::types::*;
use crate::chain::Discovery;
use crate::now_ms;

#[derive(Default)]
struct State {
    providers: HashMap<SomaAddress, ProviderRecord>,
    channels: HashMap<ChannelHandle, ChannelState>,
}

#[derive(Default, Clone)]
pub struct MemoryDiscovery {
    inner: Arc<RwLock<State>>,
}

impl MemoryDiscovery {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Discovery for MemoryDiscovery {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError> {
        Ok(self.inner.read().await.providers.values().cloned().collect())
    }

    async fn register_provider(&self, record: ProviderRecord) -> Result<(), ChainError> {
        self.inner
            .write()
            .await
            .providers
            .insert(record.address, record);
        Ok(())
    }

    async fn open_channel(&self, params: OpenChannelParams) -> Result<ChannelHandle, ChainError> {
        let mut s = self.inner.write().await;
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
        s.channels.insert(handle.clone(), state);
        Ok(handle)
    }

    async fn channel(&self, handle: &ChannelHandle) -> Result<ChannelState, ChainError> {
        let s = self.inner.read().await;
        let mut st = s.channels.get(handle).cloned().ok_or(ChainError::NotFound)?;
        if st.status == ChannelStatus::Open && now_ms() > st.expires_ms {
            st.status = ChannelStatus::Expired;
        }
        Ok(st)
    }
}
