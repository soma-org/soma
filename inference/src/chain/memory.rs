// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-process [`ProviderRegistry`] for tests.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use ::types::base::SomaAddress;

use crate::chain::types::*;
use crate::chain::ProviderRegistry;

#[derive(Default)]
struct State {
    providers: HashMap<SomaAddress, ProviderRecord>,
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
impl ProviderRegistry for MemoryDiscovery {
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
}
