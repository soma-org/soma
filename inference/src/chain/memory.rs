// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-process [`ProviderRegistry`] for tests.

use std::collections::HashMap;
use std::sync::Arc;

use ::types::base::SomaAddress;
use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::chain::ProviderRegistry;
use crate::chain::types::*;

#[derive(Default)]
struct State {
    providers: HashMap<SomaAddress, ProviderRecord>,
    catalogs: HashMap<SomaAddress, Vec<crate::catalog::ModelCard>>,
}

#[derive(Default, Clone)]
pub struct MemoryDiscovery {
    inner: Arc<RwLock<State>>,
}

impl MemoryDiscovery {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider together with its model catalog. The proxy's
    /// router reads the catalog directly (no indexer needed).
    pub async fn register_with_catalog(
        &self,
        record: ProviderRecord,
        catalog: Vec<crate::catalog::ModelCard>,
    ) {
        let mut g = self.inner.write().await;
        g.catalogs.insert(record.address, catalog);
        g.providers.insert(record.address, record);
    }
}

#[async_trait]
impl ProviderRegistry for MemoryDiscovery {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError> {
        Ok(self.inner.read().await.providers.values().cloned().collect())
    }

    async fn register_provider(&self, record: ProviderRecord) -> Result<(), ChainError> {
        self.inner.write().await.providers.insert(record.address, record);
        Ok(())
    }

    async fn catalogs(&self) -> Option<HashMap<SomaAddress, Vec<crate::catalog::ModelCard>>> {
        Some(self.inner.read().await.catalogs.clone())
    }
}
