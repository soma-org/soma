// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Filesystem-backed [`ProviderRegistry`] for development. Channels
//! moved on-chain; provider records will follow in the next PR.

use std::path::PathBuf;

use async_trait::async_trait;
use types::base::SomaAddress;

use crate::chain::types::*;
use crate::chain::ProviderRegistry;
use crate::persist::{read_json, write_json, DirLock};

pub struct LocalDiscovery {
    base: PathBuf,
}

impl LocalDiscovery {
    pub fn new(base: impl Into<PathBuf>) -> std::io::Result<Self> {
        let base = base.into();
        std::fs::create_dir_all(base.join("providers"))?;
        Ok(Self { base })
    }

    fn lock(&self) -> std::io::Result<DirLock> {
        DirLock::acquire(&self.base)
    }

    fn provider_path(&self, addr: &SomaAddress) -> PathBuf {
        self.base.join("providers").join(format!("{addr}.json"))
    }
}

#[async_trait]
impl ProviderRegistry for LocalDiscovery {
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
}
