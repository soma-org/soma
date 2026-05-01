// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::Deserialize;

use crate::persist::expand_home;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub listen: Listen,
    pub chain: Chain,
    pub wallet: Wallet,
    #[serde(default)]
    pub discovery: Discovery,
}

#[derive(Debug, Deserialize)]
pub struct Listen {
    pub addr: String,
}

#[derive(Debug, Deserialize)]
pub struct Chain {
    pub mode: String,
    pub soma_home: String,
}

impl Chain {
    pub fn soma_home_path(&self) -> PathBuf {
        expand_home(&self.soma_home)
    }
}

#[derive(Debug, Deserialize)]
pub struct Wallet {
    pub default_deposit_micros: u64,
    #[serde(default = "default_expires")]
    pub channel_expires_secs: u64,
}

fn default_expires() -> u64 {
    86_400
}

#[derive(Debug, Deserialize, Default)]
pub struct Discovery {
    #[serde(default = "default_ttl")]
    pub provider_cache_ttl_secs: u64,
}

fn default_ttl() -> u64 {
    30
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Config> {
    let s = std::fs::read_to_string(path)?;
    let cfg: Config = toml::from_str(&s)?;
    Ok(cfg)
}
