// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::Deserialize;

use crate::catalog::ModelCard;
use crate::persist::expand_home;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: Server,
    pub chain: Chain,
    pub backend: Backend,
    #[serde(default)]
    pub auth: Auth,
    #[serde(default, rename = "offerings")]
    pub offerings: Vec<ModelCard>,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    pub listen: String,
    pub public_endpoint: String,
}

#[derive(Debug, Deserialize)]
pub struct Chain {
    pub mode: String,
    pub soma_home: String,
    #[serde(default = "default_heartbeat")]
    pub heartbeat_interval_secs: u64,
}

fn default_heartbeat() -> u64 {
    600
}

impl Chain {
    pub fn soma_home_path(&self) -> PathBuf {
        expand_home(&self.soma_home)
    }
}

#[derive(Debug, Deserialize)]
pub struct Backend {
    pub kind: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub upstream_url: Option<String>,
    #[serde(default)]
    pub endpoint_name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Auth {
    #[serde(default = "default_skew")]
    pub clock_skew_tolerance_secs: u64,
}

fn default_skew() -> u64 {
    60
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Config> {
    let s = std::fs::read_to_string(path)?;
    let cfg: Config = toml::from_str(&s)?;
    Ok(cfg)
}
