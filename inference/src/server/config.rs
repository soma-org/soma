// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;

use crate::catalog::ModelCard;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: Server,
    pub backend: Backend,
    #[serde(default)]
    pub auth: Auth,
    #[serde(default, rename = "offerings")]
    pub offerings: Vec<ModelCard>,
    #[serde(default)]
    pub auto_settle: AutoSettle,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    pub listen: String,
    pub public_endpoint: String,
}

/// Background ticker that periodically settles every channel with
/// new progress since the last submitted `Settle`. Insurance against
/// provider crashes (k8s pod restarts, OOMs, panics): every minute
/// of unsettled drift is earnings the provider would lose if the
/// process exits without running the SIGTERM hook.
///
/// Default 5 minutes — short enough to bound loss to roughly the
/// inter-tick interval, long enough that a healthy provider with
/// graceful-shutdown working still settles ~once per tick. Set
/// `interval_secs = 0` to disable (e.g. in tests where you want to
/// drive settles manually).
#[derive(Debug, Deserialize)]
pub struct AutoSettle {
    #[serde(default = "default_auto_settle_interval")]
    pub interval_secs: u64,
}

impl Default for AutoSettle {
    fn default() -> Self {
        Self { interval_secs: default_auto_settle_interval() }
    }
}

fn default_auto_settle_interval() -> u64 {
    5 * 60
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
