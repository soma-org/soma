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
    #[serde(default)]
    pub autoprice: AutoPrice,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    /// The provider's advertised endpoint string recorded on-chain. With
    /// iroh-only serving this is informational (buyers dial the iroh
    /// `EndpointId`); kept for the on-chain `Provider.endpoint` record.
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
    /// llama.cpp only: number of parallel decode slots the local
    /// `llama-server` was launched with (`--parallel N`). The saturation
    /// metric divides in-flight + queued requests by this to land in
    /// `[0,1]`. Absent ⇒ treated as `1`.
    #[serde(default)]
    pub slots: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Auth {
    #[serde(default = "default_skew")]
    pub clock_skew_tolerance_secs: u64,
}

fn default_skew() -> u64 {
    60
}

/// Utilization-targeting price controller. When enabled (and the backend
/// reports a saturation metric — i.e. the local llama.cpp backend), a
/// background loop nudges the provider's on-chain offering price up or
/// down by a fixed fraction each tick to keep the device near a target
/// utilization band. Raising price sheds load when saturated; lowering it
/// (down to `min_*`) attracts load when idle.
///
/// Disabled by default — a provider on a remote backend with no saturation
/// signal, or one that prefers a fixed price, leaves the boot offering
/// untouched.
#[derive(Debug, Deserialize)]
pub struct AutoPrice {
    #[serde(default)]
    pub enabled: bool,
    /// Seconds between price-adjustment ticks ("every couple minutes").
    #[serde(default = "default_autoprice_interval")]
    pub interval_secs: u64,
    /// Below this saturation the device is under-loaded → lower price.
    #[serde(default = "default_target_low")]
    pub target_low: f64,
    /// Above this saturation the device is over-loaded → raise price.
    #[serde(default = "default_target_high")]
    pub target_high: f64,
    /// Multiplicative step per tick (e.g. `0.1` = ±10%).
    #[serde(default = "default_step_frac")]
    pub step_frac: f64,
    /// Price floor for prompt tokens, µUSDC per 1k. The configurable
    /// `min_price` — the controller never prices below this.
    #[serde(default)]
    pub min_prompt_micros_per_1k: u64,
    /// Price floor for completion tokens, µUSDC per 1k.
    #[serde(default)]
    pub min_completion_micros_per_1k: u64,
    /// Optional ceiling for completion tokens, µUSDC per 1k.
    #[serde(default)]
    pub max_completion_micros_per_1k: Option<u64>,
}

impl Default for AutoPrice {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_secs: default_autoprice_interval(),
            target_low: default_target_low(),
            target_high: default_target_high(),
            step_frac: default_step_frac(),
            min_prompt_micros_per_1k: 0,
            min_completion_micros_per_1k: 0,
            max_completion_micros_per_1k: None,
        }
    }
}

fn default_autoprice_interval() -> u64 {
    120
}
fn default_target_low() -> f64 {
    0.65
}
fn default_target_high() -> f64 {
    0.85
}
fn default_step_frac() -> f64 {
    0.10
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Config> {
    let s = std::fs::read_to_string(path)?;
    let cfg: Config = toml::from_str(&s)?;
    Ok(cfg)
}
