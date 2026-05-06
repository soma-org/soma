// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Proxy runtime config — built from CLI flags rather than a TOML
//! file. Only chain-agnostic knobs live here (listen address,
//! default deposit, provider cache TTL).

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: String,
    pub default_deposit_micros: u64,
    pub provider_cache_ttl_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:11434".to_string(),
            default_deposit_micros: 5_000_000,
            provider_cache_ttl_secs: 30,
        }
    }
}
