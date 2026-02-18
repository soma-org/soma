// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/faucet_config.rs

use clap::Parser;

/// Default faucet port
pub const DEFAULT_FAUCET_PORT: u16 = 9123;

/// Default amount per coin (200 SOMA in shannons)
pub const DEFAULT_AMOUNT: u64 = 200_000_000_000;

/// Default number of coins to send per request
pub const DEFAULT_NUM_COINS: usize = 5;

#[derive(Parser, Debug, Clone)]
#[clap(name = "soma-faucet", about = "Soma Faucet Server")]
pub struct FaucetConfig {
    /// Port to listen on
    #[clap(long, default_value_t = DEFAULT_FAUCET_PORT)]
    pub port: u16,

    /// Host IP to bind to
    #[clap(long, default_value = "0.0.0.0")]
    pub host_ip: String,

    /// Amount of shannons to send per coin
    #[clap(long, default_value_t = DEFAULT_AMOUNT)]
    pub amount: u64,

    /// Number of coins to send per request
    #[clap(long, default_value_t = DEFAULT_NUM_COINS)]
    pub num_coins: usize,

    /// Wallet client timeout in seconds
    #[clap(long, default_value_t = 60)]
    pub wallet_client_timeout_secs: u64,

    /// Path to the client config directory
    #[clap(long)]
    pub config_dir: Option<std::path::PathBuf>,
}

impl Default for FaucetConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_FAUCET_PORT,
            host_ip: "0.0.0.0".to_string(),
            amount: DEFAULT_AMOUNT,
            num_coins: DEFAULT_NUM_COINS,
            wallet_client_timeout_secs: 60,
            config_dir: None,
        }
    }
}
