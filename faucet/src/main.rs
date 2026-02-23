// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/main.rs

use clap::Parser;
use faucet::app_state::AppState;
use faucet::faucet_config::FaucetConfig;
use faucet::local_faucet::LocalFaucet;
use faucet::server::start_faucet;
use sdk::wallet_context::create_wallet_context;
use std::sync::Arc;
use tracing::info;
use types::config::soma_config_dir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = FaucetConfig::parse();

    let config_dir = match &config.config_dir {
        Some(dir) => dir.clone(),
        None => soma_config_dir()?,
    };

    info!("Starting faucet with config dir: {:?}", config_dir);

    let wallet = create_wallet_context(config.wallet_client_timeout_secs, config_dir)?;

    let faucet = LocalFaucet::new(wallet, config.clone())
        .await
        .map_err(|e| format!("Failed to initialize faucet: {e}"))?;

    let app_state = Arc::new(AppState { faucet: Arc::new(faucet), config });

    start_faucet(app_state).await?;

    Ok(())
}
