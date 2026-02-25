// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/app_state.rs

use std::sync::Arc;

use crate::faucet_config::FaucetConfig;
use crate::local_faucet::LocalFaucet;

pub struct AppState {
    pub faucet: Arc<LocalFaucet>,
    pub config: FaucetConfig,
}
