// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/errors.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FaucetError {
    #[error("Wallet error: {0}")]
    Wallet(String),

    #[error("Transfer error: {0}")]
    Transfer(String),

    #[error("Internal error: {0}")]
    Internal(String),
}
