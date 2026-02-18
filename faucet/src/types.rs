// Portions Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/types.rs

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FaucetRequest {
    FixedAmountRequest { recipient: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FaucetResponse {
    pub status: RequestStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coins_sent: Option<Vec<CoinInfo>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RequestStatus {
    Success,
    Failure(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoinInfo {
    pub amount: u64,
    pub id: String,
    pub transfer_tx_digest: String,
}
