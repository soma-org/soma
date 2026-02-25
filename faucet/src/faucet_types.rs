// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct GasRequest {
    pub recipient: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GasResponse {
    pub status: String,
    #[serde(default)]
    pub coins_sent: Vec<GasCoinInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GasCoinInfo {
    pub amount: u64,
    pub id: String,
    pub transfer_tx_digest: String,
}
