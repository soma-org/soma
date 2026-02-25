// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct AdvanceEpochRequest {}

#[derive(Debug, Deserialize, Serialize)]
pub struct AdvanceEpochResponse {
    pub epoch: u64,
}
