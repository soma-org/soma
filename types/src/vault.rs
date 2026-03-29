// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::object::ObjectID;

/// Per-seller USDC balance accumulator. Created lazily on first delivery.
/// Prevents coin fragmentation — sellers withdraw in bulk whenever they choose.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SellerVault {
    pub id: ObjectID,
    pub owner: SomaAddress,
    pub balance: u64,
}
