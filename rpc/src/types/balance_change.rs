// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use crate::types::Address;

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde_derive::Serialize,
    serde_derive::Deserialize
)]

pub struct BalanceChange {
    /// Owner of the balance change
    pub address: Address,

    /// The amount indicate the balance value changes.
    ///
    /// A negative amount means spending coin value and positive means receiving coin value.
    pub amount: i128,
}
