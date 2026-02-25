// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use prost_types::FieldMask;

use super::*;

impl ExecuteTransactionRequest {
    pub fn new(transaction: Transaction) -> Self {
        Self { transaction: Some(transaction), ..Default::default() }
    }

    pub fn with_read_mask(mut self, read_mask: FieldMask) -> Self {
        self.read_mask = Some(read_mask);
        self
    }
}

impl ExecuteTransactionResponse {
    pub fn new(transaction: ExecutedTransaction) -> Self {
        Self { transaction: Some(transaction), ..Default::default() }
    }
}

impl SimulateTransactionRequest {
    pub fn new(transaction: Transaction) -> Self {
        Self { transaction: Some(transaction), ..Default::default() }
    }

    pub fn with_read_mask(mut self, read_mask: FieldMask) -> Self {
        self.read_mask = Some(read_mask);
        self
    }
}

impl SubscribeCheckpointsRequest {
    pub fn with_read_mask(mut self, read_mask: FieldMask) -> Self {
        self.read_mask = Some(read_mask);
        self
    }
}
