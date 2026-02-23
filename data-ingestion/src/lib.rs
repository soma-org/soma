// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use types::full_checkpoint_content::CheckpointData;

pub mod executor;
pub mod reader;
mod util;

#[async_trait]
pub trait Worker: Send + Sync {
    type Result: Send + Sync + Clone;
    async fn process_checkpoint(&self, checkpoint_data: &CheckpointData) -> Result<Self::Result>;
}
