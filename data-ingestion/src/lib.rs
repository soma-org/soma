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
