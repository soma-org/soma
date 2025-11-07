use anyhow::Result;
use async_trait::async_trait;
use types::checkpoint::CommitArchiveData;

pub mod executor;
pub mod reader;
mod util;

#[async_trait]
pub trait Worker: Send + Sync {
    type Result: Send + Sync + Clone;

    async fn process_commit_archive(
        &self,
        archive_data: &CommitArchiveData,
    ) -> Result<Self::Result>;
}
