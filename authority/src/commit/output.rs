use async_trait::async_trait;
use p2p::builder::StateSyncHandle;
use std::sync::Arc;
use tracing::info;
use types::{consensus::commit::CommittedSubDag, error::SomaResult};

use crate::epoch_store::AuthorityPerEpochStore;

use super::CommitStore;

#[async_trait]
pub trait CommitOutput: Sync + Send + 'static {
    async fn commit_created(
        &self,
        commit: CommittedSubDag,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        commit_store: &Arc<CommitStore>,
    ) -> SomaResult;
}

pub struct SubmitCommitToStateSync {
    handle: StateSyncHandle,
}

impl SubmitCommitToStateSync {
    pub fn new(handle: StateSyncHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl CommitOutput for SubmitCommitToStateSync {
    async fn commit_created(
        &self,
        commit: CommittedSubDag,
        _epoch_store: &Arc<AuthorityPerEpochStore>,
        _commit_store: &Arc<CommitStore>,
    ) -> SomaResult {
        info!(
            "Certified commit with index {} and digest {}",
            commit.commit_ref.index, commit.commit_ref.digest
        );
        self.handle.send_commit(commit).await;
        Ok(())
    }
}
