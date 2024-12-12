use async_trait::async_trait;
use p2p::builder::StateSyncHandle;
use std::sync::Arc;
use tracing::info;
use types::{
    error::SomaResult,
    state_sync::{CertifiedCommitSummary, VerifiedCommitSummary},
};

use crate::epoch_store::AuthorityPerEpochStore;

use super::CommitStore;

#[async_trait]
pub trait CommitOutput: Sync + Send + 'static {
    async fn commit_created(
        &self,
        summary: &CertifiedCommitSummary,
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
        summary: &CertifiedCommitSummary,
        _epoch_store: &Arc<AuthorityPerEpochStore>,
        _commit_store: &Arc<CommitStore>,
    ) -> SomaResult {
        info!(
            "Certified commit with index {} and digest {}",
            summary.index,
            summary.digest()
        );
        self.handle
            .send_commit(VerifiedCommitSummary::new_unchecked(summary.to_owned()))
            .await;
        Ok(())
    }
}
