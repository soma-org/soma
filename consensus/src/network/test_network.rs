use crate::{
    block::{BlockRef, VerifiedBlock},
    commit::{CommitRange, TrustedCommit},
    error::ConsensusResult,
    Round,
};
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use types::committee::AuthorityIndex;

use super::NetworkService;

pub(crate) struct TestService {
    pub(crate) handle_send_block: Vec<(AuthorityIndex, Bytes)>,
    pub(crate) handle_fetch_blocks: Vec<(AuthorityIndex, Vec<BlockRef>)>,
    pub(crate) handle_fetch_commits: Vec<(AuthorityIndex, CommitRange)>,
    pub(crate) own_blocks: Vec<Bytes>,
}

impl TestService {
    pub(crate) fn new() -> Self {
        Self {
            handle_send_block: Vec::new(),
            handle_fetch_blocks: Vec::new(),
            handle_fetch_commits: Vec::new(),
            own_blocks: Vec::new(),
        }
    }

    pub(crate) fn add_own_blocks(&mut self, blocks: Vec<Bytes>) {
        self.own_blocks.extend(blocks);
    }
}

#[async_trait]
impl NetworkService for Mutex<TestService> {
    async fn handle_send_block(&self, peer: AuthorityIndex, block: Bytes) -> ConsensusResult<()> {
        let mut state = self.lock();
        state.handle_send_block.push((peer, block));
        Ok(())
    }

    async fn handle_fetch_blocks(
        &self,
        peer: AuthorityIndex,
        block_refs: Vec<BlockRef>,
        _highest_accepted_rounds: Vec<Round>,
    ) -> ConsensusResult<Vec<Bytes>> {
        self.lock().handle_fetch_blocks.push((peer, block_refs));
        Ok(vec![])
    }

    async fn handle_fetch_commits(
        &self,
        peer: AuthorityIndex,
        commit_range: CommitRange,
    ) -> ConsensusResult<(Vec<TrustedCommit>, Vec<VerifiedBlock>)> {
        self.lock().handle_fetch_commits.push((peer, commit_range));
        Ok((vec![], vec![]))
    }

    async fn handle_fetch_latest_blocks(
        &self,
        _peer: AuthorityIndex,
        _authorities: Vec<AuthorityIndex>,
    ) -> ConsensusResult<Vec<Bytes>> {
        unimplemented!("Unimplemented")
    }
}
