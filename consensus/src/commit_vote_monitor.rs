// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use parking_lot::Mutex;

use types::consensus::{
    block::{BlockAPI as _, VerifiedBlock},
    commit::{CommitIndex, GENESIS_COMMIT_INDEX},
    context::Context,
};

/// Monitors the progress of consensus commits across the network.
pub(crate) struct CommitVoteMonitor {
    context: Arc<Context>,
    // Highest commit index voted by each authority.
    highest_voted_commits: Mutex<Vec<CommitIndex>>,
}

impl CommitVoteMonitor {
    pub(crate) fn new(context: Arc<Context>) -> Self {
        let highest_voted_commits = Mutex::new(vec![0; context.committee.size()]);
        Self { context, highest_voted_commits }
    }

    /// Keeps track of the highest commit voted by each authority.
    pub(crate) fn observe_block(&self, block: &VerifiedBlock) {
        let mut highest_voted_commits = self.highest_voted_commits.lock();
        for vote in block.commit_votes() {
            if vote.index > highest_voted_commits[block.author()] {
                highest_voted_commits[block.author()] = vote.index;
            }
        }
    }

    // Finds the highest commit index certified by a quorum.
    // When an authority votes for commit index S, it is also voting for all commit indices 1 <= i < S.
    // So the quorum commit index is the smallest index S such that the sum of stakes of authorities
    // voting for commit indices >= S passes the quorum threshold.
    pub(crate) fn quorum_commit_index(&self) -> CommitIndex {
        let highest_voted_commits = self.highest_voted_commits.lock();
        let mut highest_voted_commits = highest_voted_commits
            .iter()
            .zip(self.context.committee.authorities())
            .map(|(commit_index, (_, a))| (*commit_index, a.stake))
            .collect::<Vec<_>>();
        // Sort by commit index then stake, in descending order.
        highest_voted_commits.sort_by(|a, b| a.cmp(b).reverse());
        let mut total_stake = 0;
        for (commit_index, stake) in highest_voted_commits {
            total_stake += stake;
            if total_stake >= self.context.committee.quorum_threshold() {
                return commit_index;
            }
        }
        GENESIS_COMMIT_INDEX
    }
}
