use std::{cmp::Ordering, sync::Arc};

use tokio::time::Instant;
use tracing::{debug, info};

use types::consensus::{
    block::{BlockRef, Round},
    context::Context,
    stake_aggregator::{QuorumThreshold, StakeAggregator},
};

pub(crate) struct ThresholdClock {
    context: Arc<Context>,
    aggregator: StakeAggregator<QuorumThreshold>,
    round: Round,
    // Timestamp when the last quorum was form and the current round started.
    quorum_ts: Instant,
}

impl ThresholdClock {
    pub(crate) fn new(round: Round, context: Arc<Context>) -> Self {
        info!("Recovered ThresholdClock at round {}", round);
        Self { context, aggregator: StakeAggregator::new(), round, quorum_ts: Instant::now() }
    }

    /// Adds the block reference that have been accepted and advance the round accordingly.
    /// Returns true when the round has advanced.
    pub(crate) fn add_block(&mut self, block: BlockRef) -> bool {
        match block.round.cmp(&self.round) {
            // Blocks with round less then what we currently build are irrelevant here
            Ordering::Less => false,
            Ordering::Equal => {
                let now = Instant::now();
                if self.aggregator.add(block.author, &self.context.committee) {
                    self.aggregator.clear();
                    // We have seen 2f+1 blocks for current round, advance
                    self.round = block.round + 1;
                    // Record the time of last quorum and new round start.
                    self.quorum_ts = now;
                    debug!(
                        "ThresholdClock advanced to round {} with block {} completing quorum",
                        self.round, block
                    );
                    return true;
                }

                false
            }
            // If we processed block for round r, we also have stored 2f+1 blocks from r-1
            Ordering::Greater => {
                self.aggregator.clear();
                if self.aggregator.add(block.author, &self.context.committee) {
                    // Even though this is the first block of the round, there is still a quorum at block.round.
                    self.round = block.round + 1;
                } else {
                    // There is a quorum at block.round - 1 but not block.round.
                    self.round = block.round;
                };
                self.quorum_ts = Instant::now();
                debug!(
                    "ThresholdClock advanced to round {} with block {} catching up round",
                    self.round, block
                );
                true
            }
        }
    }

    /// Add the block references that have been successfully processed and advance the round accordingly. If the round
    /// has indeed advanced then the new round is returned, otherwise None is returned.
    #[cfg(test)]
    fn add_blocks(&mut self, blocks: Vec<BlockRef>) -> Option<Round> {
        let previous_round = self.round;
        for block_ref in blocks {
            self.add_block(block_ref);
        }
        (self.round > previous_round).then_some(self.round)
    }

    pub(crate) fn get_round(&self) -> Round {
        self.round
    }

    pub(crate) fn get_quorum_ts(&self) -> Instant {
        self.quorum_ts
    }
}
