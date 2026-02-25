// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! RoundTracker computes quorum rounds for the latest received and accepted rounds.
//! This round data is gathered from peers via RoundProber or via new Blocks received. Also
//! local accepted rounds are updated from new blocks proposed from this authority.
//!
//! Quorum rounds provides insight into how effectively each authority's blocks are propagated
//! and accepted across the network.

use std::sync::Arc;

use itertools::Itertools;
use tracing::{debug, trace};
use types::committee::{AuthorityIndex, Committee};
use types::consensus::block::{BlockAPI, ExtendedBlock, Round};
use types::consensus::context::Context;

/// A [`QuorumRound`] is a round range [low, high]. It is computed from
/// highest received or accepted rounds of an authority reported by all
/// authorities.
/// The bounds represent:
/// - the highest round lower or equal to rounds from a quorum (low)
/// - the lowest round higher or equal to rounds from a quorum (high)
///
/// [`QuorumRound`] is useful because:
/// - [low, high] range is BFT, always between the lowest and highest rounds
///   of honest validators, with < validity threshold of malicious stake.
/// - It provides signals about how well blocks from an authority propagates
///   in the network. If low bound for an authority is lower than its last
///   proposed round, the last proposed block has not propagated to a quorum.
///   If a new block is proposed from the authority, it will not get accepted
///   immediately by a quorum.
pub(crate) type QuorumRound = (Round, Round);

pub(crate) struct PeerRoundTracker {
    context: Arc<Context>,
    /// Highest accepted round per authority from received blocks (included/excluded ancestors)
    block_accepted_rounds: Vec<Vec<Round>>,
    /// Highest accepted round per authority from round prober
    probed_accepted_rounds: Vec<Vec<Round>>,
    /// Highest received round per authority from round prober
    probed_received_rounds: Vec<Vec<Round>>,
}

impl PeerRoundTracker {
    pub(crate) fn new(context: Arc<Context>) -> Self {
        let size = context.committee.size();
        Self {
            context,
            block_accepted_rounds: vec![vec![0; size]; size],
            probed_accepted_rounds: vec![vec![0; size]; size],
            probed_received_rounds: vec![vec![0; size]; size],
        }
    }

    /// Update accepted rounds based on a new block created locally or received from the network
    /// and its excluded ancestors
    pub(crate) fn update_from_accepted_block(&mut self, extended_block: &ExtendedBlock) {
        let block = &extended_block.block;
        let excluded_ancestors = &extended_block.excluded_ancestors;
        let author = block.author();

        // Update author accepted round from block round
        self.block_accepted_rounds[author][author] =
            self.block_accepted_rounds[author][author].max(block.round());

        // Update accepted rounds from included ancestors
        for ancestor in block.ancestors() {
            self.block_accepted_rounds[author][ancestor.author] =
                self.block_accepted_rounds[author][ancestor.author].max(ancestor.round);
        }

        // Update accepted rounds from excluded ancestors
        for excluded_ancestor in excluded_ancestors {
            self.block_accepted_rounds[author][excluded_ancestor.author] = self
                .block_accepted_rounds[author][excluded_ancestor.author]
                .max(excluded_ancestor.round);
        }
    }

    /// Update accepted & received rounds based on probing results
    pub(crate) fn update_from_probe(
        &mut self,
        accepted_rounds: Vec<Vec<Round>>,
        received_rounds: Vec<Vec<Round>>,
    ) {
        self.probed_accepted_rounds = accepted_rounds;
        self.probed_received_rounds = received_rounds;
    }

    // Returns the propagation delay of own blocks.
    pub(crate) fn calculate_propagation_delay(&self, last_proposed_round: Round) -> Round {
        let own_index = self.context.own_index;

        let received_quorum_rounds = self.compute_received_quorum_rounds();
        let accepted_quorum_rounds = self.compute_accepted_quorum_rounds();

        // TODO: consider using own quorum round gap to control proposing in addition to
        // propagation delay. For now they seem to be about the same.

        // It is possible more blocks arrive at a quorum of peers before the get_latest_rounds
        // requests arrive.
        // Using the lower bound to increase sensitivity about block propagation issues
        // that can reduce round rate.
        // Because of the nature of TCP and block streaming, propagation delay is expected to be
        // 0 in most cases, even when the actual latency of broadcasting blocks is high.
        // We will use the min propagation delay from either accepted or received rounds.
        // As stated above new blocks can arrive after the rounds have been probed, so its
        // likely accepted rounds from new blocks will provide us with the more accurate
        // propagation delay which is important because we now calculate the propagation
        // delay more frequently then before.
        let propagation_delay = last_proposed_round
            .saturating_sub(received_quorum_rounds[own_index].0)
            .min(last_proposed_round.saturating_sub(accepted_quorum_rounds[own_index].0));

        debug!(
            "Computed propagation delay of {propagation_delay} based on last proposed \
                round ({last_proposed_round})."
        );

        propagation_delay
    }

    pub(crate) fn compute_accepted_quorum_rounds(&self) -> Vec<QuorumRound> {
        let highest_accepted_rounds = self
            .probed_accepted_rounds
            .iter()
            .zip(self.block_accepted_rounds.iter())
            .map(|(probed_rounds, block_rounds)| {
                probed_rounds
                    .iter()
                    .zip(block_rounds.iter())
                    .map(|(probed_round, block_round)| *probed_round.max(block_round))
                    .collect::<Vec<Round>>()
            })
            .collect::<Vec<Vec<Round>>>();
        let accepted_quorum_rounds = self
            .context
            .committee
            .authorities()
            .map(|(peer, _)| {
                compute_quorum_round(&self.context.committee, peer, &highest_accepted_rounds)
            })
            .collect::<Vec<_>>();

        trace!(
            "Computed accepted quorum round per authority: {}",
            self.context
                .committee
                .authorities()
                .zip(accepted_quorum_rounds.iter())
                .map(|((i, _), rounds)| format!("{i}: {rounds:?}"))
                .join(", ")
        );

        accepted_quorum_rounds
    }

    fn compute_received_quorum_rounds(&self) -> Vec<QuorumRound> {
        let received_quorum_rounds = self
            .context
            .committee
            .authorities()
            .map(|(peer, _)| {
                compute_quorum_round(&self.context.committee, peer, &self.probed_received_rounds)
            })
            .collect::<Vec<_>>();

        trace!(
            "Computed received quorum round per authority: {}",
            self.context
                .committee
                .authorities()
                .zip(received_quorum_rounds.iter())
                .map(|((i, _), rounds)| format!("{i}: {rounds:?}"))
                .join(", ")
        );

        received_quorum_rounds
    }
}

/// For the peer specified with target_index, compute and return its [`QuorumRound`].
fn compute_quorum_round(
    committee: &Committee,
    target_index: AuthorityIndex,
    highest_rounds: &[Vec<Round>],
) -> QuorumRound {
    let mut rounds_with_stake = highest_rounds
        .iter()
        .zip(committee.authorities())
        .map(|(rounds, (_, authority))| (rounds[target_index], authority.stake))
        .collect::<Vec<_>>();
    rounds_with_stake.sort();

    // Forward iteration and stopping at validity threshold would produce the same result currently,
    // with fault tolerance of f/3f+1 votes. But it is not semantically correct, and will provide an
    // incorrect value when fault tolerance and validity threshold are different.
    let mut total_stake = 0;
    let mut low = 0;
    for (round, stake) in rounds_with_stake.iter().rev() {
        total_stake += stake;
        if total_stake >= committee.quorum_threshold() {
            low = *round;
            break;
        }
    }

    let mut total_stake = 0;
    let mut high = 0;
    for (round, stake) in rounds_with_stake.iter() {
        total_stake += stake;
        if total_stake >= committee.quorum_threshold() {
            high = *round;
            break;
        }
    }

    (low, high)
}
