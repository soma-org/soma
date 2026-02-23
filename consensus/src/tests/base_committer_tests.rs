// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use parking_lot::RwLock;
use types::committee::AuthorityIndex;
use types::consensus::{
    block::{BlockAPI, TestBlock, Transaction, VerifiedBlock},
    commit::LeaderStatus,
    context::Context,
};
use types::storage::consensus::mem_store::MemStore;

use crate::{
    base_committer::base_committer_builder::BaseCommitterBuilder,
    dag_state::DagState,
    test_dag::{build_dag, build_dag_layer},
};

/// Commit one leader.
#[tokio::test]
async fn try_direct_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let num_rounds_in_dag = 8;
    let voting_round_wave_2 = committer.leader_round(2) + 1;
    let incomplete_wave_leader_round = 6;
    build_dag(context, dag_state, None, voting_round_wave_2);

    let mut leader_rounds: Vec<u32> = (1..num_rounds_in_dag)
        .map(|r| committer.leader_round(r))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    leader_rounds.sort_by(|a, b| b.cmp(a));
    for round in leader_rounds.into_iter() {
        let leader = committer.elect_leader(round).expect("should have elected leader");
        let leader_status = committer.try_direct_decide(leader);

        if round < incomplete_wave_leader_round {
            if let LeaderStatus::Commit(ref committed_block) = leader_status {
                assert_eq!(committed_block.author(), leader.authority)
            } else {
                panic!("Expected a committed leader at round {}", round)
            };
        } else if let LeaderStatus::Undecided(undecided_slot) = leader_status {
            assert_eq!(undecided_slot, leader)
        } else {
            panic!("Expected an undecided leader")
        }
    }
}

/// Ensure idempotent replies.
#[tokio::test]
async fn idempotence() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let decision_round_wave_1 = committer.decision_round(1);
    build_dag(context, dag_state, None, decision_round_wave_1);

    let leader_round_wave_1 = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round_wave_1).expect("should have elected leader");
    let leader_status = committer.try_direct_decide(leader);

    if let LeaderStatus::Commit(ref block) = leader_status {
        assert_eq!(block.author(), leader.authority)
    } else {
        panic!("Expected a committed leader")
    };

    let leader_status = committer.try_direct_decide(leader);

    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader.authority)
    } else {
        panic!("Expected a committed leader")
    };
}

/// Commit one by one each leader as the dag progresses in ideal conditions.
#[tokio::test]
async fn multiple_direct_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let mut ancestors = None;
    for n in 1..=10 {
        let decision_round = committer.decision_round(n);
        ancestors = Some(build_dag(context.clone(), dag_state.clone(), ancestors, decision_round));

        let leader_round = committer.leader_round(n);
        let leader = committer.elect_leader(leader_round).expect("should have elected leader");
        let leader_status = committer.try_direct_decide(leader);

        if let LeaderStatus::Commit(ref committed_block) = leader_status {
            assert_eq!(committed_block.author(), leader.authority)
        } else {
            panic!("Expected a committed leader")
        };
    }
}

/// We directly skip the leader if it has enough blame.
#[tokio::test]
async fn direct_skip() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let leader_round_wave_1 = committer.leader_round(1);
    let references_leader_round_wave_1 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_1);

    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("should have elected leader");
    let references_without_leader_wave_1: Vec<_> = references_leader_round_wave_1
        .into_iter()
        .filter(|x| x.author != leader_wave_1.authority)
        .collect();

    let decision_round_wave_1 = committer.decision_round(1);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_without_leader_wave_1),
        decision_round_wave_1,
    );

    let leader_status = committer.try_direct_decide(leader_wave_1);

    if let LeaderStatus::Skip(skipped_leader) = leader_status {
        assert_eq!(skipped_leader, leader_wave_1);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

/// Indirect-commit the first leader.
#[tokio::test]
async fn indirect_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let leader_round_wave_1 = committer.leader_round(1);
    let references_leader_round_wave_1 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_1);

    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("should have elected leader");
    let references_without_leader_wave_1: Vec<_> = references_leader_round_wave_1
        .iter()
        .cloned()
        .filter(|x| x.author != leader_wave_1.authority)
        .collect();

    // Only 2f+1 validators vote for the leader of wave 1.
    let connections_with_leader_wave_1 = context
        .committee
        .authorities()
        .take(context.committee.quorum_n())
        .map(|authority| (authority.0, references_leader_round_wave_1.clone()))
        .collect();
    let references_with_votes_for_leader_wave_1 =
        build_dag_layer(connections_with_leader_wave_1, dag_state.clone());

    let connections_without_leader_wave_1 = context
        .committee
        .authorities()
        .skip(context.committee.quorum_n())
        .map(|authority| (authority.0, references_without_leader_wave_1.clone()))
        .collect();
    let references_without_votes_for_leader_wave_1 =
        build_dag_layer(connections_without_leader_wave_1, dag_state.clone());

    // Only f+1 validators certify the leader of wave 1.
    let mut references_decision_round_wave_1 = Vec::new();

    let connections_with_certs_for_leader_wave_1 = context
        .committee
        .authorities()
        .take(context.committee.validity_n())
        .map(|authority| (authority.0, references_with_votes_for_leader_wave_1.clone()))
        .collect();
    references_decision_round_wave_1
        .extend(build_dag_layer(connections_with_certs_for_leader_wave_1, dag_state.clone()));

    let references_voting_round_wave_1: Vec<_> = references_without_votes_for_leader_wave_1
        .into_iter()
        .chain(references_with_votes_for_leader_wave_1)
        .take(context.committee.quorum_n())
        .collect();

    let connections_without_votes_for_leader_1 = context
        .committee
        .authorities()
        .skip(context.committee.validity_n())
        .map(|authority| (authority.0, references_voting_round_wave_1.clone()))
        .collect();
    references_decision_round_wave_1
        .extend(build_dag_layer(connections_without_votes_for_leader_1, dag_state.clone()));

    let decision_round_wave_2 = committer.decision_round(2);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_decision_round_wave_1),
        decision_round_wave_2,
    );

    let leader_wave_2 =
        committer.elect_leader(committer.leader_round(2)).expect("should have elected leader");
    let leader_status = committer.try_direct_decide(leader_wave_2);

    let mut decided_leaders = vec![];
    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader_wave_2.authority);
        decided_leaders.push(leader_status);
    } else {
        panic!("Expected a committed leader")
    };

    let leader_status = committer.try_direct_decide(leader_wave_1);

    if let LeaderStatus::Undecided(undecided_slot) = leader_status {
        assert_eq!(undecided_slot, leader_wave_1)
    } else {
        panic!("Expected an undecided leader")
    };

    let leader_status = committer.try_indirect_decide(leader_wave_1, decided_leaders.iter());

    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader_wave_1.authority)
    } else {
        panic!("Expected a committed leader")
    };
}

/// Commit the first leader, indirectly skip the 2nd, and commit the 3rd leader.
#[tokio::test]
async fn indirect_skip() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let leader_round_wave_2 = committer.leader_round(2);
    let references_leader_round_wave_2 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_2);

    let leader_wave_2 =
        committer.elect_leader(leader_round_wave_2).expect("should have elected leader");
    let references_without_leader_wave_2: Vec<_> = references_leader_round_wave_2
        .iter()
        .cloned()
        .filter(|x| x.author != leader_wave_2.authority)
        .collect();

    let mut references_voting_round_wave_2 = Vec::new();

    let connections_with_vote_leader_wave_2 = context
        .committee
        .authorities()
        .take(context.committee.validity_n())
        .map(|authority| (authority.0, references_leader_round_wave_2.clone()))
        .collect();

    references_voting_round_wave_2
        .extend(build_dag_layer(connections_with_vote_leader_wave_2, dag_state.clone()));

    let connections_without_vote_leader_wave_2 = context
        .committee
        .authorities()
        .skip(context.committee.validity_n())
        .map(|authority| (authority.0, references_without_leader_wave_2.clone()))
        .collect();

    references_voting_round_wave_2
        .extend(build_dag_layer(connections_without_vote_leader_wave_2, dag_state.clone()));

    let decision_round_wave_3 = committer.decision_round(3);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_voting_round_wave_2),
        decision_round_wave_3,
    );

    // 1. Ensure we commit the leader of wave 3.
    let leader_round_wave_3 = committer.leader_round(3);
    let leader_wave_3 =
        committer.elect_leader(leader_round_wave_3).expect("should have elected leader");
    let leader_status = committer.try_direct_decide(leader_wave_3);

    let mut decided_leaders = vec![];
    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader_wave_3.authority);
        decided_leaders.push(leader_status);
    } else {
        panic!("Expected a committed leader")
    };

    // 2. Ensure we directly mark leader of wave 2 undecided.
    let leader_wave_2 =
        committer.elect_leader(leader_round_wave_2).expect("should have elected leader");
    let leader_status = committer.try_direct_decide(leader_wave_2);

    if let LeaderStatus::Undecided(undecided_slot) = leader_status {
        assert_eq!(undecided_slot, leader_wave_2)
    } else {
        panic!("Expected an undecided leader")
    };

    // 3. Ensure we skip leader of wave 2 indirectly.
    let leader_status = committer.try_indirect_decide(leader_wave_2, decided_leaders.iter());

    if let LeaderStatus::Skip(skipped_slot) = leader_status {
        assert_eq!(skipped_slot, leader_wave_2)
    } else {
        panic!("Expected a skipped leader")
    };

    // Ensure we directly commit the leader of wave 1.
    let leader_round_wave_1 = committer.leader_round(1);
    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("should have elected leader");
    let leader_status = committer.try_direct_decide(leader_wave_1);

    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader_wave_1.authority);
    } else {
        panic!("Expected a committed leader")
    };
}

/// If there is no leader with enough support nor blame, we commit nothing.
#[tokio::test]
async fn undecided() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let leader_round_wave_1 = committer.leader_round(1);
    let references_leader_round_wave_1 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_1);

    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("should have elected leader");
    let references_without_leader_wave_1: Vec<_> = references_leader_round_wave_1
        .iter()
        .cloned()
        .filter(|x| x.author != leader_wave_1.authority)
        .collect();

    // Create a dag layer where only one authority votes for the leader of wave 1.
    let mut authorities = context.committee.authorities();
    let connections_leader_wave_1 =
        vec![(authorities.next().unwrap().0, references_leader_round_wave_1)];

    let connections_without_leader_wave_1: Vec<_> = authorities
        .take(context.committee.quorum_n() - 1)
        .map(|authority| (authority.0, references_without_leader_wave_1.clone()))
        .collect();

    let connections_voting_round_wave_1 =
        connections_leader_wave_1.into_iter().chain(connections_without_leader_wave_1).collect();
    let references_voting_round_wave_1 =
        build_dag_layer(connections_voting_round_wave_1, dag_state.clone());

    let decision_round_wave_1 = committer.decision_round(1);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_voting_round_wave_1),
        decision_round_wave_1,
    );

    let leader_status = committer.try_direct_decide(leader_wave_1);

    if let LeaderStatus::Undecided(undecided_slot) = leader_status {
        assert_eq!(undecided_slot, leader_wave_1)
    } else {
        panic!("Expected an undecided leader")
    };

    let leader_status = committer.try_indirect_decide(leader_wave_1, [].iter());

    if let LeaderStatus::Undecided(undecided_slot) = leader_status {
        assert_eq!(undecided_slot, leader_wave_1)
    } else {
        panic!("Expected an undecided leader")
    };
}

/// Byzantine validator sends equivocating blocks; commit rule still works.
#[tokio::test]
async fn test_byzantine_direct_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let leader_round_wave_4 = committer.leader_round(4);
    let references_leader_round_wave_4 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_4);

    let voting_round_wave_4 = leader_round_wave_4 + 1;
    let good_references_voting_round_wave_4 = build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_leader_round_wave_4.clone()),
        voting_round_wave_4,
    );

    let leader_wave_4 =
        committer.elect_leader(leader_round_wave_4).expect("should have elected leader");

    let references_without_leader_round_wave_4: Vec<_> = references_leader_round_wave_4
        .into_iter()
        .filter(|x| x.author != leader_wave_4.authority)
        .collect();

    let byzantine_block_c13_1 = VerifiedBlock::new_for_test(
        TestBlock::new(13, 2)
            .set_ancestors(references_without_leader_round_wave_4.clone())
            .set_transactions(vec![Transaction::new(vec![1])])
            .build(),
    );
    dag_state.write().accept_block(byzantine_block_c13_1.clone());

    let byzantine_block_c13_2 = VerifiedBlock::new_for_test(
        TestBlock::new(13, 2)
            .set_ancestors(references_without_leader_round_wave_4.clone())
            .set_transactions(vec![Transaction::new(vec![2])])
            .build(),
    );
    dag_state.write().accept_block(byzantine_block_c13_2.clone());

    let byzantine_block_c13_3 = VerifiedBlock::new_for_test(
        TestBlock::new(13, 2)
            .set_ancestors(references_without_leader_round_wave_4)
            .set_transactions(vec![Transaction::new(vec![3])])
            .build(),
    );
    dag_state.write().accept_block(byzantine_block_c13_3.clone());

    let decison_block_a14 = VerifiedBlock::new_for_test(
        TestBlock::new(14, 0).set_ancestors(good_references_voting_round_wave_4.clone()).build(),
    );
    dag_state.write().accept_block(decison_block_a14.clone());

    let good_references_voting_round_wave_4_without_c13 = good_references_voting_round_wave_4
        .into_iter()
        .filter(|r| r.author != AuthorityIndex::new_for_test(2))
        .collect::<Vec<_>>();

    let decison_block_b14 = VerifiedBlock::new_for_test(
        TestBlock::new(14, 1)
            .set_ancestors(
                good_references_voting_round_wave_4_without_c13
                    .iter()
                    .cloned()
                    .chain(std::iter::once(byzantine_block_c13_1.reference()))
                    .collect(),
            )
            .build(),
    );
    dag_state.write().accept_block(decison_block_b14.clone());

    let decison_block_c14 = VerifiedBlock::new_for_test(
        TestBlock::new(14, 2)
            .set_ancestors(
                good_references_voting_round_wave_4_without_c13
                    .iter()
                    .cloned()
                    .chain(std::iter::once(byzantine_block_c13_2.reference()))
                    .collect(),
            )
            .build(),
    );
    dag_state.write().accept_block(decison_block_c14.clone());

    let decison_block_d14 = VerifiedBlock::new_for_test(
        TestBlock::new(14, 3)
            .set_ancestors(
                good_references_voting_round_wave_4_without_c13
                    .iter()
                    .cloned()
                    .chain(std::iter::once(byzantine_block_c13_3.reference()))
                    .collect(),
            )
            .build(),
    );
    dag_state.write().accept_block(decison_block_d14.clone());

    let leader_status = committer.try_direct_decide(leader_wave_4);

    if let LeaderStatus::Commit(ref committed_block) = leader_status {
        assert_eq!(committed_block.author(), leader_wave_4.authority);
    } else {
        panic!("Expected a committed leader")
    };
}
