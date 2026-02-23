// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use parking_lot::RwLock;
use types::committee::AuthorityIndex;
use types::consensus::{
    block::{BlockAPI, Slot, TestBlock, Transaction, VerifiedBlock},
    commit::DecidedLeader,
    context::Context,
};
use types::storage::consensus::mem_store::MemStore;

use crate::{
    dag_state::DagState,
    leader_schedule::{LeaderSchedule, LeaderSwapTable},
    test_dag::{build_dag, build_dag_layer},
    test_dag_builder::DagBuilder,
    test_dag_parser::parse_dag,
    universal_committer::universal_committer_builder::UniversalCommitterBuilder,
};

/// Commit one leader.
#[tokio::test]
async fn direct_commit() {
    let mut test_setup = basic_dag_builder_test_setup();

    let leader_round_wave_1 = test_setup.committer.committers[0].leader_round(1);
    let voting_round_wave_2 = test_setup.committer.committers[0].leader_round(2) + 1;
    test_setup
        .dag_builder
        .layers(1..=voting_round_wave_2)
        .build()
        .persist_layers(test_setup.dag_state);

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));

    let sequence = test_setup.committer.try_decide(last_decided);

    assert_eq!(sequence.len(), 1);
    if let DecidedLeader::Commit(ref block, direct) = sequence[0] {
        assert_eq!(block.author(), test_setup.committer.get_leaders(leader_round_wave_1)[0]);
        assert!(direct);
    } else {
        panic!("Expected a committed leader")
    };
}

/// Ensure idempotent replies.
#[tokio::test]
async fn idempotence() {
    let (context, dag_state, committer) = basic_test_setup();

    let leader_round_wave_1 = committer.committers[0].leader_round(1);
    let decision_round_wave_1 = committer.committers[0].decision_round(1);
    let references_decision_round_wave_1 =
        build_dag(context.clone(), dag_state.clone(), None, decision_round_wave_1);

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    let first_sequence = committer.try_decide(last_decided);
    assert_eq!(first_sequence.len(), 1);

    if let DecidedLeader::Commit(ref block, _direct) = first_sequence[0] {
        assert_eq!(first_sequence[0].slot().round, leader_round_wave_1);
        assert_eq!(block.author(), committer.get_leaders(leader_round_wave_1)[0])
    } else {
        panic!("Expected a committed leader")
    };

    // Same input -> same output
    let first_sequence = committer.try_decide(last_decided);

    assert_eq!(first_sequence.len(), 1);
    if let DecidedLeader::Commit(ref block, _direct) = first_sequence[0] {
        assert_eq!(first_sequence[0].slot().round, leader_round_wave_1);
        assert_eq!(block.author(), committer.get_leaders(leader_round_wave_1)[0])
    } else {
        panic!("Expected a committed leader")
    };

    // Add more rounds
    let decision_round_wave_2 = committer.committers[0].decision_round(2);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_decision_round_wave_1),
        decision_round_wave_2,
    );

    let leader_status_wave_1 = first_sequence.last().unwrap();
    let last_decided = leader_status_wave_1.slot();
    let leader_round_wave_2 = committer.committers[0].leader_round(2);
    let second_sequence = committer.try_decide(last_decided);

    assert_eq!(second_sequence.len(), 1);
    if let DecidedLeader::Commit(ref block, _direct) = second_sequence[0] {
        assert_eq!(second_sequence[0].slot().round, leader_round_wave_2);
        assert_eq!(block.author(), committer.get_leaders(leader_round_wave_2)[0]);
    } else {
        panic!("Expected a committed leader")
    };
}

/// Commit one by one each leader as the dag progresses in ideal conditions.
#[tokio::test]
async fn multiple_direct_commit() {
    let (context, dag_state, committer) = basic_test_setup();

    let mut ancestors = None;
    let mut last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    for n in 1..=10 {
        let decision_round = committer.committers[0].decision_round(n);
        ancestors = Some(build_dag(context.clone(), dag_state.clone(), ancestors, decision_round));

        let leader_round = committer.committers[0].leader_round(n);
        let sequence = committer.try_decide(last_decided);

        assert_eq!(sequence.len(), 1);
        if let DecidedLeader::Commit(ref block, direct) = sequence[0] {
            assert_eq!(block.round(), leader_round);
            assert_eq!(block.author(), committer.get_leaders(leader_round)[0]);
            assert!(direct);
        } else {
            panic!("Expected a committed leader")
        }

        let leader_status = sequence.last().unwrap();
        last_decided = leader_status.slot();
    }
}

/// Commit 10 leaders in a row (calling the committer after adding them).
#[tokio::test]
async fn direct_commit_late_call() {
    let (context, dag_state, committer) = basic_test_setup();

    let num_waves = 11;
    let decision_round_wave_10 = committer.committers[0].decision_round(10);
    build_dag(context.clone(), dag_state.clone(), None, decision_round_wave_10);

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = committer.try_decide(last_decided);

    assert_eq!(sequence.len(), num_waves - 1_usize);
    for (i, leader_block) in sequence.iter().enumerate() {
        let leader_round = committer.committers[0].leader_round(i as u32 + 1);
        if let DecidedLeader::Commit(block, direct) = leader_block {
            assert_eq!(block.round(), leader_round);
            assert_eq!(block.author(), committer.get_leaders(leader_round)[0]);
            assert!(direct);
        } else {
            panic!("Expected a committed leader")
        };
    }
}

/// Do not commit anything if we are still in the first wave.
#[tokio::test]
async fn no_genesis_commit() {
    let (context, dag_state, committer) = basic_test_setup();

    let decision_round_wave_1 = committer.committers[0].decision_round(1);
    let mut ancestors = None;
    for r in 0..decision_round_wave_1 {
        ancestors = Some(build_dag(context.clone(), dag_state.clone(), ancestors, r));

        let last_committed = Slot::new(0, AuthorityIndex::new_for_test(0));
        let sequence = committer.try_decide(last_committed);
        assert!(sequence.is_empty());
    }
}

/// We directly skip the leader if there are enough non-votes (blames).
#[tokio::test]
async fn direct_skip_no_leader_votes() {
    let mut test_setup = basic_dag_builder_test_setup();

    let leader_round_wave_1 = test_setup.committer.committers[0].leader_round(1);
    test_setup
        .dag_builder
        .layers(1..=leader_round_wave_1)
        .build()
        .persist_layers(test_setup.dag_state.clone());

    let leader_wave_1 = test_setup.committer.get_leaders(leader_round_wave_1)[0];
    let voting_round_wave_1 = leader_round_wave_1 + 1;
    test_setup
        .dag_builder
        .layer(voting_round_wave_1)
        .no_leader_link(leader_round_wave_1, vec![])
        .persist_layers(test_setup.dag_state.clone());

    let decision_round_wave_1 = test_setup.committer.committers[0].decision_round(1);
    test_setup
        .dag_builder
        .layer(decision_round_wave_1)
        .build()
        .persist_layers(test_setup.dag_state);

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = test_setup.committer.try_decide(last_decided);

    assert_eq!(sequence.len(), 1);
    if let DecidedLeader::Skip(leader) = sequence[0] {
        assert_eq!(leader.authority, leader_wave_1);
        assert_eq!(leader.round, leader_round_wave_1);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

/// We directly skip the leader if it is missing.
#[tokio::test]
async fn direct_skip_missing_leader_block() {
    let mut test_setup = basic_dag_builder_test_setup();

    let decision_round_wave_0 = test_setup.committer.committers[0].decision_round(0);
    test_setup.dag_builder.layers(1..=decision_round_wave_0).build();

    let leader_round_wave_1 = test_setup.committer.committers[0].leader_round(1);
    test_setup.dag_builder.layer(leader_round_wave_1).no_leader_block(vec![]).build();

    let voting_round_wave_1 = leader_round_wave_1 + 1;
    let decision_round_wave_1 = test_setup.committer.committers[0].decision_round(1);
    test_setup.dag_builder.layers(voting_round_wave_1..=decision_round_wave_1).build();

    test_setup.dag_builder.persist_all_blocks(test_setup.dag_state.clone());

    let last_committed = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = test_setup.committer.try_decide(last_committed);

    assert_eq!(sequence.len(), 1);
    if let DecidedLeader::Skip(leader) = sequence[0] {
        assert_eq!(leader.authority, test_setup.committer.get_leaders(leader_round_wave_1)[0]);
        assert_eq!(leader.round, leader_round_wave_1);
    } else {
        panic!("Expected to directly skip the leader");
    }
}

/// Indirect-commit the first leader.
#[tokio::test]
async fn indirect_commit() {
    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : {
            A -> [-D3],
            B -> [*],
            C -> [*],
            D -> [*],
        },
        Round 5 : {
            A -> [*],
            B -> [*],
            C -> [A4],
            D -> [A4],
        },
        Round 6 : { * },
        Round 7 : { * },
        Round 8 : { * },
     }";

    let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag");
    let dag_state = Arc::new(RwLock::new(DagState::new(
        dag_builder.context.clone(),
        Arc::new(MemStore::new()),
    )));
    let leader_schedule =
        Arc::new(LeaderSchedule::new(dag_builder.context.clone(), LeaderSwapTable::default()));

    dag_builder.persist_all_blocks(dag_state.clone());

    let committer = UniversalCommitterBuilder::new(
        dag_builder.context.clone(),
        leader_schedule,
        dag_state.clone(),
    )
    .build();
    assert!(committer.committers.len() == 1);

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = committer.try_decide(last_decided);
    assert_eq!(sequence.len(), 2);

    for (idx, decided_leader) in sequence.iter().enumerate() {
        let leader_round = committer.committers[0].leader_round(idx as u32 + 1);
        let expected_leader = committer.get_leaders(leader_round)[0];
        if let DecidedLeader::Commit(block, _direct) = decided_leader {
            assert_eq!(block.round(), leader_round);
            assert_eq!(block.author(), expected_leader);
        } else {
            panic!("Expected a committed leader")
        };
    }
}

/// Commit the first leader, skip the 2nd, and commit the 3rd leader.
#[tokio::test]
async fn indirect_skip() {
    let (context, dag_state, committer) = basic_test_setup();

    let leader_round_wave_2 = committer.committers[0].leader_round(2);
    let references_leader_round_wave_2 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_2);

    let leader_wave_2 = committer.get_leaders(leader_round_wave_2)[0];
    let references_without_leader_wave_2: Vec<_> = references_leader_round_wave_2
        .iter()
        .cloned()
        .filter(|x| x.author != leader_wave_2)
        .collect();

    let mut references = Vec::new();

    let connections_with_leader_wave_2 = context
        .committee
        .authorities()
        .take(context.committee.validity_n())
        .map(|authority| (authority.0, references_leader_round_wave_2.clone()))
        .collect::<Vec<_>>();

    references.extend(build_dag_layer(connections_with_leader_wave_2, dag_state.clone()));

    let connections_without_leader_wave_2 = context
        .committee
        .authorities()
        .skip(context.committee.validity_n())
        .map(|authority| (authority.0, references_without_leader_wave_2.clone()))
        .collect();

    references.extend(build_dag_layer(connections_without_leader_wave_2, dag_state.clone()));

    let decision_round_wave_3 = committer.committers[0].decision_round(3);
    build_dag(context.clone(), dag_state.clone(), Some(references), decision_round_wave_3);

    let last_committed = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = committer.try_decide(last_committed);
    assert_eq!(sequence.len(), 3);

    // Commit leader of wave 1
    let leader_round_wave_1 = committer.committers[0].leader_round(1);
    let leader_wave_1 = committer.get_leaders(leader_round_wave_1)[0];
    if let DecidedLeader::Commit(ref block, _direct) = sequence[0] {
        assert_eq!(block.round(), leader_round_wave_1);
        assert_eq!(block.author(), leader_wave_1);
    } else {
        panic!("Expected a committed leader")
    };

    // Skip leader of wave 2
    if let DecidedLeader::Skip(leader) = sequence[1] {
        assert_eq!(leader.authority, leader_wave_2);
        assert_eq!(leader.round, leader_round_wave_2);
    } else {
        panic!("Expected a skipped leader")
    }

    // Commit leader of wave 3
    let leader_round_wave_3 = committer.committers[0].leader_round(3);
    let leader_wave_3 = committer.get_leaders(leader_round_wave_3)[0];
    if let DecidedLeader::Commit(ref block, _direct) = sequence[2] {
        assert_eq!(block.round(), leader_round_wave_3);
        assert_eq!(block.author(), leader_wave_3);
    } else {
        panic!("Expected a committed leader")
    }
}

/// If there is no leader with enough support nor blame, we commit nothing.
#[tokio::test]
async fn undecided() {
    let (context, dag_state, committer) = basic_test_setup();

    let leader_round_wave_1 = committer.committers[0].leader_round(1);
    let references_leader_round_wave_1 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_1);

    let references_without_leader_1: Vec<_> = references_leader_round_wave_1
        .iter()
        .cloned()
        .filter(|x| x.author != committer.get_leaders(leader_round_wave_1)[0])
        .collect();

    let mut authorities = context.committee.authorities();
    let leader_wave_1_connection =
        vec![(authorities.next().unwrap().0, references_leader_round_wave_1)];
    let non_leader_wave_1_connections: Vec<_> = authorities
        .take(context.committee.quorum_n() - 1)
        .map(|authority| (authority.0, references_without_leader_1.clone()))
        .collect();

    let connections_voting_round_wave_1 = leader_wave_1_connection
        .into_iter()
        .chain(non_leader_wave_1_connections)
        .collect::<Vec<_>>();
    let references_voting_round_wave_1 =
        build_dag_layer(connections_voting_round_wave_1, dag_state.clone());

    let decision_round_wave_1 = committer.committers[0].decision_round(1);
    build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_voting_round_wave_1),
        decision_round_wave_1,
    );

    let last_committed = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = committer.try_decide(last_committed);
    assert!(sequence.is_empty());
}

/// Byzantine validator sends equivocating blocks.
#[tokio::test]
async fn test_byzantine_direct_commit() {
    let (context, dag_state, committer) = basic_test_setup();

    let leader_round_wave_4 = committer.committers[0].leader_round(4);
    let references_leader_round_wave_4 =
        build_dag(context.clone(), dag_state.clone(), None, leader_round_wave_4);

    let voting_round_wave_4 = committer.committers[0].leader_round(4) + 1;
    let good_references_voting_round_wave_4 = build_dag(
        context.clone(),
        dag_state.clone(),
        Some(references_leader_round_wave_4.clone()),
        voting_round_wave_4,
    );

    let leader_wave_4 = committer.get_leaders(leader_round_wave_4)[0];

    let references_without_leader_round_wave_4: Vec<_> =
        references_leader_round_wave_4.into_iter().filter(|x| x.author != leader_wave_4).collect();

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

    let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
    let sequence = committer.try_decide(last_decided);

    assert_eq!(sequence.len(), 4);
    if let DecidedLeader::Commit(ref block, direct) = sequence[3] {
        assert_eq!(block.author(), committer.get_leaders(leader_round_wave_4)[0]);
        assert!(direct);
    } else {
        panic!("Expected a committed leader")
    };
}

fn basic_test_setup() -> (Arc<Context>, Arc<RwLock<DagState>>, super::UniversalCommitter) {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let leader_schedule =
        Arc::new(LeaderSchedule::new(context.clone(), LeaderSwapTable::default()));

    let committer =
        UniversalCommitterBuilder::new(context.clone(), leader_schedule, dag_state.clone()).build();

    assert!(committer.committers.len() == 1);

    (context, dag_state, committer)
}

struct TestSetup {
    dag_builder: DagBuilder,
    dag_state: Arc<RwLock<DagState>>,
    committer: super::UniversalCommitter,
}

fn basic_dag_builder_test_setup() -> TestSetup {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_builder = DagBuilder::new(context);

    let dag_state = Arc::new(RwLock::new(DagState::new(
        dag_builder.context.clone(),
        Arc::new(MemStore::new()),
    )));
    let leader_schedule =
        Arc::new(LeaderSchedule::new(dag_builder.context.clone(), LeaderSwapTable::default()));

    let committer = UniversalCommitterBuilder::new(
        dag_builder.context.clone(),
        leader_schedule,
        dag_state.clone(),
    )
    .build();
    assert!(committer.committers.len() == 1);

    TestSetup { dag_builder, dag_state, committer }
}
