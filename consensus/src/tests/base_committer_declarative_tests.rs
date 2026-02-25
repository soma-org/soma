// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use parking_lot::RwLock;
use types::consensus::{
    block::{BlockAPI, TestBlock, VerifiedBlock},
    commit::LeaderStatus,
    context::Context,
};
use types::storage::consensus::mem_store::MemStore;

use crate::{
    base_committer::base_committer_builder::BaseCommitterBuilder, dag_state::DagState,
    test_dag_parser::parse_dag,
};

#[tokio::test]
async fn direct_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : {
            A -> [D3],
            B -> [D3],
            C -> [D3],
            D -> [],
        },
        Round 5 : {
            A -> [A4, B4, C4],
            B -> [A4, B4, C4],
            C -> [A4, B4, C4],
            D -> [],
        },
        }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round).expect("there should be a leader at wave 1");
    let leader_status = committer.try_direct_decide(leader);
    if let LeaderStatus::Commit(_) = leader_status {
        // ok
    } else {
        panic!("Expected a committed leader, got {leader_status}");
    }
}

#[tokio::test]
async fn direct_skip() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : {
            A -> [D3],
            B -> [],
            C -> [],
            D -> [],
        },
        Round 5 : { * },
        }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round).expect("there should be a leader at wave 1");
    let leader_status = committer.try_direct_decide(leader);
    if let LeaderStatus::Skip(_) = leader_status {
        // ok
    } else {
        panic!("Expected a skipped leader, got {leader_status}");
    }
}

#[tokio::test]
async fn direct_undecided() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : {
            A -> [D3],
            B -> [D3],
            C -> [],
            D -> [],
        },
        Round 5 : { * },
        }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round).expect("there should be a leader at wave 1");
    let leader_status = committer.try_direct_decide(leader);
    if let LeaderStatus::Undecided(_) = leader_status {
        // ok
    } else {
        panic!("Expected an undecided leader, got {leader_status}");
    }
}

#[tokio::test]
async fn indirect_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : {
            A -> [D3],
            B -> [D3],
            C -> [D3],
            D -> [],
        },
        Round 5 : {
            A -> [A4, B4, C4],
            B -> [],
            C -> [],
            D -> [],
        },
        Round 6 : {
            A -> [],
            B -> [],
            C -> [A5],
            D -> [],
        },
        Round 7 : {
            A -> [C6],
            B -> [C6],
            C -> [],
            D -> [C6],
        },
        Round 8 : {
            A -> [A7, B7, D7],
            B -> [A7, B7, D7],
            C -> [],
            D -> [A7, B7, D7],
        },
    }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round).expect("there should be a leader for wave 1");

    let leader_status_wave1 = committer.try_direct_decide(leader);
    if let LeaderStatus::Undecided(_) = leader_status_wave1 {
        // ok
    } else {
        panic!(
            "Expected LeaderStatus::Undecided for wave 1 direct decide, got {leader_status_wave1}"
        );
    }

    let leader_round_wave2 = committer.leader_round(2);
    let leader_wave2 =
        committer.elect_leader(leader_round_wave2).expect("there should be a leader for wave 2");

    let leader_status_wave_2 = committer.try_direct_decide(leader_wave2);
    if let LeaderStatus::Commit(_) = leader_status_wave_2 {
        // ok
    } else {
        panic!("Expected LeaderStatus::Commit for wave 2, got {leader_status_wave_2}");
    };

    let leader_status_wave1_indirect =
        committer.try_indirect_decide(leader, [leader_status_wave_2].iter());

    if let LeaderStatus::Commit(_) = leader_status_wave1_indirect {
        // ok
    } else {
        panic!(
            "Expected LeaderStatus::Commit for wave 1 indirect, got {leader_status_wave1_indirect}"
        );
    };
}

#[tokio::test]
async fn indirect_skip() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * },
        Round 4 : { * },
        Round 5 : { * },
        Round 6 : { * },
        Round 7 : {
            A -> [*],
            B -> [*],
            C -> [-C6],
            D -> [-C6],
        },
        Round 8 : { * },
        Round 9 : { * },
        Round 10 : { * },
        Round 11 : { * },
    }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    // Wave 1 leader should commit directly
    let leader_round = committer.leader_round(1);
    let leader = committer.elect_leader(leader_round).expect("there should be a leader for wave 1");
    let leader_status_wave1 = committer.try_direct_decide(leader);
    if let LeaderStatus::Commit(_) = leader_status_wave1 {
        // ok
    } else {
        panic!("Expected LeaderStatus::Commit for wave 1, got {leader_status_wave1}");
    }

    // Wave 2 leader should be undecided directly
    let leader_round_wave_2 = committer.leader_round(2);
    let leader_wave2 =
        committer.elect_leader(leader_round_wave_2).expect("there should be a leader for wave 2");
    let leader_status_wave_2 = committer.try_direct_decide(leader_wave2);
    if let LeaderStatus::Undecided(_) = leader_status_wave_2 {
        // ok
    } else {
        panic!("Expected LeaderStatus::Undecided for wave 2, got {leader_status_wave_2}");
    };

    // Wave 3 leader should commit directly
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

    // Wave 2 leader should skip indirectly
    let leader_status = committer.try_indirect_decide(leader_wave2, decided_leaders.iter());

    if let LeaderStatus::Skip(skipped_slot) = leader_status {
        assert_eq!(skipped_slot, leader_wave2)
    } else {
        panic!("Expected a skipped leader")
    };
}

/// Equivocating authority votes for leader -> still committed
#[tokio::test]
async fn test_equivocating_direct_commit() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * }
        Round 4 : {
            A -> [],
            B -> [],
            C -> [*],
            D -> [*],
        },
        Round 5 : { * },
    }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round_wave_1 = committer.leader_round(1);
    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("there should be a leader for wave 1");

    let leader_status_wave_1 = committer.try_direct_decide(leader_wave_1);
    if let LeaderStatus::Undecided(_) = leader_status_wave_1 {
        // ok - initially undecided
    } else {
        panic!("Expected LeaderStatus::Undecided for wave 1, got {leader_status_wave_1}");
    };

    // Authority B equivocates by producing a block that votes for the leader
    let block_refs_round_3: Vec<_> =
        dag_builder.blocks(3u32..=3).iter().map(|b| b.reference()).collect();

    // Authority B index is 1
    let b4_votes_all = VerifiedBlock::new_for_test(
        TestBlock::new(4, 1)
            .set_ancestors(block_refs_round_3)
            .set_timestamp_ms(4 * 1000 + 1_u64)
            .build(),
    );

    let round_4_refs: Vec<_> = dag_builder
        .blocks(4u32..=4)
        .iter()
        .map(|b| if b.author().value() == 1 { b4_votes_all.reference() } else { b.reference() })
        .collect();

    dag_state.write().accept_block(b4_votes_all);

    for block in dag_builder.blocks(5u32..=5).iter() {
        let author_index = block.author().value();
        if author_index == 0 {
            continue;
        }
        let block = VerifiedBlock::new_for_test(
            TestBlock::new(5, author_index as u32)
                .set_ancestors(round_4_refs.clone())
                .set_timestamp_ms(5 * 1000 + author_index as u64)
                .build(),
        );
        dag_state.write().accept_block(block);
    }

    let leader_status_wave_1 = committer.try_direct_decide(leader_wave_1);
    if let LeaderStatus::Commit(_) = leader_status_wave_1 {
        // ok - now committed with equivocating block
    } else {
        panic!("Expected LeaderStatus::Commit for wave 1, got {leader_status_wave_1}");
    };
}

/// Equivocating authority doesn't vote for leader -> skip
#[tokio::test]
async fn test_equivocating_direct_skip() {
    let context = Arc::new(Context::new_for_test(4).0);
    let dag_state =
        Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
    let committer = BaseCommitterBuilder::new(context.clone(), dag_state.clone()).build();

    let dag_str = "DAG {
        Round 0 : { 4 },
        Round 1 : { * },
        Round 2 : { * },
        Round 3 : { * }
        Round 4 : {
            A -> [],
            B -> [],
            C -> [*],
            D -> [*],
        },
        Round 5 : { * },
    }";

    let (_, dag_builder) = parse_dag(dag_str).expect("a DAG should be valid");
    dag_builder.persist_all_blocks(dag_state.clone());

    let leader_round_wave_1 = committer.leader_round(1);
    let leader_wave_1 =
        committer.elect_leader(leader_round_wave_1).expect("there should be a leader for wave 1");

    let leader_status_wave_1 = committer.try_direct_decide(leader_wave_1);
    if let LeaderStatus::Undecided(_) = leader_status_wave_1 {
        // ok
    } else {
        panic!("Expected LeaderStatus::Undecided for wave 1, got {leader_status_wave_1}");
    };

    // Authority C equivocates by producing a block with no ancestors (no vote for leader)
    let c4_votes_none = VerifiedBlock::new_for_test(
        TestBlock::new(4, 2).set_ancestors(vec![]).set_timestamp_ms(4 * 1000 + 2_u64).build(),
    );

    let round_4_refs: Vec<_> = dag_builder
        .blocks(4u32..=4)
        .iter()
        .map(|b| if b.author().value() == 1 { c4_votes_none.reference() } else { b.reference() })
        .collect();

    dag_state.write().accept_block(c4_votes_none);

    for block in dag_builder.blocks(5u32..=5).iter() {
        let author_index = block.author().value();
        if author_index == 0 {
            continue;
        }
        let block = VerifiedBlock::new_for_test(
            TestBlock::new(5, author_index as u32)
                .set_ancestors(round_4_refs.clone())
                .set_timestamp_ms(5 * 1000 + author_index as u64)
                .build(),
        );
        dag_state.write().accept_block(block);
    }

    let leader_status_wave_1 = committer.try_direct_decide(leader_wave_1);
    if let LeaderStatus::Skip(_) = leader_status_wave_1 {
        // ok
    } else {
        panic!("Expected LeaderStatus::Skip for wave 1, got {leader_status_wave_1}");
    };
}
