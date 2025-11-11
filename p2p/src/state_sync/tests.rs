use anyhow::anyhow;
use bytes::Bytes;
use fastcrypto::hash::HashFunction as _;
use fastcrypto::traits::{AllowedRng, KeyPair as _};
use rand::{rngs::StdRng, SeedableRng as _};
use std::fs;
use std::io::Write as _;
use std::sync::Arc;
use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
use tempfile::tempdir;
use types::checkpoint::CommitArchiveData;
use types::config::state_sync_config::StateSyncConfig;
use types::consensus::block::{
    genesis_blocks_from_committee, Block, BlockRef, BlockTimestampMs, SignedBlock, GENESIS_ROUND,
};
use types::consensus::commit::{Commit, CommitIndex, TrustedCommit};
use types::crypto::AuthorityPublicKeyBytes;
use types::storage::read_store::ReadStore as _;
use types::storage::write_store::WriteStore as _;
use types::{
    base::AuthorityName,
    committee::{Authority, Committee, EpochId},
    config::{
        network_config::NetworkConfig,
        node_config::ArchiveReaderConfig,
        object_store_config::ObjectStoreConfig,
        p2p_config::{P2pConfig, SeedPeer},
    },
    consensus::{
        block::{Round, TestBlock, VerifiedBlock},
        commit::{CommitDigest, CommitRef, CommittedSubDag},
        committee::get_available_local_address,
    },
    crypto::{AuthorityKeyPair, NetworkKeyPair, ProtocolKeyPair},
    multiaddr::Multiaddr,
    storage::write_store::TestP2pStore,
};

use crate::{
    builder::P2pBuilder, state_sync::PeerStateSyncInfo, test_utils::create_test_channel_manager,
};

pub struct CommitteeFixture {
    epoch: EpochId,
    committee: Committee,
    key_pairs: Vec<(AuthorityPublicKeyBytes, ProtocolKeyPair)>,
}

impl CommitteeFixture {
    pub fn generate<R: ::rand::RngCore + ::rand::CryptoRng + AllowedRng>(
        mut rng: R,
        epoch: EpochId,
        committee_size: usize,
    ) -> Self {
        let mut authorities = BTreeMap::new();
        let mut voting_weights = BTreeMap::new();
        let mut key_pairs = Vec::new();

        for i in 0..committee_size {
            let authority_keypair = AuthorityKeyPair::generate(&mut rng);
            let protocol_keypair = ProtocolKeyPair::generate(&mut rng);
            let network_keypair = NetworkKeyPair::generate(&mut rng);

            let name = AuthorityName::from(authority_keypair.public());

            authorities.insert(
                name,
                Authority {
                    stake: 1,
                    address: get_available_local_address(),
                    hostname: format!("test_host_{i}"),
                    authority_key: authority_keypair.public().clone(),
                    protocol_key: protocol_keypair.public(),
                    network_key: network_keypair.public(),
                },
            );

            voting_weights.insert(name, 1);

            // Store the protocol key pair for signing blocks
            key_pairs.push((name, protocol_keypair));
        }

        let committee = Committee::new_for_testing_with_normalized_voting_power(
            epoch,
            voting_weights,
            authorities,
        );

        Self {
            epoch,
            committee,
            key_pairs,
        }
    }

    pub fn key_pairs(&self) -> &[(AuthorityPublicKeyBytes, ProtocolKeyPair)] {
        &self.key_pairs
    }

    pub fn committee(&self) -> &Committee {
        &self.committee
    }
}

pub fn empty_commit_blocks(
    committee: &Committee,
    key_pairs: &[(AuthorityPublicKeyBytes, ProtocolKeyPair)],
    round: u32,
    now: BlockTimestampMs,
    ancestors: Vec<BlockRef>,
    commit_ref: CommitRef,
) -> Vec<VerifiedBlock> {
    if round == GENESIS_ROUND {
        return genesis_blocks_from_committee(Arc::new(committee.clone()));
    }

    committee
        .authorities()
        .map(|(index, authority)| {
            // Find the key pair that matches this authority
            let key_pair = key_pairs
                .iter()
                .find(|(name, _)| *name == AuthorityName::from(&authority.authority_key))
                .map(|(_, kp)| kp)
                .expect("Key pair not found for authority");

            // Arrange ancestors correctly: own block first, then others
            let mut block_ancestors = Vec::new();

            // First, find our own previous block (same author)
            if let Some(own_ancestor) = ancestors.iter().find(|a| a.author == index) {
                block_ancestors.push(*own_ancestor);
            }

            // Then add other ancestors (different authors)
            for ancestor in &ancestors {
                if ancestor.author != index {
                    block_ancestors.push(*ancestor);
                }
            }

            let block = Block::new(
                committee.epoch(),
                round,
                index,
                now,
                block_ancestors, // Use properly ordered ancestors
                vec![],
                vec![commit_ref],
                None,
            );

            let signed_block = SignedBlock::new(block, key_pair).expect("Failed to sign block");
            let serialized = signed_block
                .serialize()
                .expect("Failed to serialize signed block");

            VerifiedBlock::new_verified(signed_block, serialized)
        })
        .collect()
}

pub fn make_empty_commits(
    committee: &Committee,
    key_pairs: &[(AuthorityPublicKeyBytes, ProtocolKeyPair)],
    number_of_commits: usize,
    previous_commit: Option<CommittedSubDag>,
) -> (Vec<CommittedSubDag>, Vec<Vec<VerifiedBlock>>) {
    let mut commits = Vec::new();
    let mut all_blocks = Vec::new();

    // Handle genesis case specially
    let (mut prev_digest, mut prev_index, mut prev_round, start_i) =
        if let Some(prev) = previous_commit {
            (
                prev.commit_ref.digest,
                prev.commit_ref.index,
                prev.leader.round,
                0,
            )
        } else {
            // Create genesis commit (index 0) using standard genesis blocks
            let genesis_blocks = genesis_blocks_from_committee(Arc::new(committee.clone()));
            let genesis_leader = genesis_blocks[0].reference();

            let genesis_commit = CommittedSubDag::new(
                genesis_leader,
                genesis_blocks.clone(),
                0,                                    // Genesis timestamp
                CommitRef::new(0, CommitDigest::MIN), // Genesis has index 0
                CommitDigest::MIN,                    // Genesis has MIN previous_digest
            );

            commits.push(genesis_commit.clone());
            all_blocks.push(genesis_blocks.clone());

            (genesis_commit.commit_ref.digest, 0, GENESIS_ROUND, 1)
        };

    // Create remaining commits
    for i in start_i..number_of_commits {
        let round = prev_round + 1;
        let timestamp_ms = round as u64 * 1000;

        // Use previous blocks as ancestors
        let ancestors: Vec<BlockRef> = if !all_blocks.is_empty() {
            all_blocks
                .last()
                .unwrap()
                .iter()
                .map(|b| b.reference())
                .collect()
        } else {
            genesis_blocks_from_committee(Arc::new(committee.clone()))
                .iter()
                .map(|b| b.reference())
                .collect()
        };

        let commit_index = prev_index + 1;

        // Use a deterministic digest for testing
        // In production, this would be computed from actual blocks
        let digest = {
            use types::crypto::DefaultHash;
            let mut hasher = DefaultHash::new();
            hasher.update(&commit_index.to_le_bytes());
            hasher.update(&prev_digest.into_inner());
            hasher.update(&timestamp_ms.to_le_bytes());
            CommitDigest::new(hasher.finalize().into())
        };

        let commit_ref = CommitRef::new(commit_index, digest);

        // Create blocks only once with the correct commit ref
        let blocks = empty_commit_blocks(
            committee,
            key_pairs,
            round,
            timestamp_ms,
            ancestors,
            commit_ref,
        );

        // Now create a TrustedCommit to store the ACTUAL digest that matches the blocks
        let leader = blocks[0].reference();
        let block_refs: Vec<BlockRef> = blocks.iter().map(|b| b.reference()).collect();

        let commit = Commit::new(
            commit_index,
            prev_digest,
            timestamp_ms,
            leader,
            block_refs,
            committee.epoch(),
        );

        let serialized = commit.serialize().expect("Failed to serialize commit");
        let actual_digest = TrustedCommit::compute_digest(&serialized);

        // Update our digest to the actual one
        let actual_commit_ref = CommitRef::new(commit_index, actual_digest);

        let committed_subdag = CommittedSubDag::new(
            leader,
            blocks.clone(),
            timestamp_ms,
            actual_commit_ref,
            prev_digest,
        );

        prev_digest = actual_digest;
        prev_index = commit_index;
        prev_round = round;

        all_blocks.push(blocks);
        commits.push(committed_subdag);
    }

    (commits, all_blocks)
}

pub fn populate_archive_directory(
    temp_dir: &Path,
    commits: &[CommittedSubDag],
    certifier_blocks_per_commit: &[Vec<VerifiedBlock>],
) -> anyhow::Result<()> {
    for (idx, commit) in commits.iter().enumerate() {
        // Skip genesis commit (index 0) from archive since nodes should always have it
        if commit.commit_ref.index == 0 {
            continue;
        }

        // Create TrustedCommit with exact same data as in the CommittedSubDag
        let trusted_commit = TrustedCommit::new_for_test(
            commit.commit_ref.index,
            commit.previous_digest,
            commit.timestamp_ms,
            commit.leader,
            commit.blocks.iter().map(|b| b.reference()).collect(),
            commit.epoch(),
        );

        // The digest MUST match what's in commit.commit_ref.digest
        let computed_digest = TrustedCommit::compute_digest(trusted_commit.serialized());
        assert_eq!(
            computed_digest, commit.commit_ref.digest,
            "Digest mismatch for commit {}",
            commit.commit_ref.index
        );

        let archive_data = CommitArchiveData {
            commit: trusted_commit.serialized().clone(),
            blocks: commit
                .blocks
                .iter()
                .map(|b| b.serialized().clone())
                .collect(),
            certifier_blocks: certifier_blocks_per_commit
                .get(idx)
                .map(|blocks| blocks.iter().map(|b| b.serialized().clone()).collect())
                .unwrap_or_default(),
        };

        let serialized = bcs::to_bytes(&archive_data)?;
        let file_path = temp_dir.join(format!("{}.dat", commit.commit_ref.index));
        let mut file = fs::File::create(file_path)?;
        file.write_all(&serialized)?;
    }

    Ok(())
}

#[tokio::test]
async fn test_state_sync_using_archive() -> anyhow::Result<()> {
    // Generate committee fixture with deterministic keys
    let committee_fixture = CommitteeFixture::generate(StdRng::from_seed([0; 32]), 0, 4);
    let committee = committee_fixture.committee();
    let key_pairs = committee_fixture.key_pairs();

    // Generate test commits with properly signed blocks
    let (ordered_commits, blocks_per_commit) = make_empty_commits(committee, key_pairs, 100, None);

    let temp_dir = tempdir()?;
    // TODO: change this to 10 after you fix the commit votes
    let oldest_commit_to_keep: u64 = 100;

    // Create certifier blocks for each commit (for stake verification in archive)
    let certifier_blocks_per_commit: Vec<Vec<VerifiedBlock>> = ordered_commits
        .iter()
        .enumerate()
        .map(|(idx, commit)| {
            if idx == 0 {
                // For genesis, return empty certifier blocks
                return vec![];
            }

            let round = commit.leader.round + 1;
            let timestamp = (round as u64) * 1000;

            // Use blocks from current commit as ancestors
            let ancestors: Vec<BlockRef> = blocks_per_commit[idx]
                .iter()
                .map(|b| b.reference())
                .collect();

            committee
                .authorities()
                .map(|(auth_idx, authority)| {
                    let key_pair = key_pairs
                        .iter()
                        .find(|(name, _)| *name == AuthorityName::from(&authority.authority_key))
                        .map(|(_, kp)| kp)
                        .expect("Key pair not found for authority");

                    // Arrange ancestors correctly for this authority
                    let mut block_ancestors = Vec::new();

                    // First, add own previous block
                    if let Some(own_ancestor) = ancestors.iter().find(|a| a.author == auth_idx) {
                        block_ancestors.push(*own_ancestor);
                    }

                    // Then add others
                    for ancestor in &ancestors {
                        if ancestor.author != auth_idx {
                            block_ancestors.push(*ancestor);
                        }
                    }

                    let block = Block::new(
                        committee.epoch(),
                        round,
                        auth_idx,
                        timestamp,
                        block_ancestors, // Properly ordered ancestors
                        vec![],
                        vec![commit.commit_ref],
                        None,
                    );

                    let signed_block =
                        SignedBlock::new(block, key_pair).expect("Failed to sign certifier block");
                    let serialized = signed_block
                        .serialize()
                        .expect("Failed to serialize certifier block");

                    VerifiedBlock::new_verified(signed_block, serialized)
                })
                .collect()
        })
        .collect();

    // Populate archive directory
    populate_archive_directory(
        &temp_dir.path(),
        &ordered_commits,
        &certifier_blocks_per_commit,
    )?;

    let archive_reader_config = ArchiveReaderConfig {
        remote_store_config: ObjectStoreConfig::default(),
        download_concurrency: NonZeroUsize::new(1).unwrap(),
        ingestion_url: Some(format!("file://{}", temp_dir.path().display())),
        remote_store_options: vec![],
    };

    // Setup peer 2 with all data initially
    let store_2 = TestP2pStore::new();
    store_2.insert_committee(committee.clone())?;
    store_2.insert_genesis_state(ordered_commits[0].clone(), committee.clone());
    for commit in &ordered_commits[1..] {
        store_2.insert_commit_with_blocks(commit);
        store_2.update_highest_synced_commit(commit)?;
    }

    // Prune first 10 commits from peer 2 (but not genesis at index 0)
    for i in 1..oldest_commit_to_keep {
        store_2.delete_commit_content_test_only(i as CommitIndex)?;
    }

    // Verify peer 2 state
    assert_eq!(
        store_2.get_lowest_available_commit_inner() as u64,
        oldest_commit_to_keep
    );
    assert_eq!(
        store_2
            .get_highest_synced_commit_inner()
            .unwrap()
            .commit_ref
            .index,
        ordered_commits.last().unwrap().commit_ref.index
    );

    // Setup peer 1 with archive config
    let store_1 = TestP2pStore::new();
    // IMPORTANT: Insert the SAME committee that was used to sign the blocks
    store_1.insert_committee(committee.clone())?;
    store_1.insert_genesis_state(ordered_commits[0].clone(), committee.clone());

    assert!(
        store_1.get_commit_by_index(0).is_some(),
        "Store 1 should have genesis at index 0"
    );
    assert!(
        store_2.get_commit_by_index(0).is_some(),
        "Store 2 should have genesis at index 0"
    );

    let store_1_clone = store_1.clone(); // Clone for verification later

    // Build and connect two nodes where Node 1 will be given access to an archive store
    // Node 2 will prune older checkpoints, so Node 1 is forced to backfill from the archive
    let peer_1_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1234".parse().unwrap();
    let peer_2_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1235".parse().unwrap();

    let config2 = P2pConfig {
        external_address: Some(peer_2_addr.clone()),
        state_sync: Some(StateSyncConfig::default()),
        ..Default::default()
    };

    let (disc_builder_2, ss_builder_2, server_2) =
        P2pBuilder::new().store(store_2).config(config2).build();
    let (manager_tx_2, mut events_2, active_peers_2, network_key_pair_2) =
        create_test_channel_manager(peer_2_addr.clone(), server_2).await;

    // Setup first peer with second as seed
    let config1 = P2pConfig {
        external_address: Some(peer_1_addr.clone()),
        seed_peers: vec![SeedPeer {
            peer_id: Some(network_key_pair_2.public().into()),
            address: peer_2_addr.clone(),
        }],
        state_sync: Some(StateSyncConfig::default()),
        ..Default::default()
    };

    let (disc_builder_1, ss_builder_1, server_1) = P2pBuilder::new()
        .store(store_1)
        .config(config1)
        .archive_config(Some(archive_reader_config))
        .build();
    let (manager_tx_1, mut events_1, active_peers_1, network_key_pair_1) =
        create_test_channel_manager(peer_1_addr.clone(), server_1).await;

    // Start discovery loops
    let (disc_event_loop_1, _) = disc_builder_1.build(
        active_peers_1.clone(),
        manager_tx_1,
        network_key_pair_1.clone(),
    );
    let (disc_event_loop_2, _) = disc_builder_2.build(
        active_peers_2.clone(),
        manager_tx_2,
        network_key_pair_2.clone(),
    );

    // Start state sync loops
    let (ss_event_loop_1, _ss_handle_1) = ss_builder_1.build(active_peers_1, events_1);
    let (ss_event_loop_2, _ss_handle_2) = ss_builder_2.build(active_peers_2, events_2);

    // Node 1 will know that Node 2 has the data starting checkpoint 10
    ss_event_loop_1.peer_heights.write().peers.insert(
        network_key_pair_2.public().into(),
        PeerStateSyncInfo {
            genesis_commit_digest: ordered_commits[0].commit_ref.digest,
            height: ordered_commits.last().unwrap().commit_ref.index,
            lowest: oldest_commit_to_keep.try_into().unwrap(),
        },
    );

    tokio::spawn(disc_event_loop_1.start());
    tokio::spawn(disc_event_loop_2.start());

    tokio::spawn(ss_event_loop_1.start());
    // tokio::spawn(ss_event_loop_2.start());

    // Verification loop - wait for Node 1 to sync from archive
    let total_time = Instant::now();
    loop {
        // Check if Node 1 has synced to the latest commit
        if let Ok(highest_synced) = store_1_clone.get_highest_synced_commit() {
            if highest_synced.commit_ref.index == ordered_commits.last().unwrap().commit_ref.index {
                // Node 1 is fully synced to the latest commit on Node 2

                // Verify all commits are present
                for expected_commit in &ordered_commits {
                    let actual_commit = store_1_clone
                        .get_commit_by_index(expected_commit.commit_ref.index)
                        .expect(&format!(
                            "Missing commit {}",
                            expected_commit.commit_ref.index
                        ));

                    assert_eq!(
                        actual_commit.commit_ref, expected_commit.commit_ref,
                        "Commit mismatch at index {}",
                        expected_commit.commit_ref.index
                    );
                }

                // Verify commits that should have been synced from archive (1-9)
                for i in 1..oldest_commit_to_keep {
                    let commit = store_1_clone
                        .get_commit_by_index(i as CommitIndex)
                        .expect(&format!("Missing archive-synced commit {}", i));
                    assert_eq!(
                        commit.commit_ref.index, i as CommitIndex,
                        "Archive sync failed for commit {}",
                        i
                    );
                }

                // Verify commits that should have been synced from peer (10+)
                for i in
                    oldest_commit_to_keep..=ordered_commits.last().unwrap().commit_ref.index.into()
                {
                    let commit = store_1_clone
                        .get_commit_by_index(i.try_into().unwrap())
                        .expect(&format!("Missing peer-synced commit {}", i));
                    assert_eq!(
                        commit.commit_ref.index, i as u32,
                        "Peer sync failed for commit {}",
                        i
                    );
                }

                break;
            }
        }

        // Check for timeout
        if total_time.elapsed() > Duration::from_secs(30) {
            // Print diagnostic information
            if let Ok(highest) = store_1_clone.get_highest_synced_commit() {
                eprintln!(
                    "Test timed out. Node 1 highest synced: {}",
                    highest.commit_ref.index
                );
            } else {
                eprintln!("Test timed out. Node 1 has no highest synced commit");
            }
            return Err(anyhow!("Test timed out after 30 seconds"));
        }

        // Sleep briefly before checking again
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
