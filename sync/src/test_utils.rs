// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-network/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the Soma project.

#![allow(unused_imports, clippy::unwrap_used, clippy::expect_used)]

use rand::{SeedableRng as _, rngs::StdRng};
use std::collections::BTreeMap;
use tokio::sync::{broadcast, mpsc};
use types::{
    checkpoints::{
        CertifiedCheckpointSummary, CheckpointContents, CheckpointSequenceNumber, CheckpointSummary,
        EndOfEpochData, FullCheckpointContents, VerifiedCheckpoint, VerifiedCheckpointContents,
    },
    committee::{Authority, Committee, EpochId, get_available_local_address},
    crypto::{AuthorityKeyPair, NetworkKeyPair},
    digests::CheckpointDigest,
    multiaddr::Multiaddr,
    storage::{shared_in_memory_store::SharedInMemoryStore, write_store::WriteStore},
    sync::{
        PeerEvent,
        active_peers::ActivePeers,
        channel_manager::{ChannelManager, ChannelManagerRequest},
    },
    tx_fee::TransactionFee,
};

use crate::{server::P2pService, tonic_gen::p2p_server::P2pServer};

/// Test fixture for generating certified checkpoint chains.
///
/// Analogous to Sui's `CommitteeFixture` in `crates/sui-swarm-config/src/test_utils.rs`.
pub struct CommitteeFixture {
    pub epoch: EpochId,
    pub committee: Committee,
    pub keypairs: Vec<AuthorityKeyPair>,
}

impl CommitteeFixture {
    /// Generate a committee fixture with the given number of validators.
    /// Uses a deterministic seed so tests are reproducible.
    pub fn generate(epoch: EpochId, committee_size: usize) -> Self {
        let (committee, keypairs) = if epoch == 0 {
            Committee::new_simple_test_committee_of_size(committee_size)
        } else {
            // For non-zero epochs, create with epoch 0 then rebuild with correct epoch
            let (committee, keypairs) = Committee::new_simple_test_committee_of_size(committee_size);
            let voting_rights: BTreeMap<_, _> = committee.voting_rights.iter().cloned().collect();
            let authorities: BTreeMap<_, _> =
                committee.authorities.iter().map(|(k, v)| (*k, v.clone())).collect();
            let committee = Committee::new(epoch, voting_rights, authorities);
            (committee, keypairs)
        };

        Self { epoch, committee, keypairs }
    }

    /// Create empty CheckpointContents (no transactions).
    pub fn empty_contents() -> CheckpointContents {
        CheckpointContents::new_with_digests_only_for_tests(std::iter::empty())
    }

    /// Create empty VerifiedCheckpointContents (no transactions).
    pub fn empty_verified_contents() -> VerifiedCheckpointContents {
        let full =
            FullCheckpointContents::new_with_causally_ordered_transactions(std::iter::empty());
        VerifiedCheckpointContents::new_unchecked(full)
    }

    /// Sign a CheckpointSummary with all validators to produce a VerifiedCheckpoint.
    pub fn create_certified_checkpoint(&self, summary: CheckpointSummary) -> VerifiedCheckpoint {
        let certified = CertifiedCheckpointSummary::new_from_keypairs_for_testing(
            summary,
            &self.keypairs,
            &self.committee,
        );
        VerifiedCheckpoint::new_from_verified(certified)
    }

    /// Create the genesis checkpoint (sequence_number = 0).
    pub fn create_root_checkpoint(&self) -> (VerifiedCheckpoint, VerifiedCheckpointContents) {
        assert_eq!(self.epoch, 0, "root checkpoint must be epoch 0");
        let contents = Self::empty_contents();
        let summary = CheckpointSummary::new(
            0, 0, 0, &contents, None, TransactionFee::default(), None, 0, vec![],
        );
        (self.create_certified_checkpoint(summary), Self::empty_verified_contents())
    }

    /// Create a chain of empty checkpoints.
    /// If `previous` is None, starts from genesis (and includes it in the result).
    /// Returns the ordered checkpoint chain.
    pub fn make_empty_checkpoints(
        &self,
        count: usize,
        previous: Option<&VerifiedCheckpoint>,
    ) -> Vec<VerifiedCheckpoint> {
        let mut checkpoints = Vec::with_capacity(count);

        let mut prev = if let Some(p) = previous {
            p.clone()
        } else {
            let (root, _) = self.create_root_checkpoint();
            checkpoints.push(root.clone());
            if count <= 1 {
                return checkpoints;
            }
            root
        };

        let remaining = if previous.is_some() { count } else { count.saturating_sub(1) };

        for i in 0..remaining {
            let seq = prev.sequence_number + 1;
            let contents = Self::empty_contents();
            let summary = CheckpointSummary::new(
                self.epoch,
                seq,
                0,
                &contents,
                Some(*prev.digest()),
                TransactionFee::default(),
                None,
                seq, // Use sequence number as timestamp
                vec![],
            );
            let checkpoint = self.create_certified_checkpoint(summary);
            prev = checkpoint.clone();
            checkpoints.push(checkpoint);
        }

        checkpoints
    }

    /// Create an end-of-epoch checkpoint.
    pub fn make_end_of_epoch_checkpoint(
        &self,
        previous: &VerifiedCheckpoint,
        next_committee: Committee,
    ) -> VerifiedCheckpoint {
        let contents = Self::empty_contents();
        let end_of_epoch_data = EndOfEpochData {
            next_epoch_validator_committee: next_committee,
            next_epoch_protocol_version: 1.into(),
            epoch_commitments: vec![],
        };
        let summary = CheckpointSummary::new(
            self.epoch,
            previous.sequence_number + 1,
            0,
            &contents,
            Some(*previous.digest()),
            TransactionFee::default(),
            Some(end_of_epoch_data),
            previous.sequence_number + 1,
            vec![],
        );
        self.create_certified_checkpoint(summary)
    }

    /// Initialize a SharedInMemoryStore with genesis state.
    pub fn init_store(&self) -> SharedInMemoryStore {
        let (genesis, contents) = self.create_root_checkpoint();
        let store = SharedInMemoryStore::default();
        store.inner_mut().insert_genesis_state(genesis, contents, self.committee.clone());
        store
    }

    /// Initialize a store and populate it with `count` empty checkpoints (including genesis).
    pub fn init_store_with_checkpoints(
        &self,
        count: usize,
    ) -> (SharedInMemoryStore, Vec<VerifiedCheckpoint>) {
        let store = SharedInMemoryStore::default();
        let checkpoints = self.make_empty_checkpoints(count, None);

        // Insert genesis state
        let genesis = checkpoints.first().unwrap().clone();
        store.inner_mut().insert_genesis_state(
            genesis,
            Self::empty_verified_contents(),
            self.committee.clone(),
        );

        // Insert remaining checkpoints with contents
        for cp in checkpoints.iter().skip(1) {
            store.inner_mut().insert_checkpoint(cp);
            store.inner_mut().insert_checkpoint_contents(cp, Self::empty_verified_contents());
            store.inner_mut().update_highest_synced_checkpoint(cp);
        }

        (store, checkpoints)
    }
}

// Commented-out ChannelManager test helper -- kept for future integration tests.
// pub(crate) async fn create_test_channel_manager<S>(
//     own_address: Multiaddr,
//     server: P2pServer<P2pService<S>>,
// ) -> (
//     mpsc::Sender<ChannelManagerRequest>,
//     broadcast::Receiver<PeerEvent>,
//     ActivePeers,
//     NetworkKeyPair,
// )
// where
//     S: WriteStore + Clone + Send + Sync + 'static,
// {
//     let mut rng = StdRng::from_seed([0; 32]);
//     let active_peers = ActivePeers::new(1000);
//     let network_key_pair = NetworkKeyPair::generate(&mut rng);
//
//     let (manager, tx) = ChannelManager::new(
//         own_address,
//         network_key_pair.clone(),
//         server,
//         active_peers.clone(),
//     );
//     let rx = manager.subscribe();
//     tokio::spawn(manager.start());
//     (tx, rx, active_peers, network_key_pair)
// }
