// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Builder for creating synthetic `Checkpoint` data for handler unit tests.
//!
//! This avoids needing a running network or real cryptographic signing while still
//! producing structurally valid checkpoints that the indexer handlers can process.

use std::collections::{BTreeMap, BTreeSet};

use crate::SYSTEM_STATE_OBJECT_ID;
use crate::base::{ExecutionDigests, SomaAddress};
use crate::checkpoints::{
    CertifiedCheckpointSummary, CheckpointContents, CheckpointSequenceNumber, CheckpointSummary,
    EndOfEpochData,
};
use crate::committee::{Committee, EpochId};
use crate::crypto::AuthorityKeyPair;
use crate::digests::{CheckpointDigest, ObjectDigest, TransactionDigest, TransactionEffectsDigest};
use crate::effects::object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
use crate::effects::{ExecutionStatus, TransactionEffects, TransactionEffectsV1};
use crate::full_checkpoint_content::{Checkpoint, ExecutedTransaction, ObjectSet};
use crate::object::{
    CoinType, OBJECT_START_VERSION, Object, ObjectData, ObjectID, ObjectType, Owner, Version,
};
use crate::bridge::{BridgeCommittee, MarketplaceParameters};
use crate::system_state::{SystemState, SystemStateTrait as _};
use crate::transaction::{
    ChangeEpoch, GenesisTransaction, TransactionData, TransactionKind,
};
use crate::tx_fee::TransactionFee;

/// Builder for creating synthetic `Checkpoint` data.
///
/// Tracks objects across transactions and assigns consistent versions.
pub struct TestCheckpointBuilder {
    sequence_number: CheckpointSequenceNumber,
    epoch: EpochId,
    timestamp_ms: u64,
    previous_digest: Option<CheckpointDigest>,
    end_of_epoch_data: Option<EndOfEpochData>,
    network_total_transactions: u64,
    transactions: Vec<TestTransaction>,
    committee: Committee,
    keypairs: Vec<AuthorityKeyPair>,
    /// Running version counter — each transaction bumps this.
    next_version: u64,
    /// Optional system state to include in a genesis transaction at checkpoint 0.
    /// This is prepended as the first transaction in `build()` so that handlers like
    /// `kv_epoch_starts` and `soma_models` can find it at `transactions[0]`.
    genesis_system_state: Option<SystemState>,
}

struct TestTransaction {
    tx_data: TransactionData,
    effects: TransactionEffects,
    /// Objects that should appear in the ObjectSet as inputs to this tx.
    input_objects: Vec<Object>,
    /// Objects that should appear in the ObjectSet as outputs of this tx.
    output_objects: Vec<Object>,
}

impl TestCheckpointBuilder {
    /// Create a new builder for a checkpoint with the given sequence number.
    pub fn new(sequence_number: CheckpointSequenceNumber) -> Self {
        let (committee, keypairs) = Committee::new_simple_test_committee_of_size(4);
        Self {
            sequence_number,
            epoch: 0,
            timestamp_ms: 1_000_000,
            previous_digest: if sequence_number > 0 {
                Some(CheckpointDigest::random())
            } else {
                None
            },
            end_of_epoch_data: None,
            network_total_transactions: 0,
            transactions: Vec::new(),
            committee,
            keypairs,
            next_version: 2, // version 1 is OBJECT_START_VERSION
            genesis_system_state: None,
        }
    }

    pub fn with_epoch(mut self, epoch: EpochId) -> Self {
        self.epoch = epoch;
        if epoch > 0 {
            // Rebuild committee with correct epoch
            let (committee, keypairs) = Committee::new_simple_test_committee_of_size(4);
            let voting_rights: BTreeMap<_, _> = committee.voting_rights.iter().cloned().collect();
            let authorities: BTreeMap<_, _> =
                committee.authorities.iter().map(|(k, v)| (*k, v.clone())).collect();
            self.committee = Committee::new(epoch, voting_rights, authorities);
            self.keypairs = keypairs;
        }
        self
    }

    pub fn with_timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    pub fn with_previous_digest(mut self, d: CheckpointDigest) -> Self {
        self.previous_digest = Some(d);
        self
    }

    pub fn with_network_total_transactions(mut self, n: u64) -> Self {
        self.network_total_transactions = n;
        self
    }

    pub fn with_end_of_epoch(mut self, next_committee: Committee) -> Self {
        self.end_of_epoch_data = Some(EndOfEpochData {
            next_epoch_validator_committee: next_committee,
            next_epoch_protocol_version: 1.into(),
            epoch_commitments: vec![],
        });
        self
    }

    /// Include a `SystemState` object in a genesis transaction at checkpoint 0.
    ///
    /// The genesis transaction is prepended as `transactions[0]` in `build()`, which is
    /// where handlers like `kv_epoch_starts` and `soma_models` look for the system state.
    ///
    /// Use [`default_test_system_state`] for a minimal state, or provide a custom one.
    pub fn with_genesis_system_state(mut self, state: SystemState) -> Self {
        self.genesis_system_state = Some(state);
        self
    }

    /// Add a simple coin transfer transaction.
    ///
    /// Creates a new coin owned by `recipient` and records a gas object for `sender`.
    pub fn add_transfer_coin(
        mut self,
        sender: SomaAddress,
        recipient: SomaAddress,
        amount: u64,
    ) -> Self {
        let tx_digest = TransactionDigest::random();
        let version = Version::from_u64(self.next_version);
        self.next_version += 1;

        let coin_id = ObjectID::random();
        let gas_id = ObjectID::random();

        // Gas coin for sender (mutated, pre-existing)
        let gas_input = Object::with_id_owner_coin_for_testing(gas_id, sender, 1_000_000);
        let gas_output = Object::new(
            ObjectData::new_with_id(
                gas_id,
                ObjectType::Coin(CoinType::Soma),
                version,
                bcs::to_bytes(&999_000u64).unwrap(),
            ),
            Owner::AddressOwner(sender),
            tx_digest,
        );

        // New coin for recipient (created)
        let coin_output = Object::new(
            ObjectData::new_with_id(
                coin_id,
                ObjectType::Coin(CoinType::Soma),
                version,
                bcs::to_bytes(&amount).unwrap(),
            ),
            Owner::AddressOwner(recipient),
            tx_digest,
        );

        let gas_input_version = gas_input.version();

        // Stage 13b: TransferCoin variant deleted; this builder
        // emits a BalanceTransfer instead.
        let tx_data = TransactionData::new(
            crate::transaction::TransactionKind::BalanceTransfer(
                crate::transaction::BalanceTransferArgs {
                    coin_type: crate::object::CoinType::Soma,
                    transfers: vec![(recipient, amount)],
                },
            ),
            sender,
            vec![gas_input.compute_object_reference()],
        );

        let changed_objects = vec![
            (
                gas_id,
                EffectsObjectChange {
                    input_state: ObjectIn::Exist((
                        (gas_input_version, ObjectDigest::random()),
                        Owner::AddressOwner(sender),
                    )),
                    output_state: ObjectOut::ObjectWrite((
                        ObjectDigest::random(),
                        Owner::AddressOwner(sender),
                    )),
                    id_operation: IDOperation::None,
                },
            ),
            (
                coin_id,
                EffectsObjectChange {
                    input_state: ObjectIn::NotExist,
                    output_state: ObjectOut::ObjectWrite((
                        ObjectDigest::random(),
                        Owner::AddressOwner(recipient),
                    )),
                    id_operation: IDOperation::Created,
                },
            ),
        ];

        let effects = TransactionEffects::V1(TransactionEffectsV1 {
            status: ExecutionStatus::Success,
            executed_epoch: self.epoch,
            transaction_digest: tx_digest,
            version,
            changed_objects,
            dependencies: vec![],
            unchanged_shared_objects: vec![],
            transaction_fee: TransactionFee::default(),
            gas_object_index: Some(0),
        });

        self.transactions.push(TestTransaction {
            tx_data,
            effects,
            input_objects: vec![gas_input],
            output_objects: vec![gas_output, coin_output],
        });
        self
    }

    /// Add a ChangeEpoch transaction that includes a SystemState in its output objects.
    ///
    /// The system state is provided as-is, so it must already be configured with the
    /// desired model registry, epoch, etc.
    pub fn add_change_epoch(mut self, system_state: SystemState) -> Self {
        let tx_digest = TransactionDigest::random();
        let version = Version::from_u64(self.next_version);
        self.next_version += 1;

        let epoch = system_state.epoch();
        let timestamp_ms = system_state.epoch_start_timestamp_ms();

        let system_bytes = bcs::to_bytes(&system_state).expect("serialize SystemState");
        let system_id = SYSTEM_STATE_OBJECT_ID;
        let data =
            ObjectData::new_with_id(system_id, ObjectType::SystemState, version, system_bytes);
        let system_obj = Object::new(
            data,
            Owner::Shared { initial_shared_version: OBJECT_START_VERSION },
            tx_digest,
        );

        let tx_data = TransactionData::new(
            TransactionKind::ChangeEpoch(ChangeEpoch {
                epoch,
                epoch_start_timestamp_ms: timestamp_ms,
                protocol_version: 1.into(),
                fees: 0,
                epoch_randomness: vec![],
            }),
            SomaAddress::default(),
            vec![],
        );

        let changed_objects = vec![(
            system_id,
            EffectsObjectChange {
                input_state: ObjectIn::NotExist,
                output_state: ObjectOut::ObjectWrite((
                    ObjectDigest::random(),
                    Owner::Shared { initial_shared_version: OBJECT_START_VERSION },
                )),
                id_operation: IDOperation::Created,
            },
        )];

        let effects = TransactionEffects::V1(TransactionEffectsV1 {
            status: ExecutionStatus::Success,
            executed_epoch: self.epoch,
            transaction_digest: tx_digest,
            version,
            changed_objects,
            dependencies: vec![],
            unchanged_shared_objects: vec![],
            transaction_fee: TransactionFee::default(),
            gas_object_index: None,
        });

        self.transactions.push(TestTransaction {
            tx_data,
            effects,
            input_objects: vec![],
            output_objects: vec![system_obj],
        });
        self
    }

    /// Build the final `Checkpoint`.
    pub fn build(mut self) -> Checkpoint {
        // If a genesis system state was provided, prepend a genesis transaction
        // containing the SystemState object as the first transaction.
        if let Some(system_state) = self.genesis_system_state.take() {
            let tx_digest = TransactionDigest::random();
            let version = Version::from_u64(self.next_version);
            self.next_version += 1;

            let system_bytes = bcs::to_bytes(&system_state).expect("serialize SystemState");
            let system_id = SYSTEM_STATE_OBJECT_ID;
            let data =
                ObjectData::new_with_id(system_id, ObjectType::SystemState, version, system_bytes);
            let system_obj = Object::new(
                data,
                Owner::Shared { initial_shared_version: OBJECT_START_VERSION },
                tx_digest,
            );

            let tx_data = TransactionData::new(
                TransactionKind::Genesis(GenesisTransaction { objects: vec![] }),
                SomaAddress::default(),
                vec![],
            );

            let changed_objects = vec![(
                system_id,
                EffectsObjectChange {
                    input_state: ObjectIn::NotExist,
                    output_state: ObjectOut::ObjectWrite((
                        ObjectDigest::random(),
                        Owner::Shared { initial_shared_version: OBJECT_START_VERSION },
                    )),
                    id_operation: IDOperation::Created,
                },
            )];

            let effects = TransactionEffects::V1(TransactionEffectsV1 {
                status: ExecutionStatus::Success,
                executed_epoch: self.epoch,
                transaction_digest: tx_digest,
                version,
                changed_objects,
                dependencies: vec![],
                unchanged_shared_objects: vec![],
                transaction_fee: TransactionFee::default(),
                gas_object_index: None,
            });

            self.transactions.insert(
                0,
                TestTransaction {
                    tx_data,
                    effects,
                    input_objects: vec![],
                    output_objects: vec![system_obj],
                },
            );
        }

        let num_txs = self.transactions.len();
        // Use num_txs as the total. Handlers compute `first_tx = network_total_transactions - len(transactions)`,
        // so total must always be >= the actual transaction count.
        let total_txs = if self.network_total_transactions >= num_txs as u64 {
            self.network_total_transactions
        } else {
            num_txs as u64
        };

        // Build execution digests for checkpoint contents
        let exec_digests: Vec<ExecutionDigests> = self
            .transactions
            .iter()
            .map(|tx| ExecutionDigests {
                transaction: tx.tx_data.digest(),
                effects: TransactionEffectsDigest::random(),
            })
            .collect();

        let contents =
            CheckpointContents::new_with_digests_only_for_tests(exec_digests.into_iter());

        let summary = CheckpointSummary::new(
            self.epoch,
            self.sequence_number,
            total_txs,
            &contents,
            self.previous_digest,
            TransactionFee::default(),
            self.end_of_epoch_data,
            self.timestamp_ms,
            vec![],
        );

        let certified = CertifiedCheckpointSummary::new_from_keypairs_for_testing(
            summary,
            &self.keypairs,
            &self.committee,
        );

        // Build the object set
        let mut object_set = ObjectSet::default();
        let mut executed_transactions = Vec::with_capacity(num_txs);

        for tx in self.transactions {
            for obj in tx.input_objects {
                object_set.insert(obj);
            }
            for obj in tx.output_objects {
                object_set.insert(obj);
            }

            executed_transactions.push(ExecutedTransaction {
                transaction: tx.tx_data,
                signatures: vec![],
                effects: tx.effects,
            });
        }

        Checkpoint { summary: certified, contents, transactions: executed_transactions, object_set }
    }
}

/// Create a minimal `SystemState` suitable for synthetic checkpoint tests.
///
/// The state has no validators, an empty model registry, and default parameters.
/// This is enough for handlers like `kv_epoch_starts` and `soma_models` to process
/// the checkpoint without errors.
pub fn default_test_system_state() -> SystemState {
    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        protocol_config::ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );
    SystemState::create(
        vec![], // no validators needed for indexer tests
        protocol_config::ProtocolVersion::MAX.as_u64(),
        1_000_000, // epoch_start_timestamp_ms
        &protocol_config,
        0,                       // emission_fund
        100_000_000_000_000,     // emission_initial_distribution_amount (100K SOMA in shannons)
        10,                      // emission_period_length
        1000,                    // emission_decrease_rate (10% in bps)
        None,
        MarketplaceParameters::default(),
        BridgeCommittee::empty(),
    )
}

