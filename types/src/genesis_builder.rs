use fastcrypto::traits::KeyPair as _;
use protocol_config::ProtocolVersion;

use crate::base::ExecutionDigests;
use crate::config::encoder_config::EncoderGenesisConfig;
use crate::config::genesis_config::{
    GenesisConfig, TokenAllocation, TokenDistributionSchedule, ValidatorGenesisConfig,
};
use crate::envelope::Message as _;
use crate::intent::{IntentMessage, IntentScope};
use crate::object::{ObjectData, ObjectType, Version};
use crate::system_state::encoder::Encoder;
use crate::system_state::epoch_start::EpochStartSystemStateTrait as _;
use crate::system_state::shard::{Target, TargetOrigin};
use crate::system_state::validator::Validator;
use crate::system_state::{get_system_state, FeeParameters, SystemParameters};
use crate::transaction::InputObjects;
use crate::tx_fee::TransactionFee;
use crate::{
    base::SomaAddress,
    checkpoints::{CertifiedCheckpointSummary, CheckpointContents, CheckpointSummary},
    committee::Committee,
    crypto::{
        AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo, AuthorityStrongQuorumSignInfo,
    },
    digests::TransactionDigest,
    effects::{ExecutionStatus, TransactionEffects},
    genesis::{Genesis, UnsignedGenesis},
    intent::Intent,
    object::{Object, ObjectID, Owner},
    system_state::{SystemState, SystemStateTrait},
    temporary_store::TemporaryStore,
    transaction::{Transaction, VerifiedTransaction},
    SYSTEM_STATE_OBJECT_ID,
};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

pub struct GenesisBuilder {
    parameters: GenesisConfig,
    token_distribution_schedule: Option<TokenDistributionSchedule>,
    validators: Vec<ValidatorGenesisConfig>,
    encoders: Vec<EncoderGenesisConfig>,
    networking_validators: Vec<ValidatorGenesisConfig>,
    signatures: BTreeMap<AuthorityPublicKeyBytes, AuthoritySignInfo>,
    built_genesis: Option<UnsignedGenesis>,
}

impl GenesisBuilder {
    pub fn new() -> Self {
        Self {
            parameters: GenesisConfig::for_local_testing(),
            token_distribution_schedule: None,
            validators: Vec::new(),
            networking_validators: Vec::new(),
            encoders: Vec::new(),
            signatures: BTreeMap::new(),
            built_genesis: None,
        }
    }

    pub fn with_parameters(mut self, parameters: GenesisConfig) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_validators(mut self, validators: Vec<ValidatorGenesisConfig>) -> Self {
        self.validators = validators;
        self
    }

    pub fn with_networking_validators(mut self, validators: Vec<ValidatorGenesisConfig>) -> Self {
        self.networking_validators = validators;
        self
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderGenesisConfig>) -> Self {
        self.encoders = encoders;
        self
    }

    pub fn with_token_distribution_schedule(mut self, schedule: TokenDistributionSchedule) -> Self {
        self.token_distribution_schedule = Some(schedule);
        self
    }

    pub fn with_protocol_version(mut self, v: ProtocolVersion) -> Self {
        self.parameters.parameters.protocol_version = v;
        self
    }

    pub fn protocol_version(&self) -> ProtocolVersion {
        self.parameters.parameters.protocol_version
    }

    pub fn add_validator_signature(mut self, keypair: &AuthorityKeyPair) -> Self {
        let unsigned_genesis = self.build_unsigned_genesis();
        let name = keypair.public().into();

        // Ensure this validator exists
        assert!(
            self.validators
                .iter()
                .any(|v| AuthorityPublicKeyBytes::from(v.key_pair.public()) == name),
            "provided keypair does not correspond to a validator in the validator set"
        );

        // Sign the checkpoint summary
        let checkpoint_signature = AuthoritySignInfo::new(
            unsigned_genesis.checkpoint.epoch,
            &unsigned_genesis.checkpoint,
            Intent::soma_app(IntentScope::CheckpointSummary),
            name,
            keypair,
        );

        self.signatures.insert(name, checkpoint_signature);
        self
    }

    pub fn build_unsigned_genesis(&mut self) -> UnsignedGenesis {
        if let Some(built) = &self.built_genesis {
            return built.clone();
        }

        let (system_state, objects) = self.create_genesis_state();
        let (transaction, effects, final_objects) =
            self.create_genesis_transaction(objects, system_state);
        let (checkpoint, checkpoint_contents) =
            self.create_genesis_checkpoint(&transaction, &effects);

        let unsigned = UnsignedGenesis {
            checkpoint,
            checkpoint_contents,
            transaction,
            effects,
            objects: final_objects,
        };

        self.built_genesis = Some(unsigned.clone());
        unsigned
    }

    pub fn build(mut self) -> Genesis {
        let unsigned = self.build_unsigned_genesis();

        // Get committee from system state
        let system_state = get_system_state(&unsigned.objects()).expect("System state must exist");
        let committee = system_state.into_epoch_start_state().get_committee();

        // Collect all signatures
        let signatures: Vec<AuthoritySignInfo> = self.signatures.into_values().collect();

        // Create certified checkpoint
        let certified_checkpoint =
            CertifiedCheckpointSummary::new(unsigned.checkpoint, signatures, &committee).unwrap();

        Genesis::new(
            certified_checkpoint,
            unsigned.checkpoint_contents,
            unsigned.transaction,
            unsigned.effects,
            unsigned.objects,
        )
    }

    fn create_genesis_state(&self) -> (SystemState, Vec<Object>) {
        let mut objects = Vec::new();

        // Create system state with validators and encoders
        let mut system_state = SystemState::create(
            self.validators
                .iter()
                .map(|v| {
                    Validator::new(
                        (&v.account_key_pair.public()).into(),
                        (v.key_pair.public()).clone(),
                        v.network_key_pair.public().into(),
                        v.worker_key_pair.public().into(),
                        v.network_address.clone(),
                        v.consensus_address.clone(),
                        v.network_address.clone(),
                        v.encoder_validator_address.clone(),
                        0,
                        v.commission_rate,
                        ObjectID::random(),
                    )
                })
                .collect(),
            self.networking_validators
                .iter()
                .map(|v| {
                    Validator::new(
                        (&v.account_key_pair.public()).into(),
                        (v.key_pair.public()).clone(),
                        v.network_key_pair.public().into(),
                        v.worker_key_pair.public().into(),
                        v.network_address.clone(),
                        v.consensus_address.clone(),
                        v.network_address.clone(),
                        v.encoder_validator_address.clone(),
                        0,
                        v.commission_rate,
                        ObjectID::random(),
                    )
                })
                .collect(),
            self.encoders
                .iter()
                .map(|e| {
                    Encoder::new(
                        (&e.account_key_pair.public()).into(),
                        e.encoder_key_pair.public().clone(),
                        e.network_key_pair.public().clone(),
                        e.internal_network_address.clone(),
                        e.external_network_address.clone(),
                        e.object_address.clone(),
                        0,
                        e.commission_rate,
                        e.byte_price,
                        ObjectID::random(),
                    )
                })
                .collect(),
            self.parameters.parameters.protocol_version.as_u64(),
            self.parameters.parameters.chain_start_timestamp_ms,
            SystemParameters {
                epoch_duration_ms: self.parameters.parameters.epoch_duration_ms,
                ..Default::default()
            },
            self.token_distribution_schedule
                .as_ref()
                .map(|s| s.emission_fund_shannons)
                .unwrap_or(0),
            self.parameters.parameters.emission_per_epoch,
        );

        // Process token allocations
        if let Some(schedule) = &self.token_distribution_schedule {
            for allocation in &schedule.allocations {
                if let Some(validator) = allocation.staked_with_validator {
                    let staked_soma = system_state
                        .request_add_stake_at_genesis(
                            allocation.recipient_address,
                            validator,
                            allocation.amount_shannons,
                        )
                        .expect("Failed to stake at genesis");

                    let staked_object = Object::new_staked_soma_object(
                        ObjectID::random(),
                        staked_soma,
                        Owner::AddressOwner(allocation.recipient_address),
                        TransactionDigest::default(),
                    );
                    objects.push(staked_object);
                } else if let Some(encoder) = allocation.staked_with_encoder {
                    let staked_soma = system_state
                        .request_add_encoder_stake_at_genesis(
                            allocation.recipient_address,
                            encoder,
                            allocation.amount_shannons,
                        )
                        .expect("Failed to stake in encoder at genesis");

                    let staked_object = Object::new_staked_soma_object(
                        ObjectID::random(),
                        staked_soma,
                        Owner::AddressOwner(allocation.recipient_address),
                        TransactionDigest::default(),
                    );
                    objects.push(staked_object);
                } else {
                    // Regular coin object
                    let coin_object = Object::new_coin(
                        ObjectID::random(),
                        allocation.amount_shannons,
                        Owner::AddressOwner(allocation.recipient_address),
                        TransactionDigest::default(),
                    );
                    objects.push(coin_object);
                }
            }
        }

        // === Create seed targets for genesis bootstrap ===
        let embeddings = &self.parameters.parameters.seed_target_embeddings;

        if !embeddings.is_empty() {
            let emission_per_epoch = self.parameters.parameters.emission_per_epoch;
            let target_allocation_bps = system_state.parameters.target_reward_allocation_bps;
            let seed_reward_pool = (emission_per_epoch * target_allocation_bps) / 10_000;

            let seed_count = embeddings.len() as u64;
            let reward_per_target = seed_reward_pool / seed_count;

            for (i, embedding) in embeddings.iter().enumerate() {
                let target = Target {
                    origin: TargetOrigin::Genesis {
                        reward_amount: reward_per_target,
                    },
                    created_epoch: 0,
                    target_embedding: embedding.clone(),
                    winning_shard: None,
                };

                let target_object = Object::new(
                    ObjectData::new_with_id(
                        ObjectID::derive_id(TransactionDigest::default(), i as u64),
                        ObjectType::Target,
                        Version::MIN,
                        bcs::to_bytes(&target).unwrap(),
                    ),
                    Owner::Shared {
                        initial_shared_version: Version::new(),
                    },
                    TransactionDigest::default(),
                );
                objects.push(target_object);
            }

            // Deduct from emission pool
            system_state.emission_pool.balance = system_state
                .emission_pool
                .balance
                .saturating_sub(seed_reward_pool);

            // Handle remainder
            let remainder = seed_reward_pool % seed_count;
            if remainder > 0 {
                system_state.emission_pool.balance += remainder;
            }

            tracing::info!(
                "Genesis: Created {} seed targets with {} reward each (total pool: {})",
                seed_count,
                reward_per_target,
                seed_reward_pool
            );
        }

        // Set voting power and build committees
        system_state.validators.set_voting_power();
        system_state.encoders.set_voting_power();

        let current_committees = system_state.build_committees_for_epoch(0);
        system_state.committees[1] = Some(current_committees);

        // Create system state object
        let state_object = Object::new(
            ObjectData::new_with_id(
                SYSTEM_STATE_OBJECT_ID,
                ObjectType::SystemState,
                Version::MIN,
                bcs::to_bytes(&system_state).unwrap(),
            ),
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            TransactionDigest::default(),
        );
        objects.push(state_object);

        (system_state, objects)
    }

    fn create_genesis_transaction(
        &self,
        objects: Vec<Object>,
        system_state: SystemState,
    ) -> (Transaction, TransactionEffects, Vec<Object>) {
        let unsigned_tx =
            VerifiedTransaction::new_genesis_transaction(objects.clone()).into_inner();
        let genesis_digest = *unsigned_tx.digest();

        // Create temporary store for effects generation
        let input_objects = InputObjects::new(Vec::new());
        let mut temp_store = TemporaryStore::new(
            input_objects,
            Vec::new(), // receiving_objects
            genesis_digest,
            0, // epoch_id
            system_state.fee_parameters(),
        );

        // Add all objects to the store
        for object in objects {
            temp_store.create_object(object);
        }

        // Generate effects
        let (inner, effects) = temp_store.into_effects(
            Vec::new(), // shared_object_refs
            &genesis_digest,
            BTreeSet::new(), // dependencies
            ExecutionStatus::Success,
            0, // epoch_id
            TransactionFee::default(),
            None,
        );

        let final_objects = inner.written.into_values().collect();

        (unsigned_tx, effects, final_objects)
    }

    fn create_genesis_checkpoint(
        &self,
        transaction: &Transaction,
        effects: &TransactionEffects,
    ) -> (CheckpointSummary, CheckpointContents) {
        let execution_digests = ExecutionDigests {
            transaction: *transaction.digest(),
            effects: effects.digest(),
        };

        let contents = CheckpointContents::new_with_digests_and_signatures(
            vec![execution_digests],
            vec![vec![]], // No individual signatures for genesis
        );

        let checkpoint = CheckpointSummary {
            epoch: 0,
            sequence_number: 0,
            network_total_transactions: contents.size() as u64,
            content_digest: *contents.digest(),
            previous_digest: None,
            epoch_rolling_transaction_fees: Default::default(),
            end_of_epoch_data: None,
            timestamp_ms: self.parameters.parameters.chain_start_timestamp_ms,
            checkpoint_commitments: Default::default(),
        };

        (checkpoint, contents)
    }

    // TODO: implement load() and save() functions to load and save Genesis from a local blob file.
}
