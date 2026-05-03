// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use camino::Utf8Path;
use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::hash::{Blake2b256, HashFunction};
use fastcrypto::traits::{KeyPair as _, ToFromBytes};
use protocol_config::ProtocolVersion;
use tracing::trace;

use crate::SYSTEM_STATE_OBJECT_ID;
use crate::base::{ExecutionDigests, SomaAddress};
use crate::checkpoints::{CertifiedCheckpointSummary, CheckpointContents, CheckpointSummary};
use crate::config::genesis_config::{
    GenesisCeremonyParameters, TOTAL_SUPPLY_SHANNONS, TokenDistributionSchedule,
    ValidatorGenesisConfig,
};
use crate::crypto::{
    AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo, AuthoritySignInfoTrait as _,
    AuthoritySignature,
};
use crate::digests::TransactionDigest;
use crate::effects::{ExecutionStatus, TransactionEffects};
use crate::envelope::Message as _;
use crate::genesis::{Genesis, UnsignedGenesis};
use crate::intent::{Intent, IntentMessage, IntentScope};
use crate::committee::EpochId;
use crate::object::{CoinType, Object, ObjectData, ObjectID, ObjectType, Owner, Version};
use crate::system_state::epoch_start::EpochStartSystemStateTrait as _;
use crate::system_state::staking::StakingPool;
use crate::system_state::validator::{Validator, ValidatorMetadata};
use crate::bridge::{BridgeCommittee, MarketplaceParameters};
use crate::system_state::{FeeParameters, SystemState, SystemStateTrait, get_system_state};
use crate::temporary_store::TemporaryStore;
use crate::transaction::{InputObjects, Transaction, VerifiedTransaction};
use crate::tx_fee::TransactionFee;
use crate::validator_info::GenesisValidatorInfo;

const GENESIS_BUILDER_COMMITTEE_DIR: &str = "committee";
const GENESIS_BUILDER_PARAMETERS_FILE: &str = "parameters";
const GENESIS_BUILDER_TOKEN_DISTRIBUTION_SCHEDULE_FILE: &str = "token-distribution-schedule";
const GENESIS_BUILDER_SIGNATURE_DIR: &str = "signatures";
const GENESIS_BUILDER_UNSIGNED_GENESIS_FILE: &str = "unsigned-genesis";

pub struct GenesisBuilder {
    parameters: GenesisCeremonyParameters,
    token_distribution_schedule: Option<TokenDistributionSchedule>,
    validators: Vec<GenesisValidatorInfo>,
    signatures: BTreeMap<AuthorityPublicKeyBytes, AuthoritySignInfo>,
    built_genesis: Option<UnsignedGenesis>,
    marketplace_params: Option<MarketplaceParameters>,
    bridge_committee: Option<BridgeCommittee>,
}

impl Default for GenesisBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GenesisBuilder {
    pub fn new() -> Self {
        Self {
            parameters: GenesisCeremonyParameters::default(),
            token_distribution_schedule: None,
            validators: Vec::new(),
            signatures: BTreeMap::new(),
            built_genesis: None,
            marketplace_params: None,
            bridge_committee: None,
        }
    }

    pub fn with_parameters(mut self, parameters: GenesisCeremonyParameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_marketplace_params(mut self, params: MarketplaceParameters) -> Self {
        self.marketplace_params = Some(params);
        self
    }

    pub fn with_bridge_committee(mut self, committee: BridgeCommittee) -> Self {
        self.bridge_committee = Some(committee);
        self
    }

    pub fn with_protocol_version(mut self, v: ProtocolVersion) -> Self {
        self.parameters.protocol_version = v;
        self
    }

    pub fn with_chain(mut self, chain: protocol_config::Chain) -> Self {
        self.parameters.chain = chain;
        self
    }

    pub fn protocol_version(&self) -> ProtocolVersion {
        self.parameters.protocol_version
    }

    pub fn with_token_distribution_schedule(mut self, schedule: TokenDistributionSchedule) -> Self {
        self.token_distribution_schedule = Some(schedule);
        self
    }

    /// Add a consensus validator from GenesisValidatorInfo (ceremony workflow)
    pub fn add_validator(mut self, validator: GenesisValidatorInfo) -> Self {
        self.validators.push(validator);
        self.built_genesis = None;
        self
    }

    // Convenience methods for local testing with config types

    /// Add validators from ValidatorGenesisConfig (local testing workflow)
    pub fn with_validator_configs(mut self, configs: Vec<ValidatorGenesisConfig>) -> Self {
        use crate::crypto::AuthoritySignature;
        use crate::validator_info::ValidatorInfo;

        for config in configs {
            let info = ValidatorInfo::from(&config);

            self.validators.push(GenesisValidatorInfo { info });
        }
        self.built_genesis = None;
        self
    }

    pub fn validators(&self) -> &[GenesisValidatorInfo] {
        &self.validators
    }

    pub fn unsigned_genesis_checkpoint(&self) -> Option<UnsignedGenesis> {
        self.built_genesis.clone()
    }

    pub fn add_validator_signature(mut self, keypair: &AuthorityKeyPair) -> Self {
        let unsigned_genesis = self.build_unsigned_genesis();
        let name: AuthorityPublicKeyBytes = keypair.public().into();

        // Ensure this validator exists
        assert!(
            self.validators.iter().any(|v| v.info.protocol_key == name),
            "provided keypair does not correspond to a validator in the validator set"
        );

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

        self.validate().expect("Genesis validation failed");

        // Sort validators by account address for deterministic genesis output
        self.validators.sort_by_key(|v| v.info.account_address);

        let (system_state, objects, balances, delegations) = self.create_genesis_state();
        let (transaction, effects, final_objects) =
            self.create_genesis_transaction(objects, &system_state);
        let (checkpoint, checkpoint_contents) =
            self.create_genesis_checkpoint(&transaction, &effects);

        let unsigned = UnsignedGenesis {
            checkpoint,
            checkpoint_contents,
            transaction,
            effects,
            objects: final_objects,
            balances,
            delegations,
        };

        self.built_genesis = Some(unsigned.clone());
        unsigned
    }

    pub fn build(mut self) -> Genesis {
        let unsigned = self.build_unsigned_genesis();

        let system_state = get_system_state(&unsigned.objects()).expect("System state must exist");
        let committee = system_state.into_epoch_start_state().get_committee();

        let signatures: Vec<AuthoritySignInfo> = self.signatures.into_values().collect();

        let certified_checkpoint =
            CertifiedCheckpointSummary::new(unsigned.checkpoint, signatures, &committee).unwrap();

        Genesis::new(
            certified_checkpoint,
            unsigned.checkpoint_contents,
            unsigned.transaction,
            unsigned.effects,
            unsigned.objects,
            unsigned.balances,
            unsigned.delegations,
        )
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.validate_inputs()?;
        self.validate_output();
        Ok(())
    }

    fn validate_inputs(&self) -> anyhow::Result<()> {
        for validator in &self.validators {
            validator.validate().with_context(|| {
                format!("metadata for validator {} is invalid", validator.info.account_address)
            })?;
        }

        if let Some(schedule) = &self.token_distribution_schedule {
            schedule.validate();
            schedule.check_all_stake_operations_are_for_valid_validators(
                self.validators.iter().map(|v| v.info.account_address),
            );
        }

        Ok(())
    }

    fn validate_output(&self) {
        let Some(unsigned_genesis) = &self.built_genesis else {
            return;
        };

        let system_state = get_system_state(&unsigned_genesis.objects())
            .expect("System state must exist in genesis");

        assert_eq!(
            self.validators.len(),
            system_state.validators().validators.len(),
            "Validator count mismatch"
        );

        let committee = system_state.into_epoch_start_state().get_committee();
        for signature in self.signatures.values() {
            let validator_exists =
                self.validators.iter().any(|v| v.info.protocol_key == signature.authority);

            if !validator_exists {
                panic!("found signature for unknown validator: {:?}", signature.authority);
            }

            signature
                .verify_secure(
                    unsigned_genesis.checkpoint(),
                    Intent::soma_app(IntentScope::CheckpointSummary),
                    &committee,
                )
                .expect("signature should be valid");
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let path: &Utf8Path = path.try_into()?;
        trace!("Reading Genesis Builder from {}", path);

        if !path.is_dir() {
            bail!("path must be a directory");
        }

        // Load parameters
        let parameters_file = path.join(GENESIS_BUILDER_PARAMETERS_FILE);
        let parameters: GenesisCeremonyParameters = serde_yaml::from_slice(
            &fs::read(&parameters_file).context("unable to read genesis parameters file")?,
        )
        .context("unable to deserialize genesis parameters")?;

        // Load token distribution schedule if present
        let token_distribution_schedule_file =
            path.join(GENESIS_BUILDER_TOKEN_DISTRIBUTION_SCHEDULE_FILE);
        let token_distribution_schedule = if token_distribution_schedule_file.exists() {
            Some(TokenDistributionSchedule::from_csv(fs::File::open(
                token_distribution_schedule_file,
            )?)?)
        } else {
            None
        };

        // Load validators (sorted by account address for deterministic ordering)
        let mut validators = Vec::new();
        let committee_dir = path.join(GENESIS_BUILDER_COMMITTEE_DIR);
        if committee_dir.exists() {
            let mut entries: Vec<_> = committee_dir
                .read_dir_utf8()?
                .filter_map(|e| e.ok())
                .filter(|e| !e.file_name().starts_with('.'))
                .collect();
            entries.sort_by_key(|e| e.file_name().to_string());
            for entry in entries {
                let validator_path = entry.path();
                let validator_bytes = fs::read(validator_path)?;
                let validator_info: GenesisValidatorInfo = serde_yaml::from_slice(&validator_bytes)
                    .with_context(|| {
                        format!("unable to load validator info for {}", validator_path)
                    })?;
                validators.push(validator_info);
            }
        }

        // Load signatures
        let mut signatures = BTreeMap::new();
        let signature_dir = path.join(GENESIS_BUILDER_SIGNATURE_DIR);
        if signature_dir.exists() {
            for entry in signature_dir.read_dir_utf8()? {
                let entry = entry?;
                if entry.file_name().starts_with('.') {
                    continue;
                }
                let sig_path = entry.path();
                let sig_bytes = fs::read(sig_path)?;
                let sig: AuthoritySignInfo = bcs::from_bytes(&sig_bytes).with_context(|| {
                    format!("unable to load validator signature for {}", sig_path)
                })?;
                signatures.insert(sig.authority, sig);
            }
        }

        let mut builder = Self {
            parameters,
            token_distribution_schedule,
            validators,
            signatures,
            built_genesis: None,
            marketplace_params: None,
            bridge_committee: None,
        };

        // Load unsigned genesis if present and verify via BCS comparison
        // (PartialEq comparison would fail due to OnceCell/OnceLock cached digest
        // fields being Uninit after deserialization but populated after building)
        let unsigned_genesis_file = path.join(GENESIS_BUILDER_UNSIGNED_GENESIS_FILE);
        if unsigned_genesis_file.exists() {
            let unsigned_genesis_bytes = fs::read(&unsigned_genesis_file)?;
            let loaded_genesis: UnsignedGenesis = bcs::from_bytes(&unsigned_genesis_bytes)?;

            assert!(
                builder.token_distribution_schedule.is_some(),
                "If a built genesis is present, there must also be a token-distribution-schedule"
            );

            let built = builder.build_unsigned_genesis();
            let built_bytes = bcs::to_bytes(&built).expect("built genesis should serialize");
            assert_eq!(
                built_bytes, unsigned_genesis_bytes,
                "loaded genesis does not match built genesis (BCS comparison)"
            );

            // Use the built version (which has cached digests populated)
            // builder.built_genesis is already set by build_unsigned_genesis()
        }

        Ok(builder)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        trace!("Writing Genesis Builder to {}", path.display());

        fs::create_dir_all(path)?;

        // Write parameters
        let parameters_file = path.join(GENESIS_BUILDER_PARAMETERS_FILE);
        fs::write(parameters_file, serde_yaml::to_string(&self.parameters)?)?;

        // Write token distribution schedule if present
        if let Some(schedule) = &self.token_distribution_schedule {
            schedule.to_csv(fs::File::create(
                path.join(GENESIS_BUILDER_TOKEN_DISTRIBUTION_SCHEDULE_FILE),
            )?)?;
        }

        // Write validators
        let committee_dir = path.join(GENESIS_BUILDER_COMMITTEE_DIR);
        fs::create_dir_all(&committee_dir)?;
        for validator in &self.validators {
            let validator_bytes = serde_yaml::to_string(validator)?;
            fs::write(
                committee_dir.join(validator.info.account_address.to_string()),
                validator_bytes,
            )?;
        }

        // Write signatures
        let signature_dir = path.join(GENESIS_BUILDER_SIGNATURE_DIR);
        fs::create_dir_all(&signature_dir)?;
        for (pubkey, sig) in &self.signatures {
            let sig_bytes = bcs::to_bytes(sig)?;
            let name = self
                .validators
                .iter()
                .find(|v| v.info.protocol_key == *pubkey)
                .map(|v| v.info.account_address.clone().to_string())
                .unwrap_or_else(|| format!("{:?}", pubkey));
            fs::write(signature_dir.join(name), sig_bytes)?;
        }

        // Write unsigned genesis if present
        if let Some(genesis) = &self.built_genesis {
            let genesis_bytes = bcs::to_bytes(genesis)?;
            fs::write(path.join(GENESIS_BUILDER_UNSIGNED_GENESIS_FILE), genesis_bytes)?;
        }

        Ok(())
    }

    /// Generate a deterministic ObjectID for genesis object creation.
    /// Uses Blake2b256(domain || counter) to produce reproducible IDs from the same inputs.
    fn deterministic_object_id(counter: &mut u64) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update(b"soma-genesis-object-id");
        hasher.update(&counter.to_le_bytes());
        *counter += 1;
        let hash = hasher.finalize();
        ObjectID::new(hash.digest[..ObjectID::LENGTH].try_into().unwrap())
    }

    fn create_genesis_state(
        &self,
    ) -> (
        SystemState,
        Vec<Object>,
        BTreeMap<(SomaAddress, CoinType), u64>,
        BTreeMap<(ObjectID, SomaAddress), u64>,
    ) {
        let mut objects = Vec::new();
        // Accumulator-balance entries seeded alongside coin objects. Stake
        // allocations do NOT contribute here — those tokens live in the
        // validator's StakingPool, not the holder's spendable balance.
        let mut balances: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
        // Stage 9d-C1: delegation entries seeded alongside StakedSomaV1
        // objects. F1 row schema is ONE row per (pool, staker), so
        // multiple genesis allocations from the same staker into the
        // same validator collapse into a single principal sum. The
        // table consumer materialises these as `Delegation { principal,
        // last_collected_period: 0 }`.
        let mut delegations: BTreeMap<(ObjectID, SomaAddress), u64> = BTreeMap::new();
        let mut id_counter: u64 = 0;

        let protocol_config = protocol_config::ProtocolConfig::get_for_version(
            self.parameters.protocol_version,
            self.parameters.chain,
        );

        // Convert GenesisValidatorInfo to on-chain Validator
        let validators: Vec<Validator> = self
            .validators
            .iter()
            .map(|v| {
                {
                    let pop = AuthoritySignature::from_bytes(&v.info.proof_of_possession)
                        .expect("Invalid proof of possession in genesis validator");
                    Validator {
                        metadata: ValidatorMetadata {
                            soma_address: v.info.account_address,
                            protocol_pubkey: BLS12381PublicKey::from_bytes(
                                v.info.protocol_key.as_bytes(),
                            )
                            .expect("Invalid protocol key"),
                            network_pubkey: v.info.network_key.clone(),
                            worker_pubkey: v.info.worker_key.clone(),
                            net_address: v.info.network_address.clone(),
                            p2p_address: v.info.p2p_address.clone(),
                            primary_address: v.info.primary_address.clone(),
                            proxy_address: v.info.proxy_address.clone(),
                            proof_of_possession: pop,
                            next_epoch_protocol_pubkey: None,
                            next_epoch_network_pubkey: None,
                            next_epoch_net_address: None,
                            next_epoch_p2p_address: None,
                            next_epoch_primary_address: None,
                            next_epoch_worker_pubkey: None,
                            next_epoch_proof_of_possession: None,
                            next_epoch_proxy_address: None,
                            bridge_ecdsa_pubkey: None,
                            next_epoch_bridge_ecdsa_pubkey: None,
                        },
                        voting_power: 0, // Will be set by set_voting_power()
                        staking_pool: StakingPool::new(Self::deterministic_object_id(
                            &mut id_counter,
                        )),
                        commission_rate: v.info.commission_rate,
                        next_epoch_stake: 0,
                        next_epoch_commission_rate: v.info.commission_rate,
                    }
                }
            })
            .collect();

        // Create system state
        let mut system_state = SystemState::create(
            validators,
            self.parameters.protocol_version.as_u64(),
            self.parameters.chain_start_timestamp_ms,
            &protocol_config,
            self.token_distribution_schedule
                .as_ref()
                .map(|s| s.emission_fund_shannons)
                .unwrap_or(0),
            self.parameters.emission_initial_distribution_amount,
            self.parameters.emission_period_length,
            self.parameters.emission_decrease_rate,
            Some(self.parameters.epoch_duration_ms),
            self.marketplace_params.clone().unwrap_or_default(),
            self.bridge_committee.clone().unwrap_or_else(BridgeCommittee::empty),
        );

        // Process token allocations
        if let Some(schedule) = &self.token_distribution_schedule {
            for allocation in &schedule.allocations {
                if let Some(validator) = allocation.staked_with_validator {
                    // Stage 9d-C5: bump the validator's pool
                    // total_stake; the (pool, staker) row is captured
                    // in the genesis delegations map below.
                    let pool_id = system_state
                        .add_stake_to_validator_at_genesis(validator, allocation.amount_shannons)
                        .expect("Failed to stake with validator at genesis");

                    let key = (pool_id, allocation.recipient_address);
                    let entry = delegations.entry(key).or_insert(0);
                    *entry = entry
                        .checked_add(allocation.amount_shannons)
                        .expect("genesis delegation principal overflow");
                } else {
                    // Stage 13a: SOMA allocations land directly in
                    // the balance accumulator; no Coin object output.
                    // The accumulator is the sole source of truth
                    // for fungible balances post-Stage-13.
                    let entry = balances
                        .entry((allocation.recipient_address, CoinType::Soma))
                        .or_insert(0);
                    *entry = entry
                        .checked_add(allocation.amount_shannons)
                        .expect("genesis SOMA balance overflow");
                }
            }

            // Stage 13a: USDC allocations (test environments only)
            // also land balance-only.
            for usdc in &schedule.usdc_allocations {
                let entry =
                    balances.entry((usdc.recipient_address, CoinType::Usdc)).or_insert(0);
                *entry = entry
                    .checked_add(usdc.amount_microdollars)
                    .expect("genesis USDC balance overflow");
            }
        }

        // Stage 13a: validator starter USDC also lands in the
        // accumulator only. Validators submit balance-mode txs (gas
        // is debited from this USDC balance directly) so they don't
        // need a Coin object hand-out. USDC is a bridged token with
        // its own supply path, so seeding here is fine.
        // SOMA total supply is fixed (TOTAL_SUPPLY_SHANNONS) and
        // accounted for by the genesis schedule, so we do NOT seed
        // unstaked SOMA into validators here — tests that need a
        // validator with spendable SOMA should allocate it via the
        // genesis schedule.
        const VALIDATOR_GENESIS_USDC: u64 = 1_000_000_000_000; // 1M USDC microdollars
        for v in &self.validators {
            let entry =
                balances.entry((v.info.account_address, CoinType::Usdc)).or_insert(0);
            *entry = entry
                .checked_add(VALIDATOR_GENESIS_USDC)
                .expect("genesis validator USDC balance overflow");
        }

        // Set voting power and build committee
        system_state.validators_mut().set_voting_power();

        // Create system state object
        let state_object = Object::new(
            ObjectData::new_with_id(
                SYSTEM_STATE_OBJECT_ID,
                ObjectType::SystemState,
                Version::MIN,
                bcs::to_bytes(&system_state).unwrap(),
            ),
            Owner::Shared { initial_shared_version: Version::new() },
            TransactionDigest::default(),
        );
        objects.push(state_object);

        // Create the global Clock object at the reserved CLOCK_OBJECT_ID.
        // Mutated only by ConsensusCommitPrologueV1; user transactions
        // declare it as an immutable shared input so the scheduler can run
        // readers in parallel.
        objects.push(Object::new_genesis_clock());

        // Stage 14a: dual-write the genesis balance and delegation
        // state as accumulator OBJECTS in addition to the CF maps
        // returned below. Stage 14a only creates these objects; the
        // runtime continues to read from the CF rows. Stages 14b–14d
        // flip the runtime to source from the accumulator objects and
        // eventually drop the CFs entirely. By creating the objects
        // at genesis now, the migration boundary lands at the next
        // protocol version flip rather than requiring a state-import
        // pass on already-launched networks.
        for (&(owner, coin_type), &balance) in &balances {
            // Skip zero-balance rows — the BTreeMap entry exists but
            // there's no point materializing an object for it. Stage
            // 14b's dual-read path treats "absent object" the same as
            // "zero balance".
            if balance == 0 {
                continue;
            }
            let acc = crate::accumulator::BalanceAccumulator::new(owner, coin_type, balance);
            objects.push(Object::new_balance_accumulator(acc, TransactionDigest::default()));
        }

        for (&(pool_id, staker), &principal) in &delegations {
            if principal == 0 {
                continue;
            }
            let acc = crate::accumulator::DelegationAccumulator::new(
                pool_id, staker, principal, /* last_collected_period */ 0,
            );
            objects.push(Object::new_delegation_accumulator(acc, TransactionDigest::default()));
        }

        (system_state, objects, balances, delegations)
    }

    fn create_genesis_transaction(
        &self,
        objects: Vec<Object>,
        system_state: &SystemState,
    ) -> (Transaction, TransactionEffects, Vec<Object>) {
        let unsigned_tx =
            VerifiedTransaction::new_genesis_transaction(objects.clone()).into_inner();
        let genesis_digest = *unsigned_tx.digest();

        let input_objects = InputObjects::new(Vec::new());
        let mut temp_store = TemporaryStore::new(
            input_objects,
            Vec::new(),
            genesis_digest,
            0,
            system_state.fee_parameters(),
            0, // execution_version: genesis always uses v0
            self.parameters.chain,
        );

        for object in objects {
            temp_store.create_object(object);
        }

        let (inner, effects) = temp_store.into_effects(
            Vec::new(),
            &genesis_digest,
            BTreeSet::new(),
            ExecutionStatus::Success,
            0,
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
        let execution_digests =
            ExecutionDigests { transaction: *transaction.digest(), effects: effects.digest() };

        let contents = CheckpointContents::new_with_digests_and_signatures(
            vec![execution_digests],
            vec![vec![]],
        );

        let checkpoint = CheckpointSummary {
            epoch: 0,
            sequence_number: 0,
            network_total_transactions: contents.size() as u64,
            content_digest: *contents.digest(),
            previous_digest: None,
            epoch_rolling_transaction_fees: Default::default(),
            end_of_epoch_data: None,
            timestamp_ms: self.parameters.chain_start_timestamp_ms,
            checkpoint_commitments: Default::default(),
            version_specific_data: Default::default(),
        };

        (checkpoint, contents)
    }
}
