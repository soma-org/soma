use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, bail};
use camino::Utf8Path;
use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::traits::{KeyPair as _, ToFromBytes};
use protocol_config::ProtocolVersion;
use tracing::trace;

use crate::base::ExecutionDigests;
use crate::config::genesis_config::{
    GenesisCeremonyParameters, GenesisModelConfig, TOTAL_SUPPLY_SHANNONS,
    TokenDistributionSchedule, ValidatorGenesisConfig,
};
use crate::crypto::AuthoritySignInfoTrait as _;
use crate::envelope::Message as _;
use crate::intent::{IntentMessage, IntentScope};
use crate::object::{ObjectData, ObjectType, Version};
use crate::system_state::epoch_start::EpochStartSystemStateTrait as _;
use crate::system_state::staking::StakingPool;
use crate::system_state::validator::{Validator, ValidatorMetadata};
use crate::system_state::{FeeParameters, get_system_state};
use crate::transaction::InputObjects;
use crate::tx_fee::TransactionFee;
use crate::validator_info::GenesisValidatorInfo;
use crate::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    checkpoints::{CertifiedCheckpointSummary, CheckpointContents, CheckpointSummary},
    crypto::{AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo},
    digests::TransactionDigest,
    effects::{ExecutionStatus, TransactionEffects},
    genesis::{Genesis, UnsignedGenesis},
    intent::Intent,
    object::{Object, ObjectID, Owner},
    system_state::{SystemState, SystemStateTrait},
    temporary_store::TemporaryStore,
    transaction::{Transaction, VerifiedTransaction},
};
use std::collections::BTreeSet;

const GENESIS_BUILDER_COMMITTEE_DIR: &str = "committee";
const GENESIS_BUILDER_MODELS_DIR: &str = "models";
const GENESIS_BUILDER_PARAMETERS_FILE: &str = "parameters";
const GENESIS_BUILDER_TOKEN_DISTRIBUTION_SCHEDULE_FILE: &str = "token-distribution-schedule";
const GENESIS_BUILDER_SIGNATURE_DIR: &str = "signatures";
const GENESIS_BUILDER_UNSIGNED_GENESIS_FILE: &str = "unsigned-genesis";

pub struct GenesisBuilder {
    parameters: GenesisCeremonyParameters,
    token_distribution_schedule: Option<TokenDistributionSchedule>,
    validators: Vec<GenesisValidatorInfo>,
    genesis_models: Vec<GenesisModelConfig>,
    signatures: BTreeMap<AuthorityPublicKeyBytes, AuthoritySignInfo>,
    built_genesis: Option<UnsignedGenesis>,
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
            genesis_models: Vec::new(),
            signatures: BTreeMap::new(),
            built_genesis: None,
        }
    }

    pub fn with_parameters(mut self, parameters: GenesisCeremonyParameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_protocol_version(mut self, v: ProtocolVersion) -> Self {
        self.parameters.protocol_version = v;
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

    /// Add seed models to be created at genesis (skip commit-reveal).
    pub fn with_genesis_models(mut self, models: Vec<GenesisModelConfig>) -> Self {
        self.genesis_models = models;
        self.built_genesis = None;
        self
    }

    /// Add a single seed model to be created at genesis (ceremony workflow).
    pub fn add_model(mut self, model: GenesisModelConfig) -> Self {
        self.genesis_models.push(model);
        self.built_genesis = None;
        self
    }

    pub fn validators(&self) -> &[GenesisValidatorInfo] {
        &self.validators
    }

    pub fn genesis_models(&self) -> &[GenesisModelConfig] {
        &self.genesis_models
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

        let (system_state, objects) = self.create_genesis_state();
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
                format!(
                    "metadata for validator {} is invalid",
                    validator.info.account_address
                )
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
            system_state.validators.validators.len(),
            "Validator count mismatch"
        );

        let committee = system_state.into_epoch_start_state().get_committee();
        for signature in self.signatures.values() {
            let validator_exists = self
                .validators
                .iter()
                .any(|v| v.info.protocol_key == signature.authority);

            if !validator_exists {
                panic!(
                    "found signature for unknown validator: {:?}",
                    signature.authority
                );
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

        // Load validators
        let mut validators = Vec::new();
        let committee_dir = path.join(GENESIS_BUILDER_COMMITTEE_DIR);
        if committee_dir.exists() {
            for entry in committee_dir.read_dir_utf8()? {
                let entry = entry?;
                if entry.file_name().starts_with('.') {
                    continue;
                }
                let validator_path = entry.path();
                let validator_bytes = fs::read(validator_path)?;
                let validator_info: GenesisValidatorInfo = serde_yaml::from_slice(&validator_bytes)
                    .with_context(|| {
                        format!("unable to load validator info for {}", validator_path)
                    })?;
                validators.push(validator_info);
            }
        }

        // Load genesis models
        let mut genesis_models = Vec::new();
        let models_dir = path.join(GENESIS_BUILDER_MODELS_DIR);
        if models_dir.exists() {
            for entry in models_dir.read_dir_utf8()? {
                let entry = entry?;
                if entry.file_name().starts_with('.') {
                    continue;
                }
                let model_path = entry.path();
                let model_bytes = fs::read(model_path)?;
                let model_config: GenesisModelConfig = serde_yaml::from_slice(&model_bytes)
                    .with_context(|| {
                        format!("unable to load genesis model config for {}", model_path)
                    })?;
                genesis_models.push(model_config);
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
            genesis_models,
            signatures,
            built_genesis: None,
        };

        // Load unsigned genesis if present and verify
        let unsigned_genesis_file = path.join(GENESIS_BUILDER_UNSIGNED_GENESIS_FILE);
        if unsigned_genesis_file.exists() {
            let unsigned_genesis_bytes = fs::read(&unsigned_genesis_file)?;
            let loaded_genesis: UnsignedGenesis = bcs::from_bytes(&unsigned_genesis_bytes)?;

            assert!(
                builder.token_distribution_schedule.is_some(),
                "If a built genesis is present, there must also be a token-distribution-schedule"
            );

            let built = builder.build_unsigned_genesis();
            assert_eq!(
                built, loaded_genesis,
                "loaded genesis does not match built genesis"
            );
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
                committee_dir.join(&validator.info.account_address.to_string()),
                validator_bytes,
            )?;
        }

        // Write genesis models
        let models_dir = path.join(GENESIS_BUILDER_MODELS_DIR);
        fs::create_dir_all(&models_dir)?;
        for model in &self.genesis_models {
            let model_bytes = serde_yaml::to_string(model)?;
            fs::write(models_dir.join(model.model_id.to_string()), model_bytes)?;
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
            fs::write(
                path.join(GENESIS_BUILDER_UNSIGNED_GENESIS_FILE),
                genesis_bytes,
            )?;
        }

        Ok(())
    }

    fn create_genesis_state(&self) -> (SystemState, Vec<Object>) {
        let mut objects = Vec::new();

        let protocol_config = protocol_config::ProtocolConfig::get_for_version(
            self.parameters.protocol_version,
            protocol_config::Chain::Mainnet, // TODO: detect what chain to use here
        );

        // Convert GenesisValidatorInfo to on-chain Validator
        let validators: Vec<Validator> = self
            .validators
            .iter()
            .map(|v| {
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
                        next_epoch_protocol_pubkey: None,
                        next_epoch_network_pubkey: None,
                        next_epoch_net_address: None,
                        next_epoch_p2p_address: None,
                        next_epoch_primary_address: None,
                        next_epoch_worker_pubkey: None,
                    },
                    voting_power: 0, // Will be set by set_voting_power()
                    staking_pool: StakingPool::new(ObjectID::random()),
                    commission_rate: v.info.commission_rate,
                    next_epoch_stake: 0,
                    next_epoch_commission_rate: v.info.commission_rate,
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
            self.parameters.emission_per_epoch,
            Some(self.parameters.epoch_duration_ms),
        );

        // Add genesis models (skip commit-reveal, created directly as active)
        for model_config in &self.genesis_models {
            system_state.add_model_at_genesis(
                model_config.model_id,
                model_config.owner,
                model_config.weights_manifest.clone(),
                model_config.weights_url_commitment,
                model_config.weights_commitment,
                model_config.architecture_version,
                model_config.commission_rate,
            );
        }

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
                        .expect("Failed to stake with validator at genesis");

                    let staked_object = Object::new_staked_soma_object(
                        ObjectID::random(),
                        staked_soma,
                        Owner::AddressOwner(allocation.recipient_address),
                        TransactionDigest::default(),
                    );
                    objects.push(staked_object);
                } else if let Some(model_id) = allocation.staked_with_model {
                    let staked_soma = system_state
                        .request_add_stake_to_model_at_genesis(
                            &model_id,
                            allocation.amount_shannons,
                        )
                        .expect("Failed to stake with model at genesis");

                    let staked_object = Object::new_staked_soma_object(
                        ObjectID::random(),
                        staked_soma,
                        Owner::AddressOwner(allocation.recipient_address),
                        TransactionDigest::default(),
                    );
                    objects.push(staked_object);
                } else {
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

        // Set voting power and build committee
        system_state.validators.set_voting_power();

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
        let execution_digests = ExecutionDigests {
            transaction: *transaction.digest(),
            effects: effects.digest(),
        };

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
        };

        (checkpoint, contents)
    }
}
