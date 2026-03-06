// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use camino::Utf8PathBuf;
use clap::Parser;
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::traits::ToFromBytes;
use protocol_config::ProtocolVersion;
use soma_keys::keypair_file::read_authority_keypair_from_file;
use types::config::SOMA_GENESIS_FILENAME;
use types::config::genesis_config::{GenesisModelConfig, SHANNONS_PER_SOMA};
use types::crypto::AuthorityKeyPair;
use types::envelope::Message as _;
use types::genesis::{Genesis, UnsignedGenesis};
use types::genesis_builder::GenesisBuilder;
use types::object::ObjectType;
use types::system_state::{SystemStateTrait, get_system_state};
use types::validator_info::GenesisValidatorInfo;

mod genesis_inspector;
use genesis_inspector::examine_genesis_checkpoint;

#[derive(Parser)]
pub struct Ceremony {
    /// Directory for ceremony state
    #[clap(long)]
    path: Option<PathBuf>,

    /// Protocol version (defaults to latest)
    #[clap(long)]
    protocol_version: Option<u64>,

    #[clap(subcommand)]
    command: CeremonyCommand,
}

impl Ceremony {
    pub fn run(self) -> Result<()> {
        run(self)
    }
}

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum CeremonyCommand {
    /// Initialize a new genesis ceremony
    Init,

    /// Validate the current ceremony state
    ValidateState,

    /// Add a consensus validator from their info file
    AddValidator {
        /// Path to the validator.info file
        #[clap(name = "validator-info-path")]
        file: PathBuf,
    },

    /// List all validators in the ceremony
    ListValidators,

    /// Add a seed model from a YAML config file
    ///
    /// The model config file should contain: owner, manifest, decryption_key,
    /// weights_commitment, architecture_version, commission_rate, and
    /// optionally initial_stake (defaults to 1 SOMA). The model_id, embedding,
    /// and commitments are auto-generated at build time.
    AddModel {
        /// Path to the model config YAML file
        #[clap(name = "model-config-path")]
        file: PathBuf,
    },

    /// List all seed models in the ceremony
    ListModels,

    /// Build the unsigned genesis checkpoint
    ///
    /// Token distribution is auto-generated from protocol config if not present.
    /// To use a custom distribution, place a `token-distribution-schedule` CSV
    /// file in the ceremony directory before running this command.
    BuildUnsignedCheckpoint,

    /// Examine the genesis checkpoint interactively
    ExamineGenesisCheckpoint,

    /// Verify and sign the genesis checkpoint
    VerifyAndSign {
        #[clap(long)]
        key_file: PathBuf,
    },

    /// Finalize the ceremony and produce the genesis blob
    Finalize,
}

pub fn run(cmd: Ceremony) -> Result<()> {
    let dir = if let Some(path) = cmd.path { path } else { std::env::current_dir()? };
    let dir = Utf8PathBuf::try_from(dir)?;

    let protocol_version =
        cmd.protocol_version.map(ProtocolVersion::new).unwrap_or(ProtocolVersion::MAX);

    match cmd.command {
        CeremonyCommand::Init => {
            let builder = GenesisBuilder::new().with_protocol_version(protocol_version);
            builder.save(&dir)?;
            println!("Genesis ceremony initialized at {}", dir);
            println!("Protocol version: {:?}", protocol_version);
        }

        CeremonyCommand::ValidateState => {
            let builder = GenesisBuilder::load(&dir)?;
            builder.validate()?;
            println!("Ceremony state is valid");
        }

        CeremonyCommand::AddValidator { file } => {
            let mut builder = GenesisBuilder::load(&dir)?;

            // Load validator info from file (same format as validator commands)
            let validator_info: GenesisValidatorInfo = load_validator_info(&file)?;

            builder = builder.add_validator(validator_info);
            builder.save(&dir)?;

            println!("Added consensus validator from {}", file.display());
        }

        CeremonyCommand::ListValidators => {
            let builder = GenesisBuilder::load(&dir)?;

            let validators = builder.validators();

            println!("Validators ({}):", validators.len());
            println!("{:-<80}", "");

            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.write_record(["account-address", "protocol-key"])?;

            for v in validators {
                writer.write_record([
                    &v.info.account_address.to_string(),
                    &Hex::encode(v.info.protocol_key.as_bytes()),
                ])?;
            }
            writer.flush()?;
        }

        CeremonyCommand::AddModel { file } => {
            let mut builder = GenesisBuilder::load(&dir)?;

            let model_config: GenesisModelConfig = load_model_config(&file)?;
            let index = builder.genesis_models().len();

            builder = builder.add_model(model_config);
            builder.save(&dir)?;

            println!(
                "Added seed model #{} from {} (model_id assigned at build time)",
                index,
                file.display()
            );
        }

        CeremonyCommand::ListModels => {
            let builder = GenesisBuilder::load(&dir)?;

            let models = builder.genesis_models();

            println!("Seed Models ({}):", models.len());
            println!("(model_id and embedding are assigned at build time)");
            println!("{:-<80}", "");

            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.write_record([
                "index",
                "owner",
                "architecture-version",
                "commission-rate",
                "initial-stake",
            ])?;

            for (i, m) in models.iter().enumerate() {
                writer.write_record([
                    &i.to_string(),
                    &m.owner.to_string(),
                    &m.architecture_version.to_string(),
                    &m.commission_rate.to_string(),
                    &m.initial_stake.to_string(),
                ])?;
            }
            writer.flush()?;
        }

        CeremonyCommand::BuildUnsignedCheckpoint => {
            let mut builder = GenesisBuilder::load(&dir)?;

            check_protocol_version(&builder, protocol_version)?;

            let UnsignedGenesis { checkpoint, .. } = builder.build_unsigned_genesis();

            println!("Successfully built unsigned checkpoint: {}", checkpoint.digest());

            builder.save(&dir)?;
        }

        CeremonyCommand::ExamineGenesisCheckpoint => {
            let builder = GenesisBuilder::load(&dir)?;

            let Some(unsigned_genesis) = builder.unsigned_genesis_checkpoint() else {
                bail!("Unable to examine genesis checkpoint; it hasn't been built yet");
            };

            examine_genesis_checkpoint(&unsigned_genesis);
        }

        CeremonyCommand::VerifyAndSign { key_file } => {
            let keypair: AuthorityKeyPair = read_authority_keypair_from_file(&key_file)?;

            let mut builder = GenesisBuilder::load(&dir)?;

            check_protocol_version(&builder, protocol_version)?;

            // Don't sign unless the unsigned checkpoint has already been created
            if builder.unsigned_genesis_checkpoint().is_none() {
                bail!("Unable to verify and sign genesis checkpoint; it hasn't been built yet");
            }

            builder = builder.add_validator_signature(&keypair);

            let UnsignedGenesis { checkpoint, .. } = builder.unsigned_genesis_checkpoint().unwrap();

            builder.save(&dir)?;

            println!(
                "Successfully verified and signed genesis checkpoint: {}",
                checkpoint.digest()
            );
        }

        CeremonyCommand::Finalize => {
            let builder = GenesisBuilder::load(&dir)?;

            check_protocol_version(&builder, protocol_version)?;

            let genesis = builder.build();

            let genesis_path = dir.join(SOMA_GENESIS_FILENAME);
            genesis.save(&genesis_path)?;

            println!("Successfully built {}", SOMA_GENESIS_FILENAME);
            println!("{} blake2b-256: {}", SOMA_GENESIS_FILENAME, Hex::encode(genesis.hash()));
        }
    }

    Ok(())
}

fn check_protocol_version(
    builder: &GenesisBuilder,
    protocol_version: ProtocolVersion,
) -> Result<()> {
    if builder.protocol_version() != protocol_version {
        bail!(
            "Serialized protocol version does not match local --protocol-version argument. ({:?} vs {:?})",
            builder.protocol_version(),
            protocol_version
        );
    }
    Ok(())
}

fn load_validator_info(path: &PathBuf) -> Result<GenesisValidatorInfo> {
    let bytes = std::fs::read(path)?;
    serde_yaml::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse validator info from {:?}: {}", path, e))
}

fn load_model_config(path: &PathBuf) -> Result<GenesisModelConfig> {
    let bytes = std::fs::read(path)?;
    serde_yaml::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse model config from {:?}: {}", path, e))
}

/// Inspect a genesis.blob file and print diagnostic information about models, targets,
/// emission pool, and key parameters.
pub fn inspect_genesis_blob(path: &Path) -> Result<()> {
    let genesis = Genesis::load(path)?;
    let objects = genesis.objects();
    let system_state = get_system_state(&objects)?;

    println!("=== Genesis Inspection: {} ===", path.display());
    println!();

    // --- Summary ---
    println!("Summary:");
    println!("  Total Objects:  {}", objects.len());
    println!("  Epoch:          {}", system_state.epoch());
    println!();

    // --- Models ---
    let registry = system_state.model_registry();
    println!("Models (ModelRegistry):");
    println!(
        "  Active: {}  Pending: {}  Inactive: {}",
        registry.active_models().count(),
        registry.pending_models().count(),
        registry.inactive_models().count(),
    );
    println!(
        "  Total Model Stake: {} shannons ({} SOMA)",
        registry.total_model_stake,
        registry.total_model_stake / SHANNONS_PER_SOMA
    );
    println!();

    if registry.has_active_models() {
        println!("  Active Models:");
        for (i, (id, model)) in registry.active_models().enumerate() {
            let stake = model.staking_pool.soma_balance;
            println!(
                "    [{}] ID: {}  Owner: {}  Arch: {}  Commission: {} bps  Stake: {} SOMA",
                i,
                id,
                model.owner,
                model.architecture_version,
                model.commission_rate,
                stake / SHANNONS_PER_SOMA,
            );
        }
        println!();
    }

    // --- Target Parameters ---
    let params = system_state.parameters();
    println!("Target Parameters:");
    println!("  target_initial_targets_per_epoch: {}", params.target_initial_targets_per_epoch);
    println!("  target_models_per_target:         {}", params.target_models_per_target);
    println!("  target_embedding_dim:             {}", params.target_embedding_dim);
    println!("  target_reward_allocation_bps:     {}", params.target_reward_allocation_bps);
    println!(
        "  target_initial_distance_threshold: {}",
        params.target_initial_distance_threshold.as_scalar()
    );
    println!();

    // --- Target State ---
    let target_state = system_state.target_state();
    println!("Target State:");
    println!("  distance_threshold:             {}", target_state.distance_threshold.as_scalar());
    println!("  reward_per_target:              {} shannons", target_state.reward_per_target);
    println!("  targets_generated_this_epoch:   {}", target_state.targets_generated_this_epoch);
    println!("  hits_this_epoch:                {}", target_state.hits_this_epoch);
    println!("  hits_ema:                       {}", target_state.hits_ema);
    println!();

    // --- Emission Pool ---
    let pool = system_state.emission_pool();
    println!("Emission Pool:");
    println!(
        "  balance:          {} shannons ({} SOMA)",
        pool.balance,
        pool.balance / SHANNONS_PER_SOMA
    );
    println!(
        "  emission_per_epoch: {} shannons ({} SOMA)",
        pool.emission_per_epoch,
        pool.emission_per_epoch / SHANNONS_PER_SOMA
    );
    if pool.emission_per_epoch > 0 {
        println!("  epochs_remaining: ~{}", pool.balance / pool.emission_per_epoch);
    }
    println!();

    // --- Target Objects ---
    let target_objects: Vec<_> =
        objects.iter().filter(|o| *o.type_() == ObjectType::Target).collect();
    println!("Target Objects: {}", target_objects.len());
    for (i, obj) in target_objects.iter().enumerate() {
        if let Some(target) = obj.as_target() {
            println!(
                "  [{}] ID: {}  Status: {:?}  Epoch: {}  Models: {}  Reward: {} shannons",
                i,
                obj.id(),
                target.status,
                target.generation_epoch,
                target.model_ids.len(),
                target.reward_pool,
            );
        } else {
            println!("  [{}] ID: {}  (failed to deserialize)", i, obj.id());
        }
    }
    println!();

    // --- Diagnostics ---
    println!("Diagnostics:");
    let mut ok = true;

    if !registry.has_active_models() {
        println!("  [WARN] No active models — targets cannot be generated without active models");
        ok = false;
    } else {
        println!("  [OK]   {} active model(s) found", registry.active_model_count());
    }

    if params.target_initial_targets_per_epoch == 0 {
        println!("  [WARN] target_initial_targets_per_epoch = 0 — no targets will be generated");
        ok = false;
    } else {
        println!(
            "  [OK]   target_initial_targets_per_epoch = {}",
            params.target_initial_targets_per_epoch
        );
    }

    if pool.balance == 0 {
        println!("  [WARN] Emission pool balance is 0 — cannot fund targets");
        ok = false;
    } else if target_state.reward_per_target > pool.balance {
        println!(
            "  [WARN] reward_per_target ({}) > emission pool balance ({}) — insufficient funds",
            target_state.reward_per_target, pool.balance
        );
        ok = false;
    } else {
        println!("  [OK]   Emission pool has sufficient balance");
    }

    if target_objects.is_empty() && registry.has_active_models() {
        println!(
            "  [FAIL] No target objects despite active models — targets were not generated at genesis"
        );
        ok = false;
    } else if !target_objects.is_empty() {
        println!("  [OK]   {} target object(s) created", target_objects.len());
    }

    if ok {
        println!();
        println!(
            "Genesis looks correct: {} models active, {} targets created",
            registry.active_model_count(),
            target_objects.len()
        );
    }

    Ok(())
}
