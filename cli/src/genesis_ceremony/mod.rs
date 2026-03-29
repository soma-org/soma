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
use types::config::genesis_config::SHANNONS_PER_SOMA;
use types::crypto::AuthorityKeyPair;
use types::envelope::Message as _;
use types::genesis::{Genesis, UnsignedGenesis};
use types::genesis_builder::GenesisBuilder;
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

/// Inspect a genesis.blob file and print diagnostic information about
/// emission pool and key parameters.
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

    // --- Emission Pool ---
    let pool = system_state.emission_pool();
    println!("Emission Pool:");
    println!(
        "  balance:          {} shannons ({} SOMA)",
        pool.balance,
        pool.balance / SHANNONS_PER_SOMA
    );
    println!(
        "  current_distribution: {} shannons ({} SOMA)",
        pool.current_distribution_amount,
        pool.current_distribution_amount / SHANNONS_PER_SOMA
    );
    println!("  period_length:    {} epochs", pool.period_length);
    println!("  decrease_rate:    {} bps", pool.decrease_rate);
    println!("  distribution_counter: {}", pool.distribution_counter);
    if pool.current_distribution_amount > 0 {
        println!("  epochs_remaining: ~{}", pool.balance / pool.current_distribution_amount);
    }
    println!();

    // --- Diagnostics ---
    println!("Diagnostics:");

    if pool.balance == 0 {
        println!("  [WARN] Emission pool balance is 0");
    } else {
        println!("  [OK]   Emission pool has sufficient balance");
    }

    Ok(())
}
