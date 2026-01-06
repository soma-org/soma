use anyhow::{bail, Result};
use camino::Utf8PathBuf;
use clap::Parser;
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::traits::ToFromBytes;
use protocol_config::ProtocolVersion;
use soma_keys::keypair_file::read_authority_keypair_from_file;
use std::path::PathBuf;
use types::{
    config::SOMA_GENESIS_FILENAME, crypto::AuthorityKeyPair, envelope::Message as _,
    genesis::UnsignedGenesis, genesis_builder::GenesisBuilder,
    validator_info::GenesisValidatorInfo,
};

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

    /// Add a networking-only validator from their info file
    AddNetworkingValidator {
        /// Path to the validator.info file  
        #[clap(name = "validator-info-path")]
        file: PathBuf,
    },

    /// Add an encoder from their info file
    AddEncoder {
        /// Path to the encoder.info file
        #[clap(name = "encoder-info-path")]
        file: PathBuf,
        /// URL to the probe data (will be downloaded to compute checksum)
        #[clap(long)]
        probe_url: String,
    },

    /// List all validators in the ceremony
    ListValidators,

    /// List all encoders in the ceremony
    ListEncoders,

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
    let dir = if let Some(path) = cmd.path {
        path
    } else {
        std::env::current_dir()?
    };
    let dir = Utf8PathBuf::try_from(dir)?;

    let protocol_version = cmd
        .protocol_version
        .map(ProtocolVersion::new)
        .unwrap_or(ProtocolVersion::MAX);

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

        CeremonyCommand::AddNetworkingValidator { file } => {
            let mut builder = GenesisBuilder::load(&dir)?;

            let validator_info: GenesisValidatorInfo = load_validator_info(&file)?;

            builder = builder.add_networking_validator(validator_info);
            builder.save(&dir)?;

            println!("Added networking validator from {}", file.display());
        }

        CeremonyCommand::AddEncoder { file, probe_url } => {
            let mut builder = GenesisBuilder::load(&dir)?;

            let encoder_info = load_encoder_info(&file, &probe_url)?;

            builder = builder.add_encoder(encoder_info);
            builder.save(&dir)?;

            println!("Added encoder from {}", file.display());
        }

        CeremonyCommand::ListValidators => {
            let builder = GenesisBuilder::load(&dir)?;

            let validators = builder.validators();
            let networking_validators = builder.networking_validators();

            println!("Consensus Validators ({}):", validators.len());
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

            if !networking_validators.is_empty() {
                println!("\nNetworking Validators ({}):", networking_validators.len());
                println!("{:-<80}", "");

                let mut writer = csv::Writer::from_writer(std::io::stdout());
                writer.write_record(["account-address", "protocol-key"])?;

                for v in networking_validators {
                    writer.write_record([
                        &v.info.account_address.to_string(),
                        &Hex::encode(v.info.protocol_key.as_bytes()),
                    ])?;
                }
                writer.flush()?;
            }
        }

        CeremonyCommand::ListEncoders => {
            let builder = GenesisBuilder::load(&dir)?;

            let encoders = builder.encoders();

            println!("Encoders ({}):", encoders.len());
            println!("{:-<80}", "");

            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.write_record(["account-address", "encoder-pubkey"])?;

            for e in encoders {
                writer.write_record([
                    &e.info.account_address.to_string(),
                    &Hex::encode(e.info.encoder_pubkey.to_bytes()),
                ])?;
            }
            writer.flush()?;
        }

        CeremonyCommand::BuildUnsignedCheckpoint => {
            let mut builder = GenesisBuilder::load(&dir)?;

            check_protocol_version(&builder, protocol_version)?;

            let UnsignedGenesis { checkpoint, .. } = builder.build_unsigned_genesis();

            println!(
                "Successfully built unsigned checkpoint: {}",
                checkpoint.digest()
            );

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
            println!(
                "{} blake2b-256: {}",
                SOMA_GENESIS_FILENAME,
                Hex::encode(genesis.hash())
            );
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

fn load_encoder_info(
    path: &PathBuf,
    probe_url: &str,
) -> Result<types::encoder_info::GenesisEncoderInfo> {
    use types::encoder_info::{EncoderInfo, GenesisEncoderInfo};

    let bytes = std::fs::read(path)?;
    let info: EncoderInfo = serde_yaml::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse encoder info from {:?}: {}", path, e))?;

    // Download probe data and create metadata
    let probe = download_and_create_probe(probe_url)?;

    Ok(GenesisEncoderInfo { info, probe })
}

fn download_and_create_probe(url_str: &str) -> Result<types::metadata::DownloadMetadata> {
    use fastcrypto::hash::HashFunction as _;
    use types::{
        checksum::Checksum,
        crypto::DefaultHash,
        metadata::{
            DefaultDownloadMetadata, DefaultDownloadMetadataV1, DownloadMetadata, Metadata,
            MetadataV1,
        },
    };
    use url::Url;

    let parsed_url = Url::parse(url_str).map_err(|e| anyhow::anyhow!("Invalid URL: {}", e))?;

    println!("Downloading probe data from {}...", url_str);

    let response = reqwest::blocking::get(url_str)
        .map_err(|e| anyhow::anyhow!("Failed to download probe data: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
    }

    let data = response
        .bytes()
        .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?;

    let size = data.len();
    println!("Downloaded {} bytes", size);

    // Compute checksum
    let mut hasher = DefaultHash::default();
    hasher.update(&data);
    let hash: [u8; 32] = hasher.finalize().into();
    let checksum = Checksum::new_from_hash(hash);

    let metadata = Metadata::V1(MetadataV1::new(checksum, size));

    Ok(DownloadMetadata::Default(DefaultDownloadMetadata::V1(
        DefaultDownloadMetadataV1::new(parsed_url, metadata),
    )))
}
