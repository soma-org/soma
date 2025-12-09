use std::path::PathBuf;

use clap::Parser;
use soma_keys::keystore::{FileBasedKeystore, Keystore};
use types::config::{soma_config_dir, SOMA_KEYSTORE_FILENAME};

use crate::{inference::InferenceCommand, keytool::KeyToolCommand};

#[allow(clippy::large_enum_variant)]
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaCommand {
    #[clap(name = "keytool")]
    KeyTool {
        #[clap(long)]
        keystore_path: Option<PathBuf>,
        /// Subcommands.
        #[clap(subcommand)]
        cmd: KeyToolCommand,
    },
    #[clap(name = "inference")]
    Inference {
        /// Subcommands.
        #[clap(subcommand)]
        cmd: InferenceCommand,
    },
}

impl SomaCommand {
    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            SomaCommand::KeyTool { keystore_path, cmd } => {
                let keystore_path =
                    keystore_path.unwrap_or(soma_config_dir()?.join(SOMA_KEYSTORE_FILENAME));
                let mut keystore =
                    Keystore::from(FileBasedKeystore::load_or_create(&keystore_path)?);
                cmd.execute(&mut keystore).await?.print(true);
                Ok(())
            }
            SomaCommand::Inference { cmd } => {
                cmd.execute().await?.print(true);
                Ok(())
            }
        }
    }
}
