use std::path::PathBuf;

use clap::Parser;
use soma_config::{soma_dir, SOMA_KEYSTORE_FILENAME};
use soma_keys::keystore::{FileBasedKeystore, Keystore};

use crate::{
    error::{CliError, CliResult},
    inference::InferenceCommand,
    keytool::KeyToolCommand,
};

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
    pub async fn execute(self) -> CliResult<()> {
        match self {
            SomaCommand::KeyTool { keystore_path, cmd } => {
                let keystore_path = keystore_path.unwrap_or(
                    soma_dir()
                        .map_err(CliError::SomaConfig)?
                        .join(SOMA_KEYSTORE_FILENAME),
                );
                let mut keystore = Keystore::from(
                    FileBasedKeystore::new(&keystore_path).map_err(CliError::SomaKey)?,
                );
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
