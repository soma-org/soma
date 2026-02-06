use anyhow::{Result, anyhow, bail, ensure};
use bip32::DerivationPath;
use clap::Parser;
use sdk::wallet_context::WalletContext;
use soma_keys::key_derive;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore;
use std::str::FromStr;
use types::base::SomaAddress;
use types::crypto::SignatureScheme;

use crate::response::{
    AddressesOutput, ClientCommandResponse, NewAddressOutput, RemoveAddressOutput, SwitchOutput,
};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum WalletCommand {
    /// Show the current active address
    #[clap(name = "active")]
    Active,

    /// List all addresses managed by the wallet
    #[clap(name = "list")]
    List {
        /// Sort by alias instead of address
        #[clap(long, short = 's')]
        sort_by_alias: bool,
    },

    /// Generate a new address and keypair
    #[clap(name = "new")]
    New {
        /// Key scheme: ed25519, secp256k1, or secp256r1
        key_scheme: SignatureScheme,
        /// Optional alias for the address
        alias: Option<String>,
        /// Word length: word12, word15, word18, word21, word24
        word_length: Option<String>,
        /// Custom derivation path
        derivation_path: Option<DerivationPath>,
    },

    /// Remove an address from the wallet
    #[clap(name = "remove")]
    Remove {
        /// Address or alias to remove
        alias_or_address: String,
    },

    /// Switch the active address
    #[clap(name = "switch")]
    Switch {
        /// Address or alias to make active
        address: KeyIdentity,
    },
}

/// Execute the wallet command
pub async fn execute(
    context: &mut WalletContext,
    cmd: WalletCommand,
) -> Result<ClientCommandResponse> {
    match cmd {
        WalletCommand::Active => {
            let address = context.active_address().ok();
            Ok(ClientCommandResponse::ActiveAddress(address))
        }

        WalletCommand::List { sort_by_alias } => {
            let active_address = context.active_address()?;
            let mut addresses: Vec<(String, SomaAddress)> = context
                .config
                .keystore
                .addresses_with_alias()
                .into_iter()
                .map(|(address, alias)| (alias.alias.to_string(), *address))
                .collect();

            if sort_by_alias {
                addresses.sort_by(|a, b| a.0.cmp(&b.0));
            }

            Ok(ClientCommandResponse::Addresses(AddressesOutput { active_address, addresses }))
        }

        WalletCommand::New { key_scheme, alias, derivation_path, word_length } => {
            let (address, keypair, scheme, phrase) =
                key_derive::generate_new_key(key_scheme, derivation_path, word_length)
                    .map_err(|e| anyhow!("Failed to generate new key: {}", e))?;

            context.config.keystore.import(alias.clone(), keypair).await?;

            let alias = match alias {
                Some(a) => a,
                None => context.config.keystore.get_alias(&address)?,
            };

            Ok(ClientCommandResponse::NewAddress(NewAddressOutput {
                alias,
                address,
                key_scheme: scheme,
                recovery_phrase: phrase,
            }))
        }

        WalletCommand::Remove { alias_or_address } => {
            let identity = KeyIdentity::from_str(&alias_or_address)
                .map_err(|e| anyhow!("Invalid address or alias: {}", e))?;
            let address: SomaAddress = context.config.keystore.get_by_identity(&identity)?;

            context.config.keystore.remove(address).await?;

            Ok(ClientCommandResponse::RemoveAddress(RemoveAddressOutput {
                alias_or_address,
                address,
            }))
        }

        WalletCommand::Switch { address } => {
            let resolved = context.get_identity_address(Some(address))?;
            if !context.config.keystore.addresses().contains(&resolved) {
                bail!("Address {} not managed by wallet", resolved);
            }
            context.config.active_address = Some(resolved);
            context.config.save()?;

            Ok(ClientCommandResponse::Switch(SwitchOutput { address: Some(resolved), env: None }))
        }
    }
}
