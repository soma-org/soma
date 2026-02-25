// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail, ensure};
use bip32::DerivationPath;
use clap::{Parser, ValueEnum};
use sdk::wallet_context::WalletContext;
use soma_keys::key_derive;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore;
use std::str::FromStr;
use types::base::SomaAddress;
use types::crypto::SignatureScheme;

use crate::response::{
    ActiveAddressOutput, AddressesOutput, ClientCommandResponse, NewAddressOutput,
    RemoveAddressOutput, SwitchOutput,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum WordLength {
    #[value(name = "word12")]
    Word12,
    #[value(name = "word15")]
    Word15,
    #[value(name = "word18")]
    Word18,
    #[value(name = "word21")]
    Word21,
    #[value(name = "word24")]
    Word24,
}

impl WordLength {
    fn to_string_option(self) -> Option<String> {
        Some(
            match self {
                WordLength::Word12 => "word12",
                WordLength::Word15 => "word15",
                WordLength::Word18 => "word18",
                WordLength::Word21 => "word21",
                WordLength::Word24 => "word24",
            }
            .to_string(),
        )
    }
}

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
        /// Alias for the new address (e.g., "my-wallet")
        #[clap(long)]
        alias: Option<String>,
        /// Key scheme
        #[clap(long, default_value = "ed25519")]
        key_scheme: SignatureScheme,
        /// Word length for recovery phrase
        #[clap(long, value_enum, default_value = "word12")]
        word_length: WordLength,
        /// Custom derivation path
        #[clap(long)]
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
            Ok(ClientCommandResponse::ActiveAddress(address.map(|addr| {
                let alias = context.config.keystore.get_alias(&addr).ok();
                ActiveAddressOutput { address: addr, alias }
            })))
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
            let (address, keypair, scheme, phrase) = key_derive::generate_new_key(
                key_scheme,
                derivation_path,
                word_length.to_string_option(),
            )
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
