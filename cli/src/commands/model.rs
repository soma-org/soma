// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma model {list,show}` — dump the protocol-config `ModelRegistry`.
//! Read-only: there's no on-chain side to a model itself; the registry
//! is per protocol version.

use anyhow::Result;
use clap::Parser;
use protocol_config::model_registry::ModelRegistry;
use protocol_config::ProtocolVersion;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum ModelCommand {
    /// List every model in the registry at `--protocol-version` (defaults
    /// to MAX). Set `--all` to include deprecated entries.
    List {
        /// Protocol version (defaults to MAX).
        #[clap(long)]
        protocol_version: Option<u64>,
        /// Include deprecated (active=false) entries.
        #[clap(long, default_value_t = false)]
        all: bool,
        /// Print only model_ids, one per line. Useful for shell loops.
        #[clap(long, default_value_t = false)]
        ids_only: bool,
    },
    /// Print one entry as JSON. Errors if the model isn't in the
    /// registry at the requested version.
    Show {
        #[clap(long)]
        model_id: String,
        #[clap(long)]
        protocol_version: Option<u64>,
    },
}

impl ModelCommand {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::List { protocol_version, all, ids_only } => {
                let version = match protocol_version {
                    Some(v) => ProtocolVersion::new(v),
                    None => ProtocolVersion::MAX,
                };
                let registry = ModelRegistry::for_version(version);
                if ids_only {
                    let entries: Vec<_> = if all {
                        registry.entries.iter().collect()
                    } else {
                        registry.active().collect()
                    };
                    for e in entries {
                        println!("{}", e.model_id);
                    }
                } else {
                    let entries: Vec<_> = if all {
                        registry.entries.iter().collect()
                    } else {
                        registry.active().collect()
                    };
                    println!("{}", serde_json::to_string_pretty(&entries)?);
                }
                Ok(())
            }
            Self::Show { model_id, protocol_version } => {
                let version = match protocol_version {
                    Some(v) => ProtocolVersion::new(v),
                    None => ProtocolVersion::MAX,
                };
                let registry = ModelRegistry::for_version(version);
                let entry = registry
                    .get_any(&model_id)
                    .ok_or_else(|| anyhow::anyhow!("unknown model_id: {model_id}"))?;
                println!("{}", serde_json::to_string_pretty(entry)?);
                Ok(())
            }
        }
    }
}
