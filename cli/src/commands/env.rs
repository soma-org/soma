use anyhow::{Result, bail, ensure};
use clap::Parser;
use sdk::client_config::SomaEnv;
use sdk::wallet_context::WalletContext;

use crate::response::{
    ChainInfoOutput, ClientCommandResponse, EnvsOutput, NewEnvOutput, SwitchOutput,
};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum EnvCommand {
    /// Show the current active environment
    #[clap(name = "active")]
    Active,

    /// List all configured environments
    #[clap(name = "list")]
    List,

    /// Add a new environment
    #[clap(name = "new")]
    New {
        /// Alias for the environment
        #[clap(long)]
        alias: String,
        /// RPC URL
        #[clap(long)]
        rpc: String,
        /// Basic auth credentials (format: username:password)
        #[clap(long)]
        basic_auth: Option<String>,
    },

    /// Switch the active environment
    #[clap(name = "switch")]
    Switch {
        /// Environment alias to switch to
        alias: String,
    },

    /// Get chain identifier from current environment
    #[clap(name = "chain-id")]
    ChainId,
}

/// Execute the env command
pub async fn execute(
    context: &mut WalletContext,
    cmd: EnvCommand,
) -> Result<ClientCommandResponse> {
    match cmd {
        EnvCommand::Active => {
            let env = context.get_active_env().ok().map(|e| e.alias.clone());
            Ok(ClientCommandResponse::ActiveEnv(env))
        }

        EnvCommand::List => {
            let envs = context.config.envs.clone();
            let active = context.get_active_env().ok().map(|e| e.alias.clone());
            Ok(ClientCommandResponse::Envs(EnvsOutput { envs, active }))
        }

        EnvCommand::New { alias, rpc, basic_auth } => {
            if context.config.envs.iter().any(|e| e.alias == alias) {
                bail!("Environment '{}' already exists", alias);
            }

            let env = SomaEnv { alias: alias.clone(), rpc, basic_auth, chain_id: None };

            // Verify connection
            env.create_rpc_client(None).await?;

            context.config.envs.push(env.clone());
            context.config.save()?;

            // Cache chain ID
            let client = context.get_client().await?;
            let chain_id = context.cache_chain_id(&client).await?;

            Ok(ClientCommandResponse::NewEnv(NewEnvOutput { alias, chain_id }))
        }

        EnvCommand::Switch { alias } => {
            ensure!(
                context.config.get_env(&Some(alias.clone())).is_some(),
                "Environment '{}' not found. Use 'soma env new' to add it.",
                alias
            );
            context.config.active_env = Some(alias.clone());
            context.config.save()?;

            Ok(ClientCommandResponse::Switch(SwitchOutput { address: None, env: Some(alias) }))
        }

        EnvCommand::ChainId => {
            let client = context.get_client().await?;
            let chain_id = context.cache_chain_id(&client).await?;
            let server_version = client.get_server_version().await.ok();

            Ok(ClientCommandResponse::ChainInfo(ChainInfoOutput { chain_id, server_version }))
        }
    }
}
