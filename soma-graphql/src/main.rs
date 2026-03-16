// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::Parser;
use indexer_pg_db::DbArgs;
use tokio::net::TcpListener;
use tracing::info;
use url::Url;

use soma_graphql::config::GraphQlConfig;
use soma_graphql::db::PgReader;
use soma_graphql::subscriptions::{SubscriptionChannels, spawn_pg_listener};
use soma_graphql::{AppState, KvLoader, build_router, build_schema};

use indexer_kvstore::{BigTableClient, BigTableKvLoader};

#[derive(Parser, Debug)]
#[command(name = "soma-graphql", about = "Soma GraphQL API server")]
struct Args {
    /// Postgres database URL (e.g. postgres://user:pass@host/db).
    #[arg(long)]
    database_url: String,

    /// Address to bind the server to.
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen_address: String,

    /// Path to a TOML config file (optional).
    #[arg(long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let config: GraphQlConfig = match &args.config {
        Some(path) => {
            let contents = std::fs::read_to_string(path)?;
            toml::from_str(&contents)?
        }
        None => GraphQlConfig::default(),
    };

    let listen_address = args.listen_address.clone();

    let db_args = DbArgs {
        db_connection_pool_size: config.db_pool_size,
        db_connection_timeout_ms: config.db_connection_timeout_ms,
        db_statement_timeout_ms: config.db_statement_timeout_ms,
    };

    let database_url = Url::parse(&args.database_url)?;
    let pg = Arc::new(PgReader::new(database_url, db_args).await?);

    // When BigTable is configured, create a KvLoader so resolvers read BCS from BigTable.
    let kv: Option<Arc<dyn KvLoader>> = match &config.bigtable_instance {
        Some(instance_id) => {
            info!("Connecting to BigTable instance: {instance_id}");
            let client = BigTableClient::new_remote_with_credentials(
                instance_id.clone(),
                config.bigtable_project.clone(),
                true, // read-only
                None,
                None,
                "soma-graphql".to_string(),
                None,
                None,
                None,
                config.bigtable_credentials.clone(),
            )
            .await?;
            Some(Arc::new(BigTableKvLoader::new(client)))
        }
        None => None,
    };

    // Set up subscription channels and Postgres LISTEN/NOTIFY listener
    let sub_channels = SubscriptionChannels::new(1024);
    spawn_pg_listener(args.database_url.clone(), sub_channels.clone());

    let schema = build_schema(pg, config, kv, sub_channels);
    let state = AppState { schema };
    let router = build_router(state);

    let listener = TcpListener::bind(&listen_address).await?;
    info!("Soma GraphQL server listening on {}", listen_address);
    info!("WebSocket subscriptions available at ws://{}/graphql/ws", listen_address);
    axum::serve(listener, router).await?;

    Ok(())
}
