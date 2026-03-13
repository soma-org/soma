// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer end-to-end test infrastructure.
//!
//! Ported from Sui's `sui-indexer-alt-e2e-tests/src/lib.rs`.
//!
//! Provides [`OffchainCluster`] — an orchestrator that starts an ephemeral Postgres,
//! the full indexer pipeline (18 handlers), and a GraphQL server, then exposes
//! helpers for polling watermarks and querying indexed data.

use std::net::{SocketAddr, TcpListener};
use std::ops::DerefMut;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use indexer_alt::setup_indexer;
use indexer_framework::ingestion::ingestion_client::IngestionClientArgs;
use indexer_framework::ingestion::{ClientArgs, IngestionConfig};
use indexer_framework::{Indexer, IndexerArgs};
use indexer_pg_db::temp::TempDb;
use indexer_pg_db::watermarks;
use indexer_pg_db::{Db, DbArgs};
use rpc::utils::checkpoint_blob;
use soma_graphql::config::GraphQlConfig;
use soma_graphql::db::PgReader;
use soma_graphql::{build_router, build_schema, AppState};
use tokio::time::interval;
use types::full_checkpoint_content::Checkpoint;

/// Manages the off-chain indexer stack for e2e testing.
///
/// Starts:
/// - Ephemeral Postgres (via [`TempDb`])
/// - Full indexer with all 18 pipelines reading from a local checkpoint directory
/// - GraphQL server backed by the same database
///
/// Ported from Sui's `OffchainCluster` in `sui-indexer-alt-e2e-tests`.
pub struct OffchainCluster {
    db: Db,
    graphql_listen_address: SocketAddr,
    pipelines: Vec<&'static str>,
    _service: tokio::task::JoinHandle<()>,
    // _database must be dropped last — stopping Postgres after services stop.
    _database: TempDb,
}

impl OffchainCluster {
    /// Start the full off-chain stack.
    ///
    /// `checkpoint_dir` is the directory where `.binpb.zst` checkpoint files are written
    /// (either by a TestCluster fullnode or manually via [`write_checkpoint_file`]).
    ///
    /// `indexer_args` configures the indexer (e.g. `last_checkpoint` to bound ingestion).
    pub async fn new(
        checkpoint_dir: &Path,
        indexer_args: IndexerArgs,
        registry: &prometheus::Registry,
    ) -> Result<Self> {
        // 1. Ephemeral Postgres
        let database = TempDb::new();
        let db = Db::for_write(
            database.url().clone(),
            DbArgs {
                db_connection_pool_size: 10,
                db_connection_timeout_ms: 30_000,
                db_statement_timeout_ms: None,
            },
        )
        .await
        .context("Failed to create DB pool")?;

        db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS))
            .await
            .context("Failed to run migrations")?;

        // 2. Indexer with local ingestion
        let client_args = ClientArgs {
            ingestion: IngestionClientArgs {
                local_ingestion_path: Some(checkpoint_dir.to_path_buf()),
                ..Default::default()
            },
        };

        let mut indexer = Indexer::new(
            db.clone(),
            indexer_args,
            client_args,
            IngestionConfig::default(),
            None,
            registry,
        )
        .await
        .context("Failed to create indexer")?;

        setup_indexer(&mut indexer, indexer_alt::PruningConfig::default())
            .await
            .context("Failed to register pipelines")?;

        let pipelines: Vec<&'static str> = indexer.pipelines().collect();

        let mut indexer_service = indexer.run().await.context("Failed to start indexer")?;

        // 3. GraphQL server on a random port
        let graphql_port = pick_available_port();
        let graphql_listen_address: SocketAddr =
            format!("127.0.0.1:{}", graphql_port).parse().unwrap();

        let pg_reader = Arc::new(
            PgReader::new(
                database.url().clone(),
                DbArgs {
                    db_connection_pool_size: 5,
                    db_connection_timeout_ms: 30_000,
                    db_statement_timeout_ms: None,
                },
            )
            .await
            .context("Failed to create PgReader for GraphQL")?,
        );

        let schema = build_schema(pg_reader, GraphQlConfig::default(), None);
        let app = build_router(AppState { schema });
        let listener = tokio::net::TcpListener::bind(graphql_listen_address)
            .await
            .context("Failed to bind GraphQL server")?;

        // 4. Spawn all services in a background task
        let service_handle = tokio::spawn(async move {
            let graphql_task =
                tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            // Run the indexer service until it completes (last_checkpoint reached)
            // or an error occurs.
            let _ = indexer_service.join().await;

            // Once indexer is done, abort the GraphQL server too.
            graphql_task.abort();
            let _ = graphql_task.await;
        });

        Ok(Self {
            db,
            graphql_listen_address,
            pipelines,
            _service: service_handle,
            _database: database,
        })
    }

    /// Poll the `watermarks` table until the minimum `checkpoint_hi_inclusive` across
    /// all registered pipelines is >= `checkpoint`.
    ///
    /// Mirrors Sui's `wait_for_indexer()` with 200ms polling interval.
    pub async fn wait_for_indexer(
        &self,
        checkpoint: u64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, async {
            let mut interval = interval(Duration::from_millis(200));
            loop {
                interval.tick().await;
                if matches!(self.latest_checkpoint().await, Ok(Some(l)) if l >= checkpoint) {
                    break;
                }
            }
        })
        .await
    }

    /// Poll the GraphQL endpoint until it reports a checkpoint >= `checkpoint`.
    pub async fn wait_for_graphql(
        &self,
        checkpoint: u64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        let client = reqwest::Client::new();
        let url = self.graphql_url();
        tokio::time::timeout(timeout, async move {
            let mut interval = interval(Duration::from_millis(200));
            loop {
                interval.tick().await;
                let resp = client
                    .post(&url)
                    .json(&serde_json::json!({
                        "query": "{ checkpoint { sequenceNumber } }"
                    }))
                    .send()
                    .await;
                if let Ok(resp) = resp {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if let Some(seq) = json["data"]["checkpoint"]["sequenceNumber"]
                            .as_str()
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            if seq >= checkpoint {
                                break;
                            }
                        }
                    }
                }
            }
        })
        .await
    }

    /// Query the `watermarks` table and return the minimum `checkpoint_hi_inclusive`
    /// across all registered pipelines.
    ///
    /// Returns `None` until every pipeline has written at least one watermark.
    pub async fn latest_checkpoint(&self) -> Result<Option<u64>> {
        use watermarks::dsl as w;

        let mut conn = self.db.connect().await?;
        let rows: Vec<(String, i64)> = w::watermarks
            .select((w::pipeline, w::checkpoint_hi_inclusive))
            .filter(w::pipeline.eq_any(&self.pipelines))
            .load(conn.deref_mut())
            .await?;

        if rows.len() < self.pipelines.len() {
            return Ok(None);
        }

        Ok(rows.into_iter().map(|(_, cp)| cp as u64).min())
    }

    /// The GraphQL endpoint URL.
    pub fn graphql_url(&self) -> String {
        format!("http://{}/graphql", self.graphql_listen_address)
    }

    /// Direct database access for verification queries.
    pub fn db(&self) -> &Db {
        &self.db
    }
}

impl Drop for OffchainCluster {
    fn drop(&mut self) {
        self._service.abort();
    }
}

/// Write a [`Checkpoint`] to disk as `{seq}.binpb.zst`.
///
/// Used for [`TestCheckpointBuilder`]-based tests where checkpoint data is synthetic.
pub fn write_checkpoint_file(dir: &Path, checkpoint: &Checkpoint) {
    let bytes = checkpoint_blob::encode_checkpoint(checkpoint).expect("Failed to encode checkpoint");
    let path = dir.join(format!(
        "{}.binpb.zst",
        checkpoint.summary.sequence_number
    ));
    std::fs::write(path, bytes).expect("Failed to write checkpoint file");
}

/// Pick an unused TCP port by briefly binding to port 0.
fn pick_available_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}
