// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod temp;

use std::ops::Deref;
use std::ops::DerefMut;
use std::time::Duration;

use anyhow::anyhow;
use async_trait::async_trait;
use chrono::DateTime;
use chrono::Utc;
use diesel::migration::MigrationSource;
use diesel::migration::MigrationVersion;
use diesel::pg::Pg;
use diesel::prelude::*;
use diesel::sql_types::BigInt;
use diesel::ExpressionMethods;
use diesel::OptionalExtension;
use diesel_async::async_connection_wrapper::AsyncConnectionWrapper;
use diesel_async::pooled_connection::bb8::Pool;
use diesel_async::pooled_connection::bb8::PooledConnection;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;
use diesel_async::AsyncConnection;
use diesel_async::AsyncPgConnection;
use diesel_async::RunQueryDsl;
use diesel_migrations::embed_migrations;
use diesel_migrations::EmbeddedMigrations;
use diesel_migrations::MigrationHarness;
use indexer_store_traits as store;
use scoped_futures::ScopedBoxFuture;
use tracing::info;
use url::Url;

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

// ---------------------------------------------------------------------------
// Diesel schema for the `watermarks` table
// ---------------------------------------------------------------------------

diesel::table! {
    watermarks (pipeline) {
        pipeline -> Text,
        epoch_hi_inclusive -> Int8,
        checkpoint_hi_inclusive -> Int8,
        tx_hi -> Int8,
        timestamp_ms_hi_inclusive -> Int8,
        reader_lo -> Int8,
        pruner_timestamp -> Timestamp,
        pruner_hi -> Int8,
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Insertable, Selectable, Queryable, Debug, Clone)]
#[diesel(table_name = watermarks)]
pub struct StoredWatermark {
    pub pipeline: String,
    pub epoch_hi_inclusive: i64,
    pub checkpoint_hi_inclusive: i64,
    pub tx_hi: i64,
    pub timestamp_ms_hi_inclusive: i64,
    pub reader_lo: i64,
    pub pruner_timestamp: chrono::NaiveDateTime,
    pub pruner_hi: i64,
}

// ---------------------------------------------------------------------------
// Database pool
// ---------------------------------------------------------------------------

/// Arguments controlling pool behaviour.
#[derive(Debug, Clone)]
pub struct DbArgs {
    /// Number of connections to keep in the pool.
    pub db_connection_pool_size: u32,
    /// Time spent waiting for a connection from the pool, in milliseconds.
    pub db_connection_timeout_ms: u64,
    /// Time spent waiting for statements to complete, in milliseconds.
    pub db_statement_timeout_ms: Option<u64>,
}

impl Default for DbArgs {
    fn default() -> Self {
        Self {
            db_connection_pool_size: 100,
            db_connection_timeout_ms: 60_000,
            db_statement_timeout_ms: None,
        }
    }
}

impl DbArgs {
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_millis(self.db_connection_timeout_ms)
    }

    pub fn statement_timeout(&self) -> Option<Duration> {
        self.db_statement_timeout_ms.map(Duration::from_millis)
    }
}

/// Wrapper around a `bb8` connection pool backed by async-diesel.
#[derive(Clone)]
pub struct Db {
    pool: Pool<AsyncPgConnection>,
}

/// Wrapper struct over the remote `PooledConnection` type for implementing the store traits.
pub struct Connection<'a>(PooledConnection<'a, AsyncPgConnection>);

impl Db {
    /// Construct a new DB connection pool for reads and writes.
    pub async fn for_write(database_url: Url, config: DbArgs) -> anyhow::Result<Self> {
        Self::new(database_url, config, false).await
    }

    /// Construct a new DB connection pool that defaults to read-only transactions.
    pub async fn for_read(database_url: Url, config: DbArgs) -> anyhow::Result<Self> {
        Self::new(database_url, config, true).await
    }

    async fn new(database_url: Url, db_args: DbArgs, read_only: bool) -> anyhow::Result<Self> {
        let pool = create_pool(database_url, db_args, read_only).await?;
        Ok(Db { pool })
    }

    /// Retrieve a connection from the pool.
    pub async fn connect(&self) -> anyhow::Result<Connection<'_>> {
        Ok(Connection(self.pool.get().await?))
    }

    /// Statistics about the connection pool.
    pub fn state(&self) -> bb8::State {
        self.pool.state()
    }

    /// Run the embedded migrations (and optionally additional ones) against the database.
    pub async fn run_migrations(
        &self,
        migrations: Option<&'static EmbeddedMigrations>,
    ) -> anyhow::Result<Vec<MigrationVersion<'static>>> {
        let merged = merge_migrations(migrations);

        info!("Running migrations ...");
        let conn = self.pool.dedicated_connection().await?;
        let mut wrapper: AsyncConnectionWrapper<AsyncPgConnection> =
            AsyncConnectionWrapper::from(conn);

        let finished: Vec<MigrationVersion<'static>> = tokio::task::spawn_blocking(move || {
            wrapper
                .run_pending_migrations(merged)
                .map(|versions| {
                    versions
                        .iter()
                        .map(MigrationVersion::as_owned)
                        .collect::<Vec<_>>()
                })
        })
        .await?
        .map_err(|e| anyhow!("Failed to run migrations: {:?}", e))?;

        info!("Migrations complete.");
        Ok(finished)
    }

    /// Drop all tables, procedures, and functions from the database.
    pub async fn clear_database(&self) -> anyhow::Result<()> {
        info!("Clearing the database...");
        let mut conn = self.connect().await?;

        let drop_all_tables = "
        DO $$ DECLARE
            r RECORD;
        BEGIN
        FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
            LOOP
                EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
            END LOOP;
        END $$;";
        diesel::sql_query(drop_all_tables)
            .execute(conn.deref_mut())
            .await?;

        let drop_all_procedures = "
        DO $$ DECLARE
            r RECORD;
        BEGIN
            FOR r IN (SELECT proname, oidvectortypes(proargtypes) as argtypes
                      FROM pg_proc INNER JOIN pg_namespace ns ON (pg_proc.pronamespace = ns.oid)
                      WHERE ns.nspname = 'public' AND prokind = 'p')
            LOOP
                EXECUTE 'DROP PROCEDURE IF EXISTS ' || quote_ident(r.proname) || '(' || r.argtypes || ') CASCADE';
            END LOOP;
        END $$;";
        diesel::sql_query(drop_all_procedures)
            .execute(conn.deref_mut())
            .await?;

        let drop_all_functions = "
        DO $$ DECLARE
            r RECORD;
        BEGIN
            FOR r IN (SELECT proname, oidvectortypes(proargtypes) as argtypes
                      FROM pg_proc INNER JOIN pg_namespace ON (pg_proc.pronamespace = pg_namespace.oid)
                      WHERE pg_namespace.nspname = 'public' AND prokind = 'f')
            LOOP
                EXECUTE 'DROP FUNCTION IF EXISTS ' || quote_ident(r.proname) || '(' || r.argtypes || ') CASCADE';
            END LOOP;
        END $$;";
        diesel::sql_query(drop_all_functions)
            .execute(conn.deref_mut())
            .await?;

        info!("Database cleared.");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Store trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl store::Connection for Connection<'_> {
    async fn init_watermark(
        &mut self,
        pipeline_task: &str,
        default_next_checkpoint: u64,
    ) -> anyhow::Result<Option<u64>> {
        let Some(checkpoint_hi_inclusive) = default_next_checkpoint.checked_sub(1) else {
            // Do not create a watermark record with checkpoint_hi_inclusive = -1.
            return Ok(self
                .committer_watermark(pipeline_task)
                .await?
                .map(|w| w.checkpoint_hi_inclusive));
        };

        let stored = StoredWatermark {
            pipeline: pipeline_task.to_string(),
            epoch_hi_inclusive: 0,
            checkpoint_hi_inclusive: checkpoint_hi_inclusive as i64,
            tx_hi: 0,
            timestamp_ms_hi_inclusive: 0,
            reader_lo: default_next_checkpoint as i64,
            pruner_timestamp: Utc::now().naive_utc(),
            pruner_hi: default_next_checkpoint as i64,
        };

        use diesel::pg::upsert::excluded;
        let checkpoint_hi: i64 = diesel::insert_into(watermarks::table)
            .values(&stored)
            .on_conflict(watermarks::pipeline)
            // Use `do_update` instead of `do_nothing` so that `returning` works.
            .do_update()
            // Set the pipeline to itself (nothing changes) to satisfy the requirement
            // that at least one column is updated.
            .set(watermarks::pipeline.eq(excluded(watermarks::pipeline)))
            .returning(watermarks::checkpoint_hi_inclusive)
            .get_result(self.deref_mut())
            .await?;

        Ok(Some(checkpoint_hi as u64))
    }

    async fn committer_watermark(
        &mut self,
        pipeline_task: &str,
    ) -> anyhow::Result<Option<store::CommitterWatermark>> {
        let watermark: Option<(i64, i64, i64, i64)> = watermarks::table
            .select((
                watermarks::epoch_hi_inclusive,
                watermarks::checkpoint_hi_inclusive,
                watermarks::tx_hi,
                watermarks::timestamp_ms_hi_inclusive,
            ))
            .filter(watermarks::pipeline.eq(pipeline_task))
            .first(self.deref_mut())
            .await
            .optional()?;

        Ok(watermark.map(|(epoch, cp, tx, ts)| store::CommitterWatermark {
            epoch: epoch as u64,
            checkpoint_hi_inclusive: cp as u64,
            tx_hi: tx as u64,
            timestamp_ms_hi_inclusive: ts as u64,
        }))
    }

    async fn reader_watermark(
        &mut self,
        pipeline: &'static str,
    ) -> anyhow::Result<Option<store::ReaderWatermark>> {
        let watermark: Option<(i64, i64)> = watermarks::table
            .select((watermarks::checkpoint_hi_inclusive, watermarks::reader_lo))
            .filter(watermarks::pipeline.eq(pipeline))
            .first(self.deref_mut())
            .await
            .optional()?;

        Ok(watermark.map(|(cp, lo)| store::ReaderWatermark {
            checkpoint_hi_inclusive: cp as u64,
            reader_lo: lo as u64,
        }))
    }

    async fn pruner_watermark(
        &mut self,
        pipeline: &'static str,
        delay: Duration,
    ) -> anyhow::Result<Option<store::PrunerWatermark>> {
        // Compute the remaining wait time server-side:
        //
        //   wait_for = delay + (pruner_timestamp - NOW())
        //
        // If negative the pruner may proceed immediately.
        let wait_for_ms_expr = diesel::dsl::sql::<BigInt>(&format!(
            "CAST({} + 1000 * EXTRACT(EPOCH FROM pruner_timestamp - NOW()) AS BIGINT)",
            delay.as_millis() as i64,
        ));

        let watermark: Option<(i64, i64, i64)> = watermarks::table
            .select((wait_for_ms_expr, watermarks::pruner_hi, watermarks::reader_lo))
            .filter(watermarks::pipeline.eq(pipeline))
            .first(self.deref_mut())
            .await
            .optional()?;

        Ok(watermark.map(|(wait_for_ms, hi, lo)| {
            let wait_for = if wait_for_ms > 0 {
                Duration::from_millis(wait_for_ms as u64)
            } else {
                Duration::ZERO
            };
            store::PrunerWatermark {
                wait_for,
                reader_lo: lo as u64,
                pruner_hi: hi as u64,
            }
        }))
    }

    async fn set_committer_watermark(
        &mut self,
        pipeline_task: &str,
        watermark: store::CommitterWatermark,
    ) -> anyhow::Result<bool> {
        let stored = StoredWatermark {
            pipeline: pipeline_task.to_string(),
            epoch_hi_inclusive: watermark.epoch as i64,
            checkpoint_hi_inclusive: watermark.checkpoint_hi_inclusive as i64,
            tx_hi: watermark.tx_hi as i64,
            timestamp_ms_hi_inclusive: watermark.timestamp_ms_hi_inclusive as i64,
            reader_lo: 0,
            pruner_timestamp: DateTime::UNIX_EPOCH.naive_utc(),
            pruner_hi: 0,
        };

        let rows_affected =
            diesel::query_dsl::methods::FilterDsl::filter(
                diesel::insert_into(watermarks::table)
                    .values(&stored)
                    .on_conflict(watermarks::pipeline)
                    .do_update()
                    .set((
                        watermarks::epoch_hi_inclusive.eq(stored.epoch_hi_inclusive),
                        watermarks::checkpoint_hi_inclusive.eq(stored.checkpoint_hi_inclusive),
                        watermarks::tx_hi.eq(stored.tx_hi),
                        watermarks::timestamp_ms_hi_inclusive.eq(stored.timestamp_ms_hi_inclusive),
                    )),
                watermarks::checkpoint_hi_inclusive.lt(stored.checkpoint_hi_inclusive),
            )
            .execute(self.deref_mut())
            .await?;

        Ok(rows_affected > 0)
    }

    async fn set_reader_watermark(
        &mut self,
        pipeline: &'static str,
        reader_lo: u64,
    ) -> anyhow::Result<bool> {
        Ok(diesel::update(watermarks::table)
            .set((
                watermarks::reader_lo.eq(reader_lo as i64),
                watermarks::pruner_timestamp.eq(diesel::dsl::now),
            ))
            .filter(watermarks::pipeline.eq(pipeline))
            .filter(watermarks::reader_lo.lt(reader_lo as i64))
            .execute(self.deref_mut())
            .await?
            > 0)
    }

    async fn set_pruner_watermark(
        &mut self,
        pipeline: &'static str,
        pruner_hi: u64,
    ) -> anyhow::Result<bool> {
        Ok(diesel::update(watermarks::table)
            .set(watermarks::pruner_hi.eq(pruner_hi as i64))
            .filter(watermarks::pipeline.eq(pipeline))
            .execute(self.deref_mut())
            .await?
            > 0)
    }
}

#[async_trait]
impl store::Store for Db {
    type Connection<'c> = Connection<'c>;

    async fn connect<'c>(&'c self) -> anyhow::Result<Self::Connection<'c>> {
        self.connect().await
    }
}

#[async_trait]
impl store::TransactionalStore for Db {
    async fn transaction<'a, T, F>(&self, f: F) -> anyhow::Result<T>
    where
        T: Send + 'a,
        F: for<'r> FnOnce(
                &'r mut Self::Connection<'_>,
            ) -> ScopedBoxFuture<'a, 'r, anyhow::Result<T>>
            + Send
            + 'a,
    {
        let mut conn = self.connect().await?;
        conn.transaction(f).await
    }
}

/// Helper on `Connection` to run a scoped transaction.
impl<'a> Connection<'a> {
    async fn transaction<'b, T, F>(&mut self, f: F) -> anyhow::Result<T>
    where
        T: Send + 'b,
        F: for<'r> FnOnce(
                &'r mut Connection<'_>,
            ) -> ScopedBoxFuture<'b, 'r, anyhow::Result<T>>
            + Send
            + 'b,
    {
        // We need to bridge between diesel-async's transaction API (which gives us
        // `&mut AsyncPgConnection`) and our `Connection` newtype.
        //
        // `Connection` is a `#[repr(transparent)]`-equivalent newtype over the pooled
        // connection which itself derefs to `AsyncPgConnection`, so the pointer cast is sound.
        AsyncConnection::transaction(self.deref_mut(), |inner| {
            // SAFETY: `Connection` is a newtype wrapper. We cast the `&mut AsyncPgConnection`
            // reference that diesel gives us into `&mut Connection<'_>`.  The reference is valid
            // for the duration of the closure.
            let conn_ref: &mut Connection<'_> =
                unsafe { &mut *(inner as *mut AsyncPgConnection as *mut Connection<'_>) };
            f(conn_ref)
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Deref to the inner pooled connection so Diesel queries work via `&mut conn`
// ---------------------------------------------------------------------------

impl<'a> Deref for Connection<'a> {
    type Target = AsyncPgConnection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Connection<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// Pool construction
// ---------------------------------------------------------------------------

async fn create_pool(
    database_url: Url,
    args: DbArgs,
    read_only: bool,
) -> anyhow::Result<Pool<AsyncPgConnection>> {
    let manager = AsyncDieselConnectionManager::<AsyncPgConnection>::new(database_url.as_str());

    if read_only || args.statement_timeout().is_some() {
        let statement_timeout = args.statement_timeout();

        #[derive(Debug)]
        struct SessionInit {
            read_only: bool,
            statement_timeout: Option<Duration>,
        }

        #[async_trait]
        impl bb8::CustomizeConnection<AsyncPgConnection, diesel_async::pooled_connection::PoolError>
            for SessionInit
        {
            async fn on_acquire(
                &self,
                conn: &mut AsyncPgConnection,
            ) -> Result<(), diesel_async::pooled_connection::PoolError> {
                if let Some(timeout) = self.statement_timeout {
                    diesel::sql_query(format!(
                        "SET statement_timeout = {}",
                        timeout.as_millis()
                    ))
                    .execute(conn)
                    .await
                    .map_err(diesel_async::pooled_connection::PoolError::QueryError)?;
                }

                if self.read_only {
                    diesel::sql_query("SET default_transaction_read_only = 'on'")
                        .execute(conn)
                        .await
                        .map_err(diesel_async::pooled_connection::PoolError::QueryError)?;
                }

                Ok(())
            }
        }

        let pool = Pool::builder()
            .max_size(args.db_connection_pool_size)
            .connection_timeout(args.connection_timeout())
            .connection_customizer(Box::new(SessionInit {
                read_only,
                statement_timeout,
            }))
            .build(manager)
            .await?;

        return Ok(pool);
    }

    Ok(Pool::builder()
        .max_size(args.db_connection_pool_size)
        .connection_timeout(args.connection_timeout())
        .build(manager)
        .await?)
}

// ---------------------------------------------------------------------------
// Migration merging
// ---------------------------------------------------------------------------

/// Returns migrations derived from the combination of this crate's embedded migrations and any
/// additional migrations provided.
pub fn merge_migrations(
    migrations: Option<&'static EmbeddedMigrations>,
) -> impl MigrationSource<Pg> + Send + Sync + 'static {
    struct Migrations(Option<&'static EmbeddedMigrations>);
    impl MigrationSource<Pg> for Migrations {
        fn migrations(
            &self,
        ) -> diesel::migration::Result<Vec<Box<dyn diesel::migration::Migration<Pg>>>> {
            let mut migrations = MIGRATIONS.migrations()?;
            if let Some(more) = self.0 {
                migrations.extend(more.migrations()?);
            }
            Ok(migrations)
        }
    }

    Migrations(migrations)
}

/// Drop all tables and re-run migrations if supplied.
pub async fn reset_database(
    database_url: Url,
    db_args: DbArgs,
    migrations: Option<&'static EmbeddedMigrations>,
) -> anyhow::Result<()> {
    let db = Db::for_write(database_url, db_args).await?;
    db.clear_database().await?;
    if let Some(migrations) = migrations {
        db.run_migrations(Some(migrations)).await?;
    }
    Ok(())
}
