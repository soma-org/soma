// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for indexer handler tests.
//!
//! Provides:
//! - `test_db()` — spin up an ephemeral Postgres and run all migrations
//! - Helper to get a diesel connection for direct queries

use std::ops::DerefMut;

use indexer_pg_db::temp::TempDb;
use indexer_pg_db::{Db, DbArgs};

/// Spin up an ephemeral Postgres and run all indexer migrations.
/// Returns `(Db, TempDb)` — keep `TempDb` alive for the duration of the test.
pub async fn test_db() -> (Db, TempDb) {
    let temp = TempDb::new();
    let db = Db::for_write(
        temp.url().clone(),
        DbArgs {
            db_connection_pool_size: 5,
            db_connection_timeout_ms: 30_000,
            db_statement_timeout_ms: None,
        },
    )
    .await
    .expect("failed to create DB pool");

    // Run pg-db watermarks migration + indexer-alt-schema migrations
    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS))
        .await
        .expect("failed to run migrations");

    (db, temp)
}

/// Get a mutable diesel connection from the pool.
pub async fn get_connection(db: &Db) -> indexer_pg_db::Connection<'_> {
    db.connect().await.expect("failed to get DB connection")
}
