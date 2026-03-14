// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Thin wrapper around `indexer_pg_db::Db` for read-only GraphQL queries.

use std::ops::DerefMut;
use std::sync::Arc;

use diesel::prelude::*;
use diesel_async::AsyncPgConnection;
use diesel_async::RunQueryDsl;
use indexer_pg_db::{Db, DbArgs};
use url::Url;

/// A read-only database reader for the GraphQL service.
#[derive(Clone)]
pub struct PgReader {
    db: Db,
}

impl PgReader {
    pub async fn new(database_url: Url, args: DbArgs) -> anyhow::Result<Self> {
        let db = Db::for_read(database_url, args).await?;
        Ok(Self { db })
    }

    pub async fn connect(&self) -> anyhow::Result<indexer_pg_db::Connection<'_>> {
        self.db.connect().await
    }
}
