// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Configuration for the GraphQL service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphQlConfig {
    /// Address to bind the server to.
    pub listen_address: String,
    /// Maximum page size for paginated queries.
    pub max_page_size: i32,
    /// Default page size when not specified.
    pub default_page_size: i32,
    /// Maximum query depth.
    pub max_query_depth: usize,
    /// Maximum query complexity.
    pub max_query_complexity: usize,
    /// Database connection pool size.
    pub db_pool_size: u32,
    /// Database connection timeout in milliseconds.
    pub db_connection_timeout_ms: u64,
    /// Database statement timeout in milliseconds.
    pub db_statement_timeout_ms: Option<u64>,
}

impl Default for GraphQlConfig {
    fn default() -> Self {
        Self {
            listen_address: "0.0.0.0:7000".to_string(),
            max_page_size: 50,
            default_page_size: 20,
            max_query_depth: 10,
            max_query_complexity: 1000,
            db_pool_size: 30,
            db_connection_timeout_ms: 30_000,
            db_statement_timeout_ms: Some(30_000),
        }
    }
}
