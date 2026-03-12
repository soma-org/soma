// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

/// Service configuration and limits for the GraphQL API.
pub struct ServiceConfig {
    pub max_page_size: i32,
    pub default_page_size: i32,
    pub max_query_depth: i32,
}

#[Object]
impl ServiceConfig {
    /// Maximum number of items per page.
    async fn max_page_size(&self) -> i32 {
        self.max_page_size
    }

    /// Default number of items per page when not specified.
    async fn default_page_size(&self) -> i32 {
        self.default_page_size
    }

    /// Maximum query depth.
    async fn max_query_depth(&self) -> i32 {
        self.max_query_depth
    }
}
