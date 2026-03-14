// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Soma GraphQL service — provides a GraphQL API backed by the Postgres indexer.

pub mod api;
pub mod config;
pub mod db;
pub mod loaders;

use std::sync::Arc;

use async_graphql::dataloader::DataLoader;
use async_graphql::{EmptyMutation, EmptySubscription, Schema};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::Router;
use axum::extract::State;
use axum::response::Html;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;

use crate::api::query::Query;
use crate::config::GraphQlConfig;
use crate::db::PgReader;
use crate::loaders::{TargetReportersLoader, TargetRewardLoader};

pub use indexer_kvstore::KvLoader;

pub type SomaSchema = Schema<Query, EmptyMutation, EmptySubscription>;

/// Application state shared across all handlers.
#[derive(Clone)]
pub struct AppState {
    pub schema: SomaSchema,
}

/// Build the async-graphql schema with all context data attached.
///
/// When `kv` is provided, resolvers read BCS content from BigTable instead of
/// Postgres `kv_*` tables, making Postgres pruning invisible to clients.
pub fn build_schema(
    pg: Arc<PgReader>,
    config: GraphQlConfig,
    kv: Option<Arc<dyn KvLoader>>,
) -> SomaSchema {
    let reporters_loader = DataLoader::new(TargetReportersLoader { pg: pg.clone() }, tokio::spawn);
    let reward_loader = DataLoader::new(TargetRewardLoader { pg: pg.clone() }, tokio::spawn);

    let mut builder = Schema::build(Query, EmptyMutation, EmptySubscription)
        .data(pg)
        .data(config.clone())
        .data(reporters_loader)
        .data(reward_loader)
        .limit_depth(config.max_query_depth)
        .limit_complexity(config.max_query_complexity);
    if let Some(kv) = kv {
        builder = builder.data(kv);
    }
    builder.finish()
}

/// Build the axum router.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/graphql", post(graphql_handler))
        .route("/graphql", get(graphiql_handler))
        .route("/graphql/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn graphql_handler(State(state): State<AppState>, req: GraphQLRequest) -> GraphQLResponse {
    state.schema.execute(req.into_inner()).await.into()
}

async fn graphiql_handler() -> Html<String> {
    Html(async_graphql::http::GraphiQLSource::build().endpoint("/graphql").finish())
}

async fn health_handler() -> &'static str {
    "OK"
}
