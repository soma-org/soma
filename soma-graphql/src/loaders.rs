// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! DataLoader implementations for batching nested resolver queries.
//!
//! These loaders implement `async_graphql::dataloader::Loader` to batch multiple
//! per-parent database queries into a single query, preventing N+1 query storms.

// Marketplace loaders can be added here as needed (e.g., batch loading bids for
// multiple asks, or settlements for multiple addresses). The old target/reward
// loaders have been removed as part of the marketplace refactor.
