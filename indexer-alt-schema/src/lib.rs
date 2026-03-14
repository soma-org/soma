// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel_migrations::EmbeddedMigrations;
use diesel_migrations::embed_migrations;

pub mod checkpoints;
pub mod cp_sequence_numbers;
pub mod epochs;
pub mod objects;
pub mod schema;
pub mod soma;
pub mod transactions;

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");
