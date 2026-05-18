// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

// Command modules for top-level CLI commands
pub mod balance;
pub mod channel;
pub mod env;
pub mod inference;
pub mod model;
pub mod objects;
pub mod offering;
pub mod provider;
pub mod stake;
pub mod transfer;
pub mod tx;
pub mod validator;
pub mod wallet;

// Re-export subcommand enums for use in soma_commands.rs
pub use channel::ChannelCommand;
pub use env::EnvCommand;
pub use inference::InferenceCommand;
pub use model::ModelCommand;
pub use objects::ObjectsCommand;
pub use offering::OfferingCommand;
pub use provider::ProviderCommand;
pub use validator::SomaValidatorCommand;
pub use wallet::WalletCommand;
