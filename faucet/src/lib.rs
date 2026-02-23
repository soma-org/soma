// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//
// This file is derived from the Sui project (https://github.com/MystenLabs/sui),
// specifically crates/sui-faucet/src/lib.rs

// Server-only modules (need sdk, types, clap, etc.)
#[cfg(feature = "server")]
pub mod app_state;
pub mod codec;
#[cfg(feature = "server")]
pub mod errors;
#[cfg(feature = "server")]
pub mod faucet_config;
pub mod faucet_types;
#[cfg(feature = "server")]
pub mod local_faucet;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub mod types;

// Re-export tonic so downstream crates can use Channel
// without a direct tonic dependency.
pub use tonic;

// Tonic generated RPC stubs (client + server).
pub mod faucet_gen {
    include!("proto/faucet.Faucet.rs");
}
