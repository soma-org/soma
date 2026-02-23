// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//
// Modified for the Soma project.

pub mod discovery;

// Tonic generated RPC stubs.
pub mod tonic_gen {
    include!("proto/p2p.P2p.rs");
}
pub mod builder;
pub mod server;
pub mod state_sync;
pub(crate) mod test_utils;
