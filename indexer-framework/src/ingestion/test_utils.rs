// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test utilities for the ingestion module.
//!
//! NOTE: Constructing valid test checkpoints requires cryptographic signing infrastructure
//! (committee keys, authority signatures). Full test checkpoint construction should use the
//! test network infrastructure or genesis builder. The functions below are placeholders for
//! when that infrastructure is wired up.

use crate::types::full_checkpoint_content::Checkpoint;

/// Create test checkpoint data as zstd-compressed protobuf bytes.
///
/// Requires the test signing infrastructure to be available. Panics with a descriptive message
/// if called before that infrastructure is in place.
pub(crate) fn test_checkpoint_data(cp: u64) -> Vec<u8> {
    let checkpoint = test_checkpoint(cp);
    rpc::utils::checkpoint_blob::encode_checkpoint(&checkpoint)
        .expect("Failed to encode test checkpoint")
}

/// Create a test checkpoint with the given sequence number.
///
/// Requires the test signing infrastructure to be available. Panics with a descriptive message
/// if called before that infrastructure is in place.
pub(crate) fn test_checkpoint(_cp: u64) -> Checkpoint {
    unimplemented!(
        "Test checkpoint construction requires crypto signing infrastructure. \
         Use the test network or genesis builder for integration tests."
    )
}
