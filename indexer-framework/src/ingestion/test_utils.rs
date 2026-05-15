// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test utilities for the ingestion module.
//!
//! Backed by `types::test_checkpoint_data_builder::TestCheckpointBuilder`, which assembles a
//! fully-formed [`Checkpoint`] (committee-signed summary, contents, transactions, object set)
//! without a running network — the same builder soma's other indexer tests use.

use crate::types::full_checkpoint_content::Checkpoint;
use crate::types::test_checkpoint_data_builder::TestCheckpointBuilder;

/// Build a test [`Checkpoint`] with the given sequence number.
pub(crate) fn test_checkpoint(cp: u64) -> Checkpoint {
    TestCheckpointBuilder::new(cp).build()
}

/// Build a test checkpoint and encode it to the on-disk `.binpb.zst` form (protobuf + zstd) —
/// the exact bytes an ingestion client reads from the remote store.
pub(crate) fn test_checkpoint_data(cp: u64) -> Vec<u8> {
    rpc::utils::checkpoint_blob::encode_checkpoint(&test_checkpoint(cp))
        .expect("Failed to encode test checkpoint")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A built test checkpoint carries its sequence number and survives the encode→decode
    /// round-trip used by the ingestion path.
    #[test]
    fn test_checkpoint_builds_and_roundtrips() {
        let checkpoint = test_checkpoint(7);
        assert_eq!(checkpoint.summary.sequence_number, 7);

        let bytes = test_checkpoint_data(7);
        let decoded = rpc::utils::checkpoint_blob::decode_checkpoint(&bytes)
            .expect("test checkpoint bytes must decode");
        assert_eq!(decoded.summary.sequence_number, 7);
    }
}
