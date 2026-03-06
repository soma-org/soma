// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint blob encoding/decoding in `.binpb.zst` format.
//!
//! This matches Sui's checkpoint file format: protobuf-encoded checkpoint
//! compressed with zstd. The indexer framework's ingestion layer expects
//! this format.

use prost::Message;
use types::full_checkpoint_content::Checkpoint;

use crate::proto::soma::Checkpoint as ProtoCheckpoint;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

/// Encode a checkpoint into the `.binpb.zst` format (proto + zstd).
pub fn encode_checkpoint(checkpoint: &Checkpoint) -> Result<Vec<u8>, EncodeError> {
    let mask = FieldMaskTree::new_wildcard();
    let proto_checkpoint = ProtoCheckpoint::merge_from(checkpoint, &mask);
    let proto_bytes = proto_checkpoint.encode_to_vec();
    let compressed = zstd::encode_all(&proto_bytes[..], 3)?;
    Ok(compressed)
}

/// Decode a checkpoint from the `.binpb.zst` format (proto + zstd).
pub fn decode_checkpoint(bytes: &[u8]) -> Result<Checkpoint, DecodeError> {
    let decompressed = zstd::decode_all(bytes)?;
    let proto_checkpoint = ProtoCheckpoint::decode(&decompressed[..])?;
    let checkpoint = Checkpoint::try_from(&proto_checkpoint)?;
    Ok(checkpoint)
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("zstd compression error: {0}")]
    Zstd(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("zstd decompression error: {0}")]
    Zstd(#[from] std::io::Error),
    #[error("protobuf decode error: {0}")]
    Prost(#[from] prost::DecodeError),
    #[error("proto conversion error: {0}")]
    ProtoConversion(#[from] crate::proto::TryFromProtoError),
}
