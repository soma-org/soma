// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::utils::checkpoint_blob::DecodeError;
use rpc::utils::checkpoint_blob::decode_checkpoint;

use crate::types::full_checkpoint_content::Checkpoint;

/// Wrapper error type that maps DecodeError variants to reason strings for metrics.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to decompress checkpoint bytes: {0}")]
    Decompression(std::io::Error),

    #[error("Failed to deserialize checkpoint protobuf: {0}")]
    Deserialization(anyhow::Error),

    #[error("Failed to convert checkpoint protobuf to checkpoint data: {0}")]
    ProtoConversion(anyhow::Error),
}

impl Error {
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            Self::Decompression(_) => "decompression",
            Self::Deserialization(_) => "deserialization",
            Self::ProtoConversion(_) => "proto_conversion",
        }
    }
}

impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        match e {
            DecodeError::Zstd(e) => Self::Decompression(e),
            DecodeError::Prost(e) => Self::Deserialization(e.into()),
            DecodeError::ProtoConversion(e) => Self::ProtoConversion(e.into()),
        }
    }
}

/// Decode the bytes of a checkpoint from the remote store. The bytes are expected to be a
/// [Checkpoint], represented as a protobuf message, in binary form, zstd-compressed.
pub(crate) fn checkpoint(bytes: &[u8]) -> Result<Checkpoint, Error> {
    Ok(decode_checkpoint(bytes)?)
}
