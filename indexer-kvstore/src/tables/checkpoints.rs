// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Checkpoints table: stores full checkpoint data indexed by sequence number.

use anyhow::{Context, Result};
use bytes::Bytes;
use types::checkpoints::{CheckpointContents, CheckpointSequenceNumber, CheckpointSummary};
use types::crypto::AuthorityStrongQuorumSignInfo;

use crate::CheckpointData;

pub mod col {
    pub const SUMMARY: &str = "s";
    pub const SIGNATURES: &str = "sg";
    pub const CONTENTS: &str = "c";
}

pub const NAME: &str = "checkpoints";

pub fn encode_key(sequence_number: CheckpointSequenceNumber) -> Vec<u8> {
    sequence_number.to_be_bytes().to_vec()
}

pub fn encode(
    summary: &CheckpointSummary,
    signatures: &AuthorityStrongQuorumSignInfo,
    contents: &CheckpointContents,
) -> Result<[(&'static str, Bytes); 3]> {
    Ok([
        (col::SUMMARY, Bytes::from(bcs::to_bytes(summary)?)),
        (col::SIGNATURES, Bytes::from(bcs::to_bytes(signatures)?)),
        (col::CONTENTS, Bytes::from(bcs::to_bytes(contents)?)),
    ])
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<CheckpointData> {
    let mut summary = None;
    let mut contents = None;
    let mut signatures = None;

    for (column, value) in row {
        match column.as_ref() {
            b"s" => summary = Some(bcs::from_bytes::<CheckpointSummary>(value)?),
            b"c" => contents = Some(bcs::from_bytes::<CheckpointContents>(value)?),
            b"sg" => signatures = Some(bcs::from_bytes::<AuthorityStrongQuorumSignInfo>(value)?),
            _ => {}
        }
    }

    Ok(CheckpointData {
        summary: summary.context("summary field is missing")?,
        contents: contents.context("contents field is missing")?,
        signatures: signatures.context("signatures field is missing")?,
    })
}
