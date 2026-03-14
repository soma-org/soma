// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Epochs table: stores epoch info indexed by epoch ID.

use anyhow::Result;
use bytes::Bytes;
use types::committee::EpochId;

use crate::EpochData;

pub const NAME: &str = "epochs";

pub mod col {
    pub const EPOCH: &str = "ep";
    pub const PROTOCOL_VERSION: &str = "pv";
    pub const START_TIMESTAMP: &str = "st";
    pub const START_CHECKPOINT: &str = "sc";
    pub const REFERENCE_GAS_PRICE: &str = "rg";
    pub const SYSTEM_STATE: &str = "ss";
    pub const END_TIMESTAMP: &str = "et";
    pub const END_CHECKPOINT: &str = "ec";
    pub const CP_HI: &str = "ch";
    pub const TX_HI: &str = "th";
    pub const SAFE_MODE: &str = "sm";
    pub const EPOCH_COMMITMENTS: &str = "cm";
}

pub fn encode_key(epoch_id: EpochId) -> Vec<u8> {
    epoch_id.to_be_bytes().to_vec()
}

pub fn encode_key_upper_bound() -> Bytes {
    Bytes::from(u64::MAX.to_be_bytes().to_vec())
}

/// Encode epoch start data to individual columns.
pub fn encode_start(
    epoch: u64,
    protocol_version: u64,
    start_timestamp_ms: u64,
    start_checkpoint: u64,
    reference_gas_price: u64,
    system_state_bcs: &[u8],
) -> Vec<(&'static str, Bytes)> {
    vec![
        (col::EPOCH, Bytes::from(epoch.to_be_bytes().to_vec())),
        (col::PROTOCOL_VERSION, Bytes::from(protocol_version.to_be_bytes().to_vec())),
        (col::START_TIMESTAMP, Bytes::from(start_timestamp_ms.to_be_bytes().to_vec())),
        (col::START_CHECKPOINT, Bytes::from(start_checkpoint.to_be_bytes().to_vec())),
        (col::REFERENCE_GAS_PRICE, Bytes::from(reference_gas_price.to_be_bytes().to_vec())),
        (col::SYSTEM_STATE, Bytes::from(system_state_bcs.to_vec())),
    ]
}

/// Encode epoch end data to individual columns.
/// Soma has no SystemEpochInfoEvent so staking/storage fields are omitted.
pub fn encode_end(
    end_timestamp_ms: u64,
    end_checkpoint: u64,
    cp_hi: u64,
    tx_hi: u64,
    safe_mode: bool,
    epoch_commitments: &[u8],
) -> Vec<(&'static str, Bytes)> {
    vec![
        (col::END_TIMESTAMP, Bytes::from(end_timestamp_ms.to_be_bytes().to_vec())),
        (col::END_CHECKPOINT, Bytes::from(end_checkpoint.to_be_bytes().to_vec())),
        (col::CP_HI, Bytes::from(cp_hi.to_be_bytes().to_vec())),
        (col::TX_HI, Bytes::from(tx_hi.to_be_bytes().to_vec())),
        (col::SAFE_MODE, Bytes::from(vec![u8::from(safe_mode)])),
        (col::EPOCH_COMMITMENTS, Bytes::from(epoch_commitments.to_vec())),
    ]
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<EpochData> {
    let mut data = EpochData::default();

    for (col, value) in row {
        match col.as_ref() {
            b"ep" => data.epoch = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"pv" => data.protocol_version = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"st" => data.start_timestamp_ms = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"sc" => data.start_checkpoint = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"rg" => {
                data.reference_gas_price = Some(u64::from_be_bytes(value.as_ref().try_into()?))
            }
            b"ss" => data.system_state_bcs = Some(value.to_vec()),
            b"et" => data.end_timestamp_ms = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"ec" => data.end_checkpoint = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"ch" => data.cp_hi = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"th" => data.tx_hi = Some(u64::from_be_bytes(value.as_ref().try_into()?)),
            b"sm" => data.safe_mode = Some(value.as_ref().first().copied().unwrap_or(0) != 0),
            b"cm" => data.epoch_commitments = Some(value.to_vec()),
            _ => {}
        }
    }

    Ok(data)
}
