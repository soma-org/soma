// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use anyhow::{Result, anyhow, bail};
use fastcrypto::encoding::{Encoding, Hex};
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::object::ObjectRef;

/// Parse a hex string into a 32-byte array, stripping an optional 0x prefix.
pub fn parse_hex_digest_32(hex_str: &str, field_name: &str) -> Result<[u8; 32]> {
    let bytes = Hex::decode(hex_str.strip_prefix("0x").unwrap_or(hex_str))
        .map_err(|e| anyhow!("Invalid hex for {}: {}", field_name, e))?;
    let arr: [u8; 32] =
        bytes.try_into().map_err(|_| anyhow!("{} must be exactly 32 bytes", field_name))?;
    Ok(arr)
}

/// Parse a comma-separated string of f32 values into an embedding vector.
pub fn parse_embedding(embedding_str: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = embedding_str
        .split(',')
        .map(|s| {
            let trimmed = s.trim();
            trimmed
                .parse::<f32>()
                .map_err(|_| anyhow!(
                    "Invalid embedding value '{}': not a valid f32 (expected comma-separated floats, e.g., '0.1,0.2,0.3')",
                    trimmed
                ))
        })
        .collect::<Result<Vec<f32>>>()?;

    if values.is_empty() {
        bail!("Embedding cannot be empty");
    }

    Ok(values)
}

/// Read a file and compute its Blake2b-256 commitment and size.
pub fn read_and_hash_file(path: &Path) -> Result<([u8; 32], String, usize)> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow!("Failed to read file '{}': {}", path.display(), e))?;
    let commitment = sdk::crypto_utils::commitment(&data);
    let commitment_hex = sdk::crypto_utils::commitment_hex(&data);
    let size = data.len();
    Ok((commitment, commitment_hex, size))
}

/// Auto-fetch the coin with the highest balance for bond payment.
pub async fn auto_fetch_bond_coin(
    context: &WalletContext,
    sender: SomaAddress,
) -> Result<ObjectRef> {
    context
        .get_richest_gas_object_owned_by_address(sender)
        .await?
        .ok_or_else(|| anyhow!("No coins found for address {}", sender))
}
