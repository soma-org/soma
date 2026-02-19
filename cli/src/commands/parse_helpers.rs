use anyhow::{Result, anyhow, bail};
use fastcrypto::encoding::{Encoding, Hex};

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
