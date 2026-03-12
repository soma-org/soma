// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

/// Base64-encoded binary data.
#[derive(Clone, Debug)]
pub struct Base64(pub Vec<u8>);

#[Scalar]
impl ScalarType for Base64 {
    fn parse(value: Value) -> InputValueResult<Self> {
        if let Value::String(s) = &value {
            let bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                s,
            )
            .map_err(InputValueError::custom)?;
            Ok(Base64(bytes))
        } else {
            Err(InputValueError::expected_type(value))
        }
    }

    fn to_value(&self) -> Value {
        Value::String(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &self.0,
        ))
    }
}

/// A 64-bit integer represented as a string to avoid JSON precision loss.
#[derive(Clone, Debug, Copy)]
pub struct BigInt(pub i64);

#[Scalar]
impl ScalarType for BigInt {
    fn parse(value: Value) -> InputValueResult<Self> {
        match &value {
            Value::String(s) => {
                let n: i64 = s.parse().map_err(InputValueError::custom)?;
                Ok(BigInt(n))
            }
            Value::Number(n) => {
                let n = n.as_i64().ok_or_else(|| {
                    InputValueError::custom("Expected integer")
                })?;
                Ok(BigInt(n))
            }
            _ => Err(InputValueError::expected_type(value)),
        }
    }

    fn to_value(&self) -> Value {
        Value::String(self.0.to_string())
    }
}

/// A Soma address encoded as a hex string with 0x prefix.
#[derive(Clone, Debug)]
pub struct SomaAddress(pub Vec<u8>);

#[Scalar]
impl ScalarType for SomaAddress {
    fn parse(value: Value) -> InputValueResult<Self> {
        if let Value::String(s) = &value {
            let s = s.strip_prefix("0x").unwrap_or(s);
            let bytes = hex::decode(s).map_err(InputValueError::custom)?;
            Ok(SomaAddress(bytes))
        } else {
            Err(InputValueError::expected_type(value))
        }
    }

    fn to_value(&self) -> Value {
        Value::String(format!("0x{}", hex::encode(&self.0)))
    }
}

/// A transaction or object digest encoded as base58.
#[derive(Clone, Debug)]
pub struct Digest(pub Vec<u8>);

#[Scalar]
impl ScalarType for Digest {
    fn parse(value: Value) -> InputValueResult<Self> {
        if let Value::String(s) = &value {
            let bytes = bs58::decode(s)
                .into_vec()
                .map_err(InputValueError::custom)?;
            Ok(Digest(bytes))
        } else {
            Err(InputValueError::expected_type(value))
        }
    }

    fn to_value(&self) -> Value {
        Value::String(bs58::encode(&self.0).into_string())
    }
}

/// A timestamp in milliseconds since Unix epoch, rendered as ISO 8601.
#[derive(Clone, Debug, Copy)]
pub struct DateTime(pub i64);

#[Scalar]
impl ScalarType for DateTime {
    fn parse(value: Value) -> InputValueResult<Self> {
        match &value {
            Value::String(s) => {
                let dt = chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(InputValueError::custom)?;
                Ok(DateTime(dt.timestamp_millis()))
            }
            Value::Number(n) => {
                let ms = n.as_i64().ok_or_else(|| {
                    InputValueError::custom("Expected integer timestamp_ms")
                })?;
                Ok(DateTime(ms))
            }
            _ => Err(InputValueError::expected_type(value)),
        }
    }

    fn to_value(&self) -> Value {
        let secs = self.0 / 1000;
        let nanos = ((self.0 % 1000) * 1_000_000) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
            Value::String(dt.to_rfc3339())
        } else {
            Value::String(self.0.to_string())
        }
    }
}
