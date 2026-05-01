// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `Authorization: SomaPay v1 <handle> <cum_micros> <expires_ms> <sig>` —
//! per-request authorization header. The signature is detached Ed25519 over
//! a SHA-256 digest of the canonical input.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::channel::ChannelError;

#[derive(Debug, Clone)]
pub struct SomaPayHeader {
    pub handle: String,
    pub cum_micros: u64,
    pub expires_ms: u64,
    pub sig_b64: String, // url-safe-no-pad
}

impl SomaPayHeader {
    pub fn parse(value: &str) -> Result<Self, ChannelError> {
        let value = value.trim();
        let mut parts = value.split_whitespace();
        let scheme = parts.next().ok_or(ChannelError::Malformed)?;
        if !scheme.eq_ignore_ascii_case("SomaPay") {
            return Err(ChannelError::Malformed);
        }
        let version = parts.next().ok_or(ChannelError::Malformed)?;
        if version != "v1" {
            return Err(ChannelError::Malformed);
        }
        let handle = parts.next().ok_or(ChannelError::Malformed)?.to_string();
        let cum: u64 = parts
            .next()
            .ok_or(ChannelError::Malformed)?
            .parse()
            .map_err(|_| ChannelError::Malformed)?;
        let expires: u64 = parts
            .next()
            .ok_or(ChannelError::Malformed)?
            .parse()
            .map_err(|_| ChannelError::Malformed)?;
        let sig = parts.next().ok_or(ChannelError::Malformed)?.to_string();
        if parts.next().is_some() {
            return Err(ChannelError::Malformed);
        }
        Ok(Self { handle, cum_micros: cum, expires_ms: expires, sig_b64: sig })
    }

    pub fn format(&self) -> String {
        format!(
            "SomaPay v1 {} {} {} {}",
            self.handle, self.cum_micros, self.expires_ms, self.sig_b64
        )
    }

    pub fn sig_bytes(&self) -> Result<[u8; 64], ChannelError> {
        let v = URL_SAFE_NO_PAD
            .decode(self.sig_b64.as_bytes())
            .map_err(|_| ChannelError::Malformed)?;
        if v.len() != 64 {
            return Err(ChannelError::Malformed);
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&v);
        Ok(arr)
    }
}

/// Canonical bytes-to-be-signed for one authorization. Newline-delimited so
/// adversarial input can't reframe one field as another.
pub fn digest_input(
    handle: &str,
    cum_micros: u64,
    expires_ms: u64,
    method: &str,
    path: &str,
    body_sha256_hex: &str,
    request_id: &str,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(b"SomaPay/v1\n");
    out.extend_from_slice(handle.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(cum_micros.to_string().as_bytes());
    out.push(b'\n');
    out.extend_from_slice(expires_ms.to_string().as_bytes());
    out.push(b'\n');
    out.extend_from_slice(method.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(path.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(body_sha256_hex.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(request_id.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_format_parse() {
        let h = SomaPayHeader {
            handle: "01HZZK".to_string(),
            cum_micros: 12345,
            expires_ms: 1700000000000,
            sig_b64: "AAAA".to_string(),
        };
        let s = h.format();
        let p = SomaPayHeader::parse(&s).unwrap();
        assert_eq!(p.handle, h.handle);
        assert_eq!(p.cum_micros, h.cum_micros);
        assert_eq!(p.expires_ms, h.expires_ms);
        assert_eq!(p.sig_b64, h.sig_b64);
    }
}
