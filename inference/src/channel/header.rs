// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `Authorization: SomaPay v2 <channel_id> <bcs(http_voucher)_b64> <bcs(http_sig)_b64>`
//! and the companion `X-Soma-Onchain-Sig: <bcs(generic_sig)_b64>` —
//! per-request authorization headers.
//!
//! Both signatures are produced via the SDK's `sign_*_voucher`
//! helpers (Ed25519/MultiSig over `IntentMessage<X>` with their
//! respective scopes) — there is no inference-local signing
//! primitive any more. See `sdk::channel`.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use fastcrypto::traits::ToFromBytes as _;
use ::types::channel::HttpVoucher;
use ::types::crypto::GenericSignature;
use ::types::object::ObjectID;

use crate::channel::ChannelError;

#[derive(Debug, Clone)]
pub struct SomaPayHeader {
    pub channel_id: ObjectID,
    pub http_voucher: HttpVoucher,
    pub http_sig: GenericSignature,
}

const SCHEME: &str = "SomaPay";
const VERSION: &str = "v2";

impl SomaPayHeader {
    pub fn parse(value: &str) -> Result<Self, ChannelError> {
        let value = value.trim();
        let mut parts = value.split_whitespace();
        let scheme = parts.next().ok_or(ChannelError::Malformed)?;
        if !scheme.eq_ignore_ascii_case(SCHEME) {
            return Err(ChannelError::Malformed);
        }
        let version = parts.next().ok_or(ChannelError::Malformed)?;
        if version != VERSION {
            return Err(ChannelError::Malformed);
        }
        let channel_id_str = parts.next().ok_or(ChannelError::Malformed)?;
        let channel_id = channel_id_str
            .parse::<ObjectID>()
            .map_err(|_| ChannelError::Malformed)?;
        let voucher_b64 = parts.next().ok_or(ChannelError::Malformed)?;
        let sig_b64 = parts.next().ok_or(ChannelError::Malformed)?;
        if parts.next().is_some() {
            return Err(ChannelError::Malformed);
        }

        let voucher_bytes = URL_SAFE_NO_PAD
            .decode(voucher_b64.as_bytes())
            .map_err(|_| ChannelError::Malformed)?;
        let http_voucher: HttpVoucher =
            bcs::from_bytes(&voucher_bytes).map_err(|_| ChannelError::Malformed)?;
        if http_voucher.channel_id != channel_id {
            return Err(ChannelError::Malformed);
        }

        let sig_bytes = URL_SAFE_NO_PAD
            .decode(sig_b64.as_bytes())
            .map_err(|_| ChannelError::Malformed)?;
        let http_sig =
            GenericSignature::from_bytes(&sig_bytes).map_err(|_| ChannelError::Malformed)?;

        Ok(Self { channel_id, http_voucher, http_sig })
    }

    pub fn format(&self) -> String {
        let voucher_bytes =
            bcs::to_bytes(&self.http_voucher).expect("HttpVoucher BCS infallible");
        let voucher_b64 = URL_SAFE_NO_PAD.encode(&voucher_bytes);
        let sig_b64 = URL_SAFE_NO_PAD.encode(self.http_sig.as_ref());
        format!(
            "{} {} {} {} {}",
            SCHEME, VERSION, self.channel_id, voucher_b64, sig_b64
        )
    }
}

/// Wire format for the on-chain voucher signature companion header.
/// The provider stores the latest one and uses it when calling
/// `sdk::channel::settle` on shutdown.
pub fn encode_onchain_sig(sig: &GenericSignature) -> String {
    URL_SAFE_NO_PAD.encode(sig.as_ref())
}

pub fn decode_onchain_sig(value: &str) -> Result<GenericSignature, ChannelError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(value.trim().as_bytes())
        .map_err(|_| ChannelError::Malformed)?;
    GenericSignature::from_bytes(&bytes).map_err(|_| ChannelError::Malformed)
}

pub const ONCHAIN_SIG_HEADER: &str = "x-soma-onchain-sig";

#[cfg(test)]
mod tests {
    use super::*;
    use ::types::channel::HttpVoucher;
    use ::types::object::ObjectID;

    #[test]
    fn round_trip_parse_format() {
        // Minimal header round-trip with a known-shape signature; we
        // can't produce a real signature here without a keystore, so
        // fall back to verifying the parse rejects mismatched
        // channel_ids.
        let id_a = ObjectID::random();
        let id_b = ObjectID::random();
        let hv = HttpVoucher::from_request(id_a, 100, 0, b"", "rid", "POST", "/v1/x");
        // Bogus 65-byte Ed25519 sig: 1-byte flag (0 == ED25519) + 64 zero bytes + 32 zero pk = 97 bytes.
        let mut sig_bytes = vec![0u8; 97];
        sig_bytes[0] = 0; // ED25519 flag
        let http_sig = GenericSignature::from_bytes(&sig_bytes).expect("ed25519 sig parses");
        let h = SomaPayHeader { channel_id: id_a, http_voucher: hv, http_sig };
        let s = h.format();
        let parsed = SomaPayHeader::parse(&s).expect("round-trips");
        assert_eq!(parsed.channel_id, id_a);
        assert_eq!(parsed.http_voucher.cumulative_amount, 100);

        // Channel-id mismatch must be rejected.
        let hv_b = HttpVoucher::from_request(id_b, 100, 0, b"", "rid", "POST", "/v1/x");
        let mut sig_bytes2 = vec![0u8; 97];
        sig_bytes2[0] = 0;
        let bad = SomaPayHeader {
            channel_id: id_a,
            http_voucher: hv_b,
            http_sig: GenericSignature::from_bytes(&sig_bytes2).unwrap(),
        };
        let s = bad.format();
        assert!(SomaPayHeader::parse(&s).is_err());
    }
}
