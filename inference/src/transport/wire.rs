// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire framing for the iroh inference transport.
//!
//! A request/response exchange rides a single bidirectional QUIC stream.
//! Both directions use the same shape:
//!
//! ```text
//! [u32-le meta_len][bcs(meta)][raw body bytes ... until stream finish]
//! ```
//!
//! The `meta` is a small BCS-encoded header (method/path/headers for a
//! request, status/headers for a response). The body is the raw payload —
//! for a non-streaming response it's the full JSON; for a streaming (SSE)
//! response it's the `data: …\n\n` byte stream, forwarded verbatim, with
//! end-of-body signalled by the sender finishing its half of the stream.
//! This mirrors how the HTTP relay forwards `reqwest`'s `bytes_stream()`.

use std::io;

use bytes::Bytes;
use iroh::endpoint::{RecvStream, SendStream};
use serde::{Deserialize, Serialize};

/// Max size of a framed `meta` header. Generous — real headers are a few
/// hundred bytes — but bounded so a malformed length prefix can't make us
/// allocate unboundedly.
pub const MAX_META_LEN: usize = 64 * 1024;

/// Max request/response body buffered when reading to end (16 MiB), matching
/// the provider server's `MAX_BODY` in [`crate::server::auth`].
pub const MAX_BODY_LEN: usize = 16 * 1024 * 1024;

/// Per-`read_chunk` ceiling while streaming a body.
const CHUNK: usize = 64 * 1024;

/// Request header sent buyer → provider. Carries everything the HTTP relay
/// previously put on the wire as method + path + headers (including the
/// `Authorization: SomaPay …` voucher header and `x-request-id`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireRequestMeta {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
}

/// Response header sent provider → buyer, ahead of the body bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireResponseMeta {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

/// Encode `meta` to the bytes of its framed `[u32-le len][bcs]` prefix.
/// Split out from the stream write so it can be unit-tested without iroh.
pub fn encode_meta<T: Serialize>(meta: &T) -> io::Result<Vec<u8>> {
    let bytes = bcs::to_bytes(meta).map_err(io_err)?;
    if bytes.len() > MAX_META_LEN {
        return Err(io_err("meta exceeds MAX_META_LEN"));
    }
    let mut out = Vec::with_capacity(4 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
    Ok(out)
}

/// Decode a framed meta from its `[u32-le len][bcs]` bytes.
pub fn decode_meta<T: serde::de::DeserializeOwned>(framed: &[u8]) -> io::Result<T> {
    let len_bytes: [u8; 4] =
        framed.get(..4).ok_or_else(|| io_err("short meta frame"))?.try_into().unwrap();
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_META_LEN {
        return Err(io_err("meta exceeds MAX_META_LEN"));
    }
    let body = framed.get(4..4 + len).ok_or_else(|| io_err("truncated meta frame"))?;
    bcs::from_bytes(body).map_err(io_err)
}

/// Write the framed `meta` prefix to a send stream.
pub async fn write_meta<T: Serialize>(send: &mut SendStream, meta: &T) -> io::Result<()> {
    let framed = encode_meta(meta)?;
    send.write_all(&framed).await.map_err(io_err)
}

/// Read the framed `meta` prefix from a recv stream, leaving the body bytes
/// (everything after it) unread on the stream.
pub async fn read_meta<T: serde::de::DeserializeOwned>(recv: &mut RecvStream) -> io::Result<T> {
    let mut len_bytes = [0u8; 4];
    recv.read_exact(&mut len_bytes).await.map_err(io_err)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_META_LEN {
        return Err(io_err("meta exceeds MAX_META_LEN"));
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await.map_err(io_err)?;
    bcs::from_bytes(&buf).map_err(io_err)
}

/// Read the rest of a recv stream (the body) to end, bounded by `max`.
pub async fn read_body_to_end(recv: &mut RecvStream, max: usize) -> io::Result<Bytes> {
    let mut buf = Vec::new();
    while let Some(chunk) = recv.read_chunk(CHUNK).await.map_err(io_err)? {
        if buf.len() + chunk.bytes.len() > max {
            return Err(io_err("body exceeds limit"));
        }
        buf.extend_from_slice(&chunk.bytes);
    }
    Ok(Bytes::from(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_meta_round_trips() {
        let meta = WireRequestMeta {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            headers: vec![
                ("authorization".into(), "SomaPay v2 0xabc …".into()),
                ("x-request-id".into(), "deadbeef".into()),
            ],
        };
        let framed = encode_meta(&meta).unwrap();
        // First 4 bytes are the little-endian length of the bcs payload.
        let len = u32::from_le_bytes(framed[..4].try_into().unwrap()) as usize;
        assert_eq!(len, framed.len() - 4);
        let back: WireRequestMeta = decode_meta(&framed).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn response_meta_round_trips() {
        let meta = WireResponseMeta {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
        };
        let framed = encode_meta(&meta).unwrap();
        let back: WireResponseMeta = decode_meta(&framed).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn decode_rejects_truncated_frame() {
        let meta = WireResponseMeta { status: 204, headers: vec![] };
        let framed = encode_meta(&meta).unwrap();
        // Chop a byte off the bcs payload → must error, not panic.
        assert!(decode_meta::<WireResponseMeta>(&framed[..framed.len() - 1]).is_err());
        // A length prefix claiming more than is present.
        assert!(decode_meta::<WireResponseMeta>(&[0xff, 0xff, 0x00, 0x00]).is_err());
    }
}
