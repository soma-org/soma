// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! gRPC faucet client for requesting test tokens.
//!
//! Types and codec are defined inline to avoid a cyclic dependency
//! (`sdk -> faucet -> sdk`). The wire format (BCS over gRPC) is
//! identical to the faucet server.

use std::marker::PhantomData;

use bytes::{Buf, BufMut};
use serde::{Deserialize, Serialize};
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

// ---------------------------------------------------------------------------
// Faucet types (mirrors faucet::faucet_types)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct GasRequest {
    pub recipient: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasResponse {
    pub status: String,
    #[serde(default)]
    pub coins_sent: Vec<GasCoinInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCoinInfo {
    pub amount: u64,
    pub id: String,
    pub transfer_tx_digest: String,
}

// ---------------------------------------------------------------------------
// BCS codec (mirrors faucet::codec)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct BcsEncoder<T>(PhantomData<T>);

impl<T: Serialize> Encoder for BcsEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, buf: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        bcs::serialize_into(&mut buf.writer(), &item).map_err(|e| Status::internal(e.to_string()))
    }
}

#[derive(Debug)]
struct BcsDecoder<U>(PhantomData<U>);

impl<U: serde::de::DeserializeOwned> Decoder for BcsDecoder<U> {
    type Item = U;
    type Error = Status;

    fn decode(&mut self, buf: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        let remaining = buf.remaining();
        if remaining == 0 {
            match bcs::from_bytes::<Self::Item>(&[]) {
                Ok(item) => return Ok(Some(item)),
                Err(_) => return Ok(None),
            }
        }
        let bytes = buf.copy_to_bytes(remaining);
        let item: Self::Item =
            bcs::from_bytes(&bytes).map_err(|e| Status::internal(e.to_string()))?;
        Ok(Some(item))
    }
}

#[derive(Debug, Clone)]
struct BcsCodec<T, U>(PhantomData<(T, U)>);

impl<T, U> Default for BcsCodec<T, U> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T, U> Codec for BcsCodec<T, U>
where
    T: Serialize + Send + 'static,
    U: serde::de::DeserializeOwned + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = BcsEncoder<T>;
    type Decoder = BcsDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        BcsEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        BcsDecoder(PhantomData)
    }
}

// ---------------------------------------------------------------------------
// FaucetClient
// ---------------------------------------------------------------------------

/// A gRPC client for the faucet service.
#[derive(Debug, Clone)]
pub struct FaucetClient {
    inner: tonic::client::Grpc<tonic::transport::Channel>,
}

impl FaucetClient {
    /// Connect to a faucet server at the given URL.
    pub async fn connect(url: impl Into<String>) -> Result<Self, tonic::transport::Error> {
        let conn = tonic::transport::Endpoint::new(url.into())?.connect().await?;
        Ok(Self { inner: tonic::client::Grpc::new(conn) })
    }

    /// Request test tokens for the given address.
    pub async fn request_gas(
        &mut self,
        request: GasRequest,
    ) -> Result<tonic::Response<GasResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::unknown(format!("Service was not ready: {}", e))
        })?;
        let codec: BcsCodec<GasRequest, GasResponse> = BcsCodec::default();
        let path =
            tonic::codegen::http::uri::PathAndQuery::from_static("/faucet.Faucet/RequestGas");
        self.inner.unary(tonic::Request::new(request), path, codec).await
    }
}
