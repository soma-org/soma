// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use bytes::{Buf, BufMut};
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

#[derive(Debug)]
pub struct BcsEncoder<T>(PhantomData<T>);

impl<T: serde::Serialize> Encoder for BcsEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, buf: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        bcs::serialize_into(&mut buf.writer(), &item).map_err(|e| Status::internal(e.to_string()))
    }
}

#[derive(Debug)]
pub struct BcsDecoder<U>(PhantomData<U>);

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

/// A [`Codec`] that implements `application/grpc+bcs` via the serde library.
#[derive(Debug, Clone)]
pub struct BcsCodec<T, U>(PhantomData<(T, U)>);

impl<T, U> Default for BcsCodec<T, U> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T, U> Codec for BcsCodec<T, U>
where
    T: serde::Serialize + Send + 'static,
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
