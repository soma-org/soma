// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use serde::de::Deserializer;
use serde::de::Error;
use serde::ser::Error as SerError;
use serde::ser::Serializer;
use serde_with::{Bytes, DeserializeAs, SerializeAs};
use std::fmt::Debug;
use std::marker::PhantomData;
/// Use with serde_as to control serde for human-readable serialization and deserialization
/// `H` : serde_as SerializeAs/DeserializeAs delegation for human readable in/output
/// `R` : serde_as SerializeAs/DeserializeAs delegation for non-human readable in/output
///
/// # Example:
///
/// ```text
/// #[serde_as]
/// #[derive(Deserialize, Serialize)]
/// struct Example(#[serde_as(as = "Readable<DisplayFromStr, _>")] [u8; 20]);
/// ```
///
/// The above example will delegate human-readable serde to `DisplayFromStr`
/// and array tuple (default) for non-human-readable serializer.
pub struct Readable<H, R> {
    human_readable: PhantomData<H>,
    non_human_readable: PhantomData<R>,
}

impl<T: ?Sized, H, R> SerializeAs<T> for Readable<H, R>
where
    H: SerializeAs<T>,
    R: SerializeAs<T>,
{
    fn serialize_as<S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            H::serialize_as(value, serializer)
        } else {
            R::serialize_as(value, serializer)
        }
    }
}

impl<'de, R, H, T> DeserializeAs<'de, T> for Readable<H, R>
where
    H: DeserializeAs<'de, T>,
    R: DeserializeAs<'de, T>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            H::deserialize_as(deserializer)
        } else {
            R::deserialize_as(deserializer)
        }
    }
}

// Serializes a bitmap according to the roaring bitmap on-disk standard.
/// <https://github.com/RoaringBitmap/RoaringFormatSpec>
pub struct SomaBitmap;

impl SerializeAs<roaring::RoaringBitmap> for SomaBitmap {
    fn serialize_as<S>(source: &roaring::RoaringBitmap, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut bytes = vec![];

        source.serialize_into(&mut bytes).map_err(to_custom_ser_error::<S, _>)?;
        Bytes::serialize_as(&bytes, serializer)
    }
}

impl<'de> DeserializeAs<'de, roaring::RoaringBitmap> for SomaBitmap {
    fn deserialize_as<D>(deserializer: D) -> Result<roaring::RoaringBitmap, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = Bytes::deserialize_as(deserializer)?;
        roaring::RoaringBitmap::deserialize_from(&bytes[..]).map_err(to_custom_error::<'de, D, _>)
    }
}

#[inline]
fn to_custom_error<'de, D, E>(e: E) -> D::Error
where
    E: Debug,
    D: Deserializer<'de>,
{
    Error::custom(format!("byte deserialization failed, cause by: {:?}", e))
}

#[inline]
fn to_custom_ser_error<S, E>(e: E) -> S::Error
where
    E: Debug,
    S: Serializer,
{
    S::Error::custom(format!("byte serialization failed, cause by: {:?}", e))
}
