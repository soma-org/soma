// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, SomaAddress};

/// An object on the Soma network.
pub struct Object {
    pub object_id: Vec<u8>,
    pub object_version: i64,
    pub serialized_object_bcs: Option<Vec<u8>>,
    // From obj_info
    pub owner_kind: Option<String>,
    pub owner_id: Option<Vec<u8>>,
    pub object_type: Option<String>,
}

#[Object]
impl Object {
    /// The object's ID (hex with 0x prefix).
    async fn object_id(&self) -> SomaAddress {
        SomaAddress(self.object_id.clone())
    }

    /// The object's version.
    async fn version(&self) -> BigInt {
        BigInt(self.object_version)
    }

    /// The kind of owner: Immutable, Address, Object, or Shared.
    async fn owner_kind(&self) -> Option<&str> {
        self.owner_kind.as_deref()
    }

    /// The owner's address or parent object ID.
    async fn owner(&self) -> Option<SomaAddress> {
        self.owner_id.as_ref().map(|id| SomaAddress(id.clone()))
    }

    /// The object's type (module::name).
    async fn object_type(&self) -> Option<&str> {
        self.object_type.as_deref()
    }

    /// BCS-serialized object data. Null if the object was deleted.
    async fn serialized_object_bcs(&self) -> Option<Base64> {
        self.serialized_object_bcs
            .as_ref()
            .map(|b| Base64(b.clone()))
    }
}
