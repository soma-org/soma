// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::FromSqlRow;
use diesel::backend::Backend;
use diesel::deserialize;
use diesel::expression::AsExpression;
use diesel::prelude::*;
use diesel::serialize;
use diesel::sql_types::SmallInt;
use soma_field_count::FieldCount;

use crate::schema::kv_objects;
use crate::schema::obj_info;
use crate::schema::obj_info_deletion_reference;
use crate::schema::obj_versions;

#[derive(Insertable, Debug, Clone, FieldCount, Queryable)]
#[diesel(table_name = kv_objects, primary_key(object_id, object_version))]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredObject {
    pub object_id: Vec<u8>,
    pub object_version: i64,
    pub serialized_object: Option<Vec<u8>>,
    pub cp_sequence_number: i64,
}

#[derive(
    Insertable,
    Selectable,
    Debug,
    Clone,
    PartialEq,
    Eq,
    FieldCount,
    Queryable,
    QueryableByName
)]
#[diesel(table_name = obj_versions, primary_key(object_id, object_version))]
pub struct StoredObjVersion {
    pub object_id: Vec<u8>,
    pub object_version: i64,
    pub object_digest: Option<Vec<u8>>,
    pub cp_sequence_number: i64,
}

#[derive(AsExpression, FromSqlRow, Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[diesel(sql_type = SmallInt)]
#[repr(i16)]
pub enum StoredOwnerKind {
    Immutable = 0,
    Address = 1,
    Object = 2,
    Shared = 3,
    /// Stage 14a: system-managed accumulator object (account-balance
    /// or F1 delegation). The accumulator-kind discriminator (Balance
    /// vs Delegation) is recoverable from the object's ObjectType,
    /// so we don't burn another schema column for it.
    Accumulator = 4,
}

// Stage 13i: StoredCoinOwnerKind removed alongside the
// coin_balance_buckets handler.

#[derive(Insertable, Debug, Clone, FieldCount, Queryable)]
#[diesel(table_name = obj_info, primary_key(object_id, cp_sequence_number))]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredObjInfo {
    pub object_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub owner_kind: Option<StoredOwnerKind>,
    pub owner_id: Option<Vec<u8>>,
    pub package: Option<Vec<u8>>,
    pub module: Option<String>,
    pub name: Option<String>,
    pub instantiation: Option<Vec<u8>>,
}

#[derive(Insertable, Debug, Clone, FieldCount, Queryable)]
#[diesel(table_name = obj_info_deletion_reference, primary_key(cp_sequence_number, object_id))]
pub struct StoredObjInfoDeletionReference {
    pub object_id: Vec<u8>,
    pub cp_sequence_number: i64,
}

// Stage 13i: StoredCoinBalanceBucket and StoredCoinBalanceBucketDeletionReference
// removed.

impl<DB: Backend> serialize::ToSql<SmallInt, DB> for StoredOwnerKind
where
    i16: serialize::ToSql<SmallInt, DB>,
{
    fn to_sql<'b>(&'b self, out: &mut serialize::Output<'b, '_, DB>) -> serialize::Result {
        match self {
            StoredOwnerKind::Immutable => 0.to_sql(out),
            StoredOwnerKind::Address => 1.to_sql(out),
            StoredOwnerKind::Object => 2.to_sql(out),
            StoredOwnerKind::Shared => 3.to_sql(out),
            StoredOwnerKind::Accumulator => 4.to_sql(out),
        }
    }
}

impl<DB: Backend> deserialize::FromSql<SmallInt, DB> for StoredOwnerKind
where
    i16: deserialize::FromSql<SmallInt, DB>,
{
    fn from_sql(raw: DB::RawValue<'_>) -> deserialize::Result<Self> {
        Ok(match i16::from_sql(raw)? {
            0 => StoredOwnerKind::Immutable,
            1 => StoredOwnerKind::Address,
            2 => StoredOwnerKind::Object,
            3 => StoredOwnerKind::Shared,
            4 => StoredOwnerKind::Accumulator,
            o => return Err(format!("Unexpected StoredOwnerKind: {o}").into()),
        })
    }
}

// Stage 13i: StoredCoinOwnerKind diesel ToSql/FromSql impls removed.
