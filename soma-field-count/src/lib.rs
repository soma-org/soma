// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub use soma_field_count_derive::*;

/// Trait that provides a constant indicating the number of fields in a struct.
pub trait FieldCount {
    const FIELD_COUNT: usize;
}
