// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub trait Merge<T> {
    fn merge(&mut self, source: T, mask: &crate::utils::field::FieldMaskTree);

    fn merge_from(source: T, mask: &crate::utils::field::FieldMaskTree) -> Self
    where
        Self: std::default::Default,
    {
        let mut message = Self::default();
        message.merge(source, mask);
        message
    }
}
