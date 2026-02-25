// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::all)]

// Include the generated proto definitions
include!("../generated/soma.rpc.rs");

// Include generated field info impls
include!("../generated/soma.rpc.field_info.rs");

// Include generated serde impls
include!("../generated/soma.rpc.serde.rs");
include!("..//generated/soma.rpc.getters.rs");

mod balance_change;
mod checkpoint;
mod effects;
mod epoch;
mod executed_transaction;
mod execution_status;
mod ledger_service;
mod object;
mod signatures;
pub mod target;
mod transaction;
mod transaction_execution_service;

pub use descriptor::FILE_DESCRIPTOR_SET;
mod descriptor {
    /// Byte encoded FILE_DESCRIPTOR_SET.
    pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!("../generated/soma.rpc.fds.bin");

    #[cfg(test)]
    mod tests {
        use super::FILE_DESCRIPTOR_SET;
        use prost::Message as _;

        #[test]
        fn file_descriptor_set_is_valid() {
            prost_types::FileDescriptorSet::decode(FILE_DESCRIPTOR_SET).unwrap();
        }
    }
}

impl AsRef<str> for ErrorReason {
    fn as_ref(&self) -> &str {
        self.as_str_name()
    }
}

impl From<ErrorReason> for String {
    fn from(value: ErrorReason) -> Self {
        value.as_ref().into()
    }
}
