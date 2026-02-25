// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Re-export SomaTensor from protocol-config.
//!
//! SomaTensor is defined in protocol-config to avoid circular dependencies,
//! since SystemParameters (which contains SomaTensor fields) is defined there.

pub use protocol_config::SomaTensor;
