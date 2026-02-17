// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/network/tonic_tls.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use types::consensus::context::Context;

pub(crate) fn certificate_server_name(context: &Context) -> String {
    "consensus_epoch".to_string()
    // TODO: figure out why this isn't working format!("consensus_epoch_{}", context.committee.epoch())
}
