// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 13c: this file's coin-mode equivocation tests
//! (`test_conflicting_owned_transactions_same_coin`,
//! `test_concurrent_conflicting_owned_transactions`) tested object
//! locking on a contested gas/transfer coin. Balance-mode user txs
//! (BalanceTransfer) have no owned-object inputs, so the equivocation
//! semantics they tested don't apply. The analogous balance-mode
//! protections — replay rejection by digest cache, underfunded txs
//! dropped by the reservation pre-pass — are covered in
//! `reconfiguration_tests::test_replay_rejected_across_epoch_boundary`
//! and `balance_transfer_tests::test_balance_transfer_underfunded_dropped_by_prepass`.
