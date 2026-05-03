// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 13c: this file was once `test_lock_persists_after_insufficient_gas`,
//! a coin-mode-only regression test. With balance-mode gas (Stage 13c),
//! there is no per-tx gas object to lock — InsufficientGas now manifests
//! as a clean tx failure with no lock-conflict aftermath. The original
//! test's failure mode is gone; nothing remains to test here.
