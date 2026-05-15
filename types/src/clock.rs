// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Global wall-clock object updated by the consensus commit prologue.
//!
//! Modeled after Sui's `0x6` Clock. A single shared object holds the
//! validator-agreed `commit_timestamp_ms` from the most recent consensus
//! commit. The prologue executor declares Clock as a mutable shared input
//! and writes the new timestamp; user transactions that need wall-clock
//! time declare Clock as an *immutable* shared input, which the scheduler
//! treats as non-conflicting so multiple readers run in parallel.

use serde::{Deserialize, Serialize};

use crate::base::TimestampMs;
use crate::digests::TransactionDigest;
use crate::object::{Object, ObjectData, ObjectType, Owner, Version};
use crate::{CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION};

/// On-chain Clock object contents. One field, mirroring Sui exactly. The
/// object's identity is fixed at [`CLOCK_OBJECT_ID`] (`0x6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Clock {
    pub timestamp_ms: TimestampMs,
}

impl Clock {
    pub const fn new(timestamp_ms: TimestampMs) -> Self {
        Self { timestamp_ms }
    }
}

impl Object {
    /// Build the genesis Clock object. Created with `Version::new()` so the
    /// genesis execution path rewrites `initial_shared_version` to the
    /// genesis lamport timestamp (= 1 = [`CLOCK_OBJECT_SHARED_VERSION`]).
    pub fn new_genesis_clock() -> Self {
        let clock = Clock { timestamp_ms: 0 };
        let data = ObjectData::new_with_id(
            CLOCK_OBJECT_ID,
            ObjectType::Clock,
            Version::MIN,
            bcs::to_bytes(&clock).expect("Clock serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: Version::new() },
            TransactionDigest::default(),
        )
    }

    /// Test/debug constructor that lets a specific timestamp be set without
    /// going through the prologue. Real chain state should only mutate
    /// Clock through [`Self::set_clock_timestamp_ms`].
    pub fn new_clock_with_timestamp_for_testing(timestamp_ms: TimestampMs) -> Self {
        let clock = Clock { timestamp_ms };
        let data = ObjectData::new_with_id(
            CLOCK_OBJECT_ID,
            ObjectType::Clock,
            CLOCK_OBJECT_SHARED_VERSION,
            bcs::to_bytes(&clock).expect("Clock serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: CLOCK_OBJECT_SHARED_VERSION },
            TransactionDigest::default(),
        )
    }

    /// If this object is the Clock, deserialize and return its contents.
    pub fn as_clock(&self) -> Option<Clock> {
        if *self.data.object_type() == ObjectType::Clock {
            bcs::from_bytes::<Clock>(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Convenience: read just the timestamp, panicking if this is not a Clock.
    /// Use [`Self::as_clock`] if the type isn't already known.
    pub fn clock_timestamp_ms(&self) -> TimestampMs {
        self.as_clock().expect("clock_timestamp_ms called on non-Clock object").timestamp_ms
    }

    /// Update the Clock's timestamp in place. Caller must ensure this is
    /// only invoked from the consensus commit prologue path.
    ///
    /// Mirrors Sui's on-chain `clock::set_timestamp` monotonicity check:
    /// the new timestamp must be `>=` the current one. A backward jump
    /// would silently extend channel grace periods and rewind any other
    /// time-dependent invariants, so we panic rather than absorb it —
    /// the prologue is the sole writer and its inputs are validated by
    /// consensus, so a regression here means a load-bearing invariant
    /// upstream broke.
    pub fn set_clock_timestamp_ms(&mut self, timestamp_ms: TimestampMs) {
        assert_eq!(
            *self.data.object_type(),
            ObjectType::Clock,
            "set_clock_timestamp_ms called on non-Clock object"
        );
        let cur = self.clock_timestamp_ms();
        assert!(
            timestamp_ms >= cur,
            "Clock monotonicity violation: cur={cur} new={timestamp_ms}",
        );
        let clock = Clock { timestamp_ms };
        self.update_contents(&clock);
    }

    /// Test-only: advance the Clock by `tick` milliseconds. Mirrors
    /// `sui::clock::increment_for_testing`. Useful in tests that want to
    /// simulate time passing without going through the full prologue
    /// flow.
    pub fn increment_clock_for_testing(&mut self, tick: TimestampMs) {
        let cur = self.clock_timestamp_ms();
        self.set_clock_timestamp_ms(cur.checked_add(tick).expect("clock tick overflow"));
    }

    /// Test-only: set the Clock to `timestamp_ms`. Asserts monotonicity
    /// (the new timestamp must be greater than or equal to the current
    /// one), matching `sui::clock::set_for_testing`. Use this in tests
    /// instead of [`Self::set_clock_timestamp_ms`] when you want the
    /// same monotonicity contract as the production prologue path.
    pub fn set_clock_for_testing(&mut self, timestamp_ms: TimestampMs) {
        let cur = self.clock_timestamp_ms();
        assert!(
            timestamp_ms >= cur,
            "set_clock_for_testing: timestamp must be monotonic (cur={}, new={})",
            cur,
            timestamp_ms,
        );
        self.set_clock_timestamp_ms(timestamp_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_clock_starts_at_zero() {
        let obj = Object::new_genesis_clock();
        assert_eq!(obj.id(), CLOCK_OBJECT_ID);
        assert_eq!(*obj.type_(), ObjectType::Clock);
        assert!(obj.is_shared(), "Clock must be a shared object");
        let clock = obj.as_clock().expect("genesis clock deserializes");
        assert_eq!(clock.timestamp_ms, 0);
    }

    #[test]
    fn set_clock_timestamp_round_trips() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(0);
        obj.set_clock_timestamp_ms(1_700_000_000_000);
        assert_eq!(obj.clock_timestamp_ms(), 1_700_000_000_000);

        // Subsequent updates overwrite, not append.
        obj.set_clock_timestamp_ms(1_700_000_005_000);
        assert_eq!(obj.clock_timestamp_ms(), 1_700_000_005_000);
        // ID must not change across updates.
        assert_eq!(obj.id(), CLOCK_OBJECT_ID);
    }

    #[test]
    fn as_clock_returns_none_for_non_clock_object() {
        let coin = Object::with_id_owner_for_testing(
            crate::object::ObjectID::random(),
            crate::base::SomaAddress::random(),
        );
        assert!(coin.as_clock().is_none());
    }

    /// Sui-parity: `increment_for_testing` advances by an arbitrary tick.
    #[test]
    fn increment_clock_for_testing_advances() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(100);
        obj.increment_clock_for_testing(50);
        assert_eq!(obj.clock_timestamp_ms(), 150);
        obj.increment_clock_for_testing(0);
        assert_eq!(obj.clock_timestamp_ms(), 150, "zero tick is a valid no-op");
    }

    /// Sui-parity: `set_for_testing` accepts forward jumps and the
    /// current value, mirroring the `>=` assertion in sui::clock.
    #[test]
    fn set_clock_for_testing_accepts_monotonic() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(1_000);
        obj.set_clock_for_testing(1_000); // equal allowed
        assert_eq!(obj.clock_timestamp_ms(), 1_000);
        obj.set_clock_for_testing(5_000); // forward allowed
        assert_eq!(obj.clock_timestamp_ms(), 5_000);
    }

    /// Sui-parity: `set_for_testing` rejects backward jumps.
    #[test]
    #[should_panic(expected = "set_clock_for_testing: timestamp must be monotonic")]
    fn set_clock_for_testing_rejects_backward_jump() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(1_000);
        obj.set_clock_for_testing(999);
    }

    /// Production-path monotonicity: `set_clock_timestamp_ms` (used by
    /// the consensus commit prologue) must reject backward jumps in
    /// non-test builds too. A silent backward jump would rewind
    /// channel grace periods and any other time-dependent invariant.
    #[test]
    #[should_panic(expected = "Clock monotonicity violation")]
    fn set_clock_timestamp_ms_rejects_backward_jump() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(1_000);
        obj.set_clock_timestamp_ms(999);
    }

    /// Production-path equality is allowed (no-op tick).
    #[test]
    fn set_clock_timestamp_ms_accepts_equal_timestamp() {
        let mut obj = Object::new_clock_with_timestamp_for_testing(1_000);
        obj.set_clock_timestamp_ms(1_000);
        assert_eq!(obj.clock_timestamp_ms(), 1_000);
    }
}
