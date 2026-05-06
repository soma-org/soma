// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Executor for payment-channel transactions.
//!
//! Phase 1 ops: `OpenChannel`, `Settle`, `RequestClose`,
//! `WithdrawAfterTimeout`, `TopUp`. See `types::channel` for the
//! on-chain `Channel` object layout and the `Voucher` signing
//! scheme. The op semantics + access-control rules mirror Tempo's
//! MPP session `TempoStreamChannel` adapted to Soma:
//!
//! | Op                     | Caller   | Purpose                                       |
//! |------------------------|----------|-----------------------------------------------|
//! | OpenChannel            | anyone\* | Creates channel, signer becomes the payer     |
//! | Settle                 | payee    | Pay delta on a voucher; channel stays alive   |
//! | TopUp                  | payer    | Refill deposit; clears any pending close      |
//! | RequestClose           | payer    | Start grace timer for forced close            |
//! | WithdrawAfterTimeout   | payer    | After grace elapses, return remainder, delete |
//!
//! \* OpenChannel has no pre-existing channel to authorize against; any
//! signer paying the deposit becomes the payer.

use types::SYSTEM_STATE_OBJECT_ID;
use types::balance::BalanceEvent;
use types::base::{SomaAddress, TimestampMs};
use types::channel::{Channel, Voucher};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::intent::{Intent, IntentMessage, IntentScope};
use types::object::{Object, ObjectID};
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::{
    OpenChannelArgs, RequestCloseArgs, SettleArgs, TopUpArgs, TransactionKind,
    WithdrawAfterTimeoutArgs,
};

use super::{TransactionExecutor, checked_add, checked_sub};

pub struct ChannelExecutor;

impl ChannelExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }

    /// Verify a voucher signature against the channel's authorized
    /// signer. Returns Ok(()) iff `voucher_signature` is a valid
    /// `IntentMessage<Voucher>` signed by `authorized_signer`.
    fn verify_voucher_signature(
        channel: &Channel,
        voucher: Voucher,
        signature: &types::crypto::GenericSignature,
    ) -> ExecutionResult<()> {
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        signature
            .verify_authenticator(&intent_msg, channel.authorized_signer)
            .map_err(|e| {
                ExecutionFailureStatus::ChannelInvalidVoucherSignature { reason: e.to_string() }
            })?;
        Ok(())
    }

    /// Read the on-chain Channel by id. Errors with `ObjectNotFound`
    /// if the channel was already closed (deleted) or never opened.
    fn load_channel(
        store: &TemporaryStore,
        channel_id: ObjectID,
    ) -> ExecutionResult<(Object, Channel)> {
        let object = store
            .read_object(&channel_id)
            .ok_or(ExecutionFailureStatus::ObjectNotFound { object_id: channel_id })?
            .clone();
        let channel = object
            .as_channel()
            .ok_or(ExecutionFailureStatus::NotAChannel { object_id: channel_id })?;
        Ok((object, channel))
    }

    /// Read the agreed wall-clock timestamp from the Clock object —
    /// requires the caller's tx to have declared
    /// `SharedInputObject::CLOCK_OBJ_READ`.
    fn read_clock_ts(store: &TemporaryStore) -> ExecutionResult<TimestampMs> {
        store
            .read_clock_timestamp_ms()
            .ok_or(ExecutionFailureStatus::ChannelClockMissing)
    }

    /// Execute `OpenChannel`. Signer becomes the payer; deposit is
    /// debited from the signer's accumulator balance via a
    /// `BalanceEvent::Withdraw`. The reservation pre-pass guarantees
    /// the sender has the funds before this executor runs, so we
    /// emit unconditionally.
    fn execute_open(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: OpenChannelArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Reject zero-deposit channels — they're useless and create state bloat.
        if args.deposit_amount == 0 {
            return Err(ExecutionFailureStatus::ChannelAmountZero.into());
        }
        // Reject self-channels: payer == payee makes no semantic sense
        // (it'd burn gas to round-trip funds through escrow).
        if args.payee == signer {
            return Err(ExecutionFailureStatus::ChannelInvalidInput {
                reason: "payee must differ from payer".to_string(),
            }
            .into());
        }
        // Reject zero-address authorized_signer — no key can sign for
        // it, which makes the channel only closable through
        // RequestClose+timeout (and any Settle attempt fails).
        if args.authorized_signer == SomaAddress::ZERO {
            return Err(ExecutionFailureStatus::ChannelInvalidInput {
                reason: "authorized_signer must be non-zero".to_string(),
            }
            .into());
        }

        // Build the new Channel with payer = signer. The Channel's
        // internal `deposit` field is the canonical record of locked
        // funds; the accumulator no longer holds them.
        let channel = Channel::new(
            signer,
            args.payee,
            args.authorized_signer,
            args.token,
            args.deposit_amount,
        );

        // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(signer, args.token),
            types::effects::object_change::AccumulatorOperation::Split,
            args.deposit_amount,
        );

        // Create the Channel shared object.
        let channel_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let channel_object = Object::new_channel(channel_id, channel, tx_digest);
        store.create_object(channel_object);

        Ok(())
    }

    /// Execute `Settle`. Caller MUST be the payee (else any holder of
    /// a stale voucher could short-pay; see Tempo access-control).
    fn execute_settle(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: SettleArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // 1. Load channel.
        let (channel_object, mut channel) = Self::load_channel(store, args.channel_id)?;

        // 2. Caller-authorization: only the payee can submit Settle.
        if signer != channel.payee {
            return Err(ExecutionFailureStatus::ChannelCallerNotPayee {
                expected: channel.payee,
                actual: signer,
            }
            .into());
        }

        // 3. Verify voucher signature.
        let voucher = Voucher::new(args.channel_id, args.cumulative_amount);
        Self::verify_voucher_signature(&channel, voucher, &args.voucher_signature)?;

        // 4. Cumulative-monotonicity (replay protection).
        if args.cumulative_amount <= channel.settled_amount {
            return Err(ExecutionFailureStatus::ChannelVoucherNotMonotonic {
                cumulative: args.cumulative_amount,
                settled: channel.settled_amount,
            }
            .into());
        }

        // 5. Overspend check: cumulative_amount must not exceed the
        //    total funds ever escrowed (deposit + already-settled).
        let max_cumulative = channel.max_cumulative_amount();
        if args.cumulative_amount > max_cumulative {
            return Err(ExecutionFailureStatus::ChannelOverspend {
                cumulative: args.cumulative_amount,
                available: max_cumulative,
            }
            .into());
        }

        // 6. Compute and apply delta.
        let delta = checked_sub(args.cumulative_amount, channel.settled_amount)?;
        channel.deposit = checked_sub(channel.deposit, delta)?;
        channel.settled_amount = args.cumulative_amount;

        let mut updated_channel_object = channel_object;
        updated_channel_object.set_channel_data(&channel);
        store.mutate_input_object(updated_channel_object);

        // 7. Credit the payee's accumulator with the delta. The
        // Channel object's internal `deposit` field already
        // decremented, so net accumulator supply is conserved
        // (locked-in-channel ↦ accumulator-held).
        if delta > 0 {
            // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
            store.emit_accumulator_event(
                types::effects::object_change::AccumulatorAddress::balance(
                    channel.payee,
                    channel.token,
                ),
                types::effects::object_change::AccumulatorOperation::Merge,
                delta,
            );
        }

        Ok(())
    }

    /// Execute `RequestClose`. Payer-only; reads Clock and stamps
    /// the request time onto the channel.
    fn execute_request_close(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: RequestCloseArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (channel_object, mut channel) = Self::load_channel(store, args.channel_id)?;

        // Caller-authorization: only the payer can request close.
        if signer != channel.payer {
            return Err(ExecutionFailureStatus::ChannelCallerNotPayer {
                expected: channel.payer,
                actual: signer,
            }
            .into());
        }

        // Idempotent re-request: if a close is already requested,
        // overwriting with the current timestamp would extend the
        // payee's grace window. Reject to keep the original timer
        // intact.
        if channel.close_requested_at_ms.is_some() {
            return Err(ExecutionFailureStatus::ChannelCloseAlreadyPending.into());
        }

        let now_ms = Self::read_clock_ts(store)?;
        channel.close_requested_at_ms = Some(now_ms);

        let mut updated_channel_object = channel_object;
        updated_channel_object.set_channel_data(&channel);
        store.mutate_input_object(updated_channel_object);

        Ok(())
    }

    /// Execute `WithdrawAfterTimeout`. Payer-only; requires that
    /// `RequestClose` was called and that the grace period has
    /// elapsed per the current Clock.
    fn execute_withdraw_after_timeout(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: WithdrawAfterTimeoutArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (_channel_object, channel) = Self::load_channel(store, args.channel_id)?;

        if signer != channel.payer {
            return Err(ExecutionFailureStatus::ChannelCallerNotPayer {
                expected: channel.payer,
                actual: signer,
            }
            .into());
        }

        let close_requested_at_ms = channel
            .close_requested_at_ms
            .ok_or(ExecutionFailureStatus::ChannelNoCloseRequest)?;

        // Read protocol grace period from SystemState (declared as
        // read-only shared input by this tx kind).
        let state_object =
            store.read_object(&SYSTEM_STATE_OBJECT_ID).ok_or(ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?;
        let state = SystemState::deserialize(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize SystemState: {}",
                    e
                )))
            })?;
        let grace_ms = state.parameters().channel_grace_period_ms;

        let now_ms = Self::read_clock_ts(store)?;
        let earliest_withdrawable = checked_add(close_requested_at_ms, grace_ms)?;
        if now_ms < earliest_withdrawable {
            return Err(ExecutionFailureStatus::ChannelGraceNotElapsed {
                now_ms,
                earliest_ms: earliest_withdrawable,
            }
            .into());
        }

        // Credit the remainder back to the payer's accumulator and
        // delete the channel object. Mirrors execute_settle's payout
        // path — the channel's internal `deposit` field reaches zero
        // when remainder lands in the accumulator, conserving total
        // supply.
        let remainder = channel.remainder_to_payer();
        if remainder > 0 {
            // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
            store.emit_accumulator_event(
                types::effects::object_change::AccumulatorAddress::balance(
                    channel.payer,
                    channel.token,
                ),
                types::effects::object_change::AccumulatorOperation::Merge,
                remainder,
            );
        }
        store.delete_input_object(&args.channel_id);

        Ok(())
    }

    /// Execute `TopUp`. Payer-only. Adds `args.amount` to the
    /// channel's escrow (debited from the payer's accumulator),
    /// clears any pending close timer so the channel keeps running.
    /// `args.coin_type` must match the channel's token (the
    /// reservation pre-pass is keyed on it before the channel object
    /// is loaded; mismatch is rejected here as defense-in-depth).
    fn execute_top_up(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: TopUpArgs,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Reject zero-amount top-ups: pure state bloat.
        if args.amount == 0 {
            return Err(ExecutionFailureStatus::ChannelAmountZero.into());
        }

        let (channel_object, mut channel) = Self::load_channel(store, args.channel_id)?;

        // Caller-authorization: only the payer can top up.
        if signer != channel.payer {
            return Err(ExecutionFailureStatus::ChannelCallerNotPayer {
                expected: channel.payer,
                actual: signer,
            }
            .into());
        }

        // Coin-type must match — the reservation pre-pass keyed on
        // `args.coin_type`, so any mismatch would corrupt accounting.
        if args.coin_type != channel.token {
            return Err(ExecutionFailureStatus::ChannelCoinTypeMismatch.into());
        }

        // Increase the on-channel deposit and clear any pending
        // close-timer (per `Channel.close_requested_at_ms` doc:
        // "Cleared by `TopUp` so a renewing payer can withdraw
        // their close request"). Net accumulator supply is conserved
        // (payer debited via Split below; deposit grows by the same
        // amount).
        channel.deposit = checked_add(channel.deposit, args.amount)?;
        channel.close_requested_at_ms = None;

        let mut updated_channel_object = channel_object;
        updated_channel_object.set_channel_data(&channel);
        store.mutate_input_object(updated_channel_object);

        // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(channel.payer, channel.token),
            types::effects::object_change::AccumulatorOperation::Split,
            args.amount,
        );

        Ok(())
    }
}

impl TransactionExecutor for ChannelExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        // Flat one unit per channel op for now. Each op touches a
        // single Channel and at most one Coin. Revisit if op shapes
        // become more variable.
        1
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::OpenChannel(args) => self.execute_open(store, signer, args, tx_digest),
            TransactionKind::Settle(args) => self.execute_settle(store, signer, args, tx_digest),
            TransactionKind::RequestClose(args) => {
                self.execute_request_close(store, signer, args, tx_digest)
            }
            TransactionKind::WithdrawAfterTimeout(args) => {
                self.execute_withdraw_after_timeout(store, signer, args, tx_digest)
            }
            TransactionKind::TopUp(args) => self.execute_top_up(store, signer, args, tx_digest),
            _ => Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        }
    }
}

// ===========================================================================
// Unit tests — every op's happy path + every rejection rule.
// ===========================================================================
#[cfg(test)]
mod tests {
    use fastcrypto::ed25519::Ed25519KeyPair;
    use protocol_config::Chain;
    use types::CLOCK_OBJECT_ID;
    use types::CLOCK_OBJECT_SHARED_VERSION;
    use types::SYSTEM_STATE_OBJECT_SHARED_VERSION;
    use types::base::SomaAddress;
    use types::channel::Voucher;
    use types::crypto::{GenericSignature, Signature, get_key_pair};
    use types::digests::TransactionDigest;
    use types::intent::{Intent, IntentMessage, IntentScope};
    use types::object::{CoinType, Object, ObjectID, ObjectType};
    use types::system_state::FeeParameters;
    use types::temporary_store::TemporaryStore;
    use types::transaction::{
        InputObjectKind, InputObjects, ObjectReadResult, ObjectReadResultKind,
    };

    use super::*;

    const GRACE_PERIOD_MS: u64 = 600_000; // 10 minutes — matches protocol default.

    /// Common fixtures for channel-executor tests. Default: `payer`
    /// also signs vouchers (i.e., `authorized_signer == payer`), so
    /// `signer_kp` is the same key as the payer's. Tests that need a
    /// different signer set `signer_kp` explicitly.
    struct Fixture {
        payer: SomaAddress,
        payee: SomaAddress,
        signer_addr: SomaAddress,
        signer_kp: Ed25519KeyPair,
        channel_id: ObjectID,
    }

    impl Fixture {
        fn new() -> Self {
            let (payer, payer_kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
            let (payee, _payee_kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
            Self {
                payer,
                payee,
                signer_addr: payer, // signer keypair below derives this address
                signer_kp: payer_kp,
                channel_id: ObjectID::random(),
            }
        }

        /// Sign a voucher with the channel's authorized signer.
        fn sign_voucher(&self, cumulative_amount: u64) -> GenericSignature {
            let voucher = Voucher::new(self.channel_id, cumulative_amount);
            let intent_msg =
                IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
            Signature::new_secure(&intent_msg, &self.signer_kp).into()
        }

        fn channel_with_deposit(&self, deposit: u64) -> Channel {
            Channel::new(self.payer, self.payee, self.signer_addr, CoinType::Usdc, deposit)
        }
    }

    /// Helper: TemporaryStore preloaded with the inputs a channel-op tx
    /// would have declared. Pass `with_channel: Some(channel)` to
    /// preload the channel. Stage 8: no coin parameter — the deposit
    /// is balance-mode (Withdraw event), so the executor never reads
    /// an owned coin.
    fn make_store(
        clock_ts: u64,
        grace_ms: u64,
        channel: Option<(ObjectID, Channel)>,
    ) -> TemporaryStore {
        let mut inputs: Vec<ObjectReadResult> = Vec::new();

        // SystemState (read-only) carrying the protocol params. Built
        // via `SystemState::create` with empty/zero fixtures — only
        // `parameters.channel_grace_period_ms` matters here.
        let protocol_config = protocol_config::ProtocolConfig::get_for_version(
            protocol_config::ProtocolVersion::MIN,
            Chain::Unknown,
        );
        let mut state = types::system_state::SystemState::create(
            Vec::new(),
            protocol_config.version.as_u64(),
            0,
            &protocol_config,
            0,
            0,
            0,
            0,
            None,
            types::bridge::MarketplaceParameters::default(),
            types::bridge::BridgeCommittee::empty(),
        );
        state.parameters_mut().channel_grace_period_ms = grace_ms;
        let state_obj = types::object::Object::new(
            types::object::ObjectData::new_with_id(
                SYSTEM_STATE_OBJECT_ID,
                ObjectType::SystemState,
                SYSTEM_STATE_OBJECT_SHARED_VERSION,
                bcs::to_bytes(&state).unwrap(),
            ),
            types::object::Owner::Shared {
                initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
            },
            TransactionDigest::default(),
        );
        inputs.push(ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: SYSTEM_STATE_OBJECT_ID,
                initial_shared_version: SYSTEM_STATE_OBJECT_SHARED_VERSION,
                mutable: false,
            },
            ObjectReadResultKind::Object(state_obj),
        ));

        // Clock at given timestamp.
        let clock_obj = Object::new_clock_with_timestamp_for_testing(clock_ts);
        inputs.push(ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: CLOCK_OBJECT_ID,
                initial_shared_version: CLOCK_OBJECT_SHARED_VERSION,
                mutable: false,
            },
            ObjectReadResultKind::Object(clock_obj),
        ));

        // Channel (mutable shared) if requested.
        if let Some((id, channel)) = channel {
            let channel_obj = Object::new_channel_for_testing(id, channel);
            inputs.push(ObjectReadResult::new(
                InputObjectKind::SharedObject {
                    id,
                    initial_shared_version: types::object::OBJECT_START_VERSION,
                    mutable: true,
                },
                ObjectReadResultKind::Object(channel_obj),
            ));
        }

        TemporaryStore::new(
            InputObjects::new(inputs),
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            0,
            Chain::Unknown,
        )
    }

    // ---------------------------------------------------------------
    // OpenChannel tests
    // ---------------------------------------------------------------

    /// Happy path: signer becomes payer, accumulator is debited via
    /// a Withdraw event, Channel object is created with the right
    /// invariants. Stage 8: there is no deposit coin object — the
    /// reservation pre-pass guarantees funds before this executor
    /// runs, so the executor emits the Withdraw unconditionally.
    #[test]
    fn open_channel_happy_path() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);

        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_amount: 100_000,
            },
            TransactionDigest::default(),
        )
        .expect("OpenChannel succeeds");

        // Stage 14c.6: user-tx executors emit AccumulatorWriteV1
        // records; legacy BalanceEvents are gone. The per-cp
        // SettlementScheduler aggregates these and the per-tx apply
        // path drains them via accumulator_events into the CF.
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::AccumulatorOperation;
        let writes = store.accumulator_writes();
        let payer_id = BalanceAccumulator::derive_id(f.payer, CoinType::Usdc);
        let w = writes.get(&payer_id).expect("payer's withdraw must exist");
        assert_eq!(w.operation, AccumulatorOperation::Split);
        assert_eq!(w.value.as_u64(), 100_000);
        assert!(store.balance_events().is_empty());

        // Channel exists with payer/payee/deposit set correctly.
        let channel_obj = store
            .execution_results
            .written_objects
            .values()
            .find(|o| matches!(o.type_(), ObjectType::Channel))
            .expect("channel created");
        let ch = channel_obj.as_channel().unwrap();
        assert_eq!(ch.payer, f.payer);
        assert_eq!(ch.payee, f.payee);
        assert_eq!(ch.authorized_signer, f.signer_addr);
        assert_eq!(ch.deposit, 100_000);
        assert_eq!(ch.settled_amount, 0);
        assert_eq!(ch.close_requested_at_ms, None);
    }

    /// Self-channels (payer == payee) are rejected at the executor.
    /// Burning gas to round-trip your own funds through escrow is
    /// nonsense; reject up front.
    #[test]
    fn open_channel_rejects_self_channel() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payer, // same as signer
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_amount: 1_000,
            },
            TransactionDigest::default(),
        )
        .expect_err("self-channel must be rejected");
    }

    /// Zero authorized_signer makes the channel un-Settle-able and
    /// only closable via the timeout path. Reject at open.
    #[test]
    fn open_channel_rejects_zero_authorized_signer() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: SomaAddress::ZERO,
                token: CoinType::Usdc,
                deposit_amount: 1_000,
            },
            TransactionDigest::default(),
        )
        .expect_err("zero authorized_signer must be rejected");
    }

    /// Zero-deposit channels are rejected — they're useless and create
    /// state bloat. Pre-pass aside, the executor enforces this
    /// directly so a malformed tx can't create dangling channel objects.
    #[test]
    fn open_channel_rejects_zero_deposit() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_amount: 0,
            },
            TransactionDigest::default(),
        )
        .expect_err("zero-deposit channel must be rejected");
        assert!(
            store.balance_events().is_empty(),
            "no Withdraw should leak when the executor rejects",
        );
    }

    // ---------------------------------------------------------------
    // Settle tests — happy path + every rejection rule
    // ---------------------------------------------------------------

    /// Happy path: payee submits a valid voucher; channel deposit
    /// drops by delta, settled_amount advances, payee's accumulator
    /// is credited via a Deposit event (Stage 8 — no coin output).
    #[test]
    fn settle_happy_path() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store =
            make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel.clone())));

        let voucher_sig = f.sign_voucher(300);
        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 300,
                voucher_signature: voucher_sig,
            },
            TransactionDigest::default(),
        )
        .expect("Settle succeeds");

        let updated_channel = store.read_object(&f.channel_id).unwrap().as_channel().unwrap();
        assert_eq!(updated_channel.deposit, 700);
        assert_eq!(updated_channel.settled_amount, 300);

        // Stage 14c.6: payee's accumulator is credited via an
        // AccumulatorWriteV1::Merge of `delta`.
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::AccumulatorOperation;
        let writes = store.accumulator_writes();
        let payee_id = BalanceAccumulator::derive_id(f.payee, CoinType::Usdc);
        let w = writes.get(&payee_id).expect("payee's deposit must exist");
        assert_eq!(w.operation, AccumulatorOperation::Merge);
        assert_eq!(w.value.as_u64(), 300);
        assert!(store.balance_events().is_empty());
    }

    /// Sequential settles update settled_amount monotonically and
    /// pay only the delta each time.
    #[test]
    fn settle_sequential_pays_only_delta() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store =
            make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel.clone())));
        let mut exec = ChannelExecutor::new();

        // First settle at 100.
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: f.sign_voucher(100),
            },
            TransactionDigest::default(),
        )
        .unwrap();
        assert_eq!(store.read_object(&f.channel_id).unwrap().as_channel().unwrap().deposit, 900);

        // Second settle at 250 — delta = 150.
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 250,
                voucher_signature: f.sign_voucher(250),
            },
            TransactionDigest::default(),
        )
        .unwrap();
        let ch = store.read_object(&f.channel_id).unwrap().as_channel().unwrap();
        assert_eq!(ch.deposit, 750);
        assert_eq!(ch.settled_amount, 250);
    }

    /// Replay protection: re-submitting the same cumulative_amount
    /// fails (cumulative must be strictly greater).
    #[test]
    fn settle_rejects_replayed_voucher() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: f.sign_voucher(100),
            },
            TransactionDigest::default(),
        )
        .unwrap();
        // Replay
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: f.sign_voucher(100),
            },
            TransactionDigest::default(),
        )
        .expect_err("replayed voucher must be rejected");
    }

    /// Stale-voucher attack: an old voucher (cumulative below the
    /// already-settled amount) must be rejected.
    #[test]
    fn settle_rejects_stale_voucher() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(1_000);
        channel.settled_amount = 500; // pretend prior settle happened
        channel.deposit = 500;
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 200, // less than settled_amount
                voucher_signature: f.sign_voucher(200),
            },
            TransactionDigest::default(),
        )
        .expect_err("voucher below settled_amount must be rejected");
    }

    /// Overspend: cumulative_amount > deposit + settled_amount.
    #[test]
    fn settle_rejects_overspend() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(100);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 1_000_000, // way more than deposit
                voucher_signature: f.sign_voucher(1_000_000),
            },
            TransactionDigest::default(),
        )
        .expect_err("overspend must be rejected");
    }

    /// Caller-restriction: only the payee can submit Settle.
    #[test]
    fn settle_rejects_non_payee_caller() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();

        // Payer (not payee) tries to submit Settle.
        exec.execute_settle(
            &mut store,
            f.payer,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: f.sign_voucher(100),
            },
            TransactionDigest::default(),
        )
        .expect_err("non-payee caller must be rejected (Sui-spec rule)");
    }

    /// Invalid voucher signature is rejected. We construct a Settle
    /// with a signature signed by a key OTHER than the channel's
    /// authorized_signer.
    #[test]
    fn settle_rejects_invalid_signature() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));

        // Sign with a wrong key.
        let (_, wrong_kp): (_, Ed25519KeyPair) = get_key_pair();
        let voucher = Voucher::new(f.channel_id, 100);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        let bad_sig: GenericSignature = Signature::new_secure(&intent_msg, &wrong_kp).into();

        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: bad_sig,
            },
            TransactionDigest::default(),
        )
        .expect_err("voucher signed by wrong key must be rejected");
    }

    /// Channel-not-found: the channel was deleted (or never existed).
    #[test]
    fn settle_rejects_missing_channel() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);
        let mut exec = ChannelExecutor::new();
        exec.execute_settle(
            &mut store,
            f.payee,
            SettleArgs {
                channel_id: f.channel_id,
                cumulative_amount: 100,
                voucher_signature: f.sign_voucher(100),
            },
            TransactionDigest::default(),
        )
        .expect_err("missing channel must be rejected");
    }

    // ---------------------------------------------------------------
    // RequestClose tests
    // ---------------------------------------------------------------

    #[test]
    fn request_close_happy_path() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store =
            make_store(1_700_000_000_000, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_request_close(
            &mut store,
            f.payer,
            RequestCloseArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect("RequestClose succeeds");
        let ch = store.read_object(&f.channel_id).unwrap().as_channel().unwrap();
        assert_eq!(ch.close_requested_at_ms, Some(1_700_000_000_000));
    }

    #[test]
    fn request_close_rejects_non_payer() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_request_close(
            &mut store,
            f.payee, // not the payer
            RequestCloseArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("non-payer must be rejected");
    }

    /// Re-requesting close while one is already pending fails — keeps
    /// the original timer intact.
    #[test]
    fn request_close_rejects_when_already_pending() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(1_000);
        channel.close_requested_at_ms = Some(50);
        let mut store = make_store(100, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_request_close(
            &mut store,
            f.payer,
            RequestCloseArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("repeat RequestClose must be rejected");
    }

    // ---------------------------------------------------------------
    // WithdrawAfterTimeout tests
    // ---------------------------------------------------------------

    #[test]
    fn withdraw_after_timeout_happy_path() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(700);
        channel.close_requested_at_ms = Some(0);
        // Clock at exactly grace boundary.
        let mut store =
            make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_withdraw_after_timeout(
            &mut store,
            f.payer,
            WithdrawAfterTimeoutArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect("Withdraw at exactly grace boundary succeeds");

        // Channel deleted.
        assert!(
            store.execution_results.deleted_object_ids.contains(&f.channel_id),
            "channel must be deleted"
        );
        // Stage 14c.6: payer's accumulator is credited via
        // AccumulatorWriteV1::Merge of the remainder. The channel's
        // internal `deposit` was 700, so the credit matches and
        // conserves total supply.
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::AccumulatorOperation;
        let writes = store.accumulator_writes();
        let payer_id = BalanceAccumulator::derive_id(f.payer, CoinType::Usdc);
        let w = writes.get(&payer_id).expect("payer's deposit must exist");
        assert_eq!(w.operation, AccumulatorOperation::Merge);
        assert_eq!(w.value.as_u64(), 700);
        assert!(store.balance_events().is_empty());
    }

    #[test]
    fn withdraw_rejects_before_grace_elapsed() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(700);
        channel.close_requested_at_ms = Some(1_000);
        // Clock 1 ms before grace boundary.
        let mut store = make_store(
            1_000 + GRACE_PERIOD_MS - 1,
            GRACE_PERIOD_MS,
            Some((f.channel_id, channel)),
        );
        let mut exec = ChannelExecutor::new();
        exec.execute_withdraw_after_timeout(
            &mut store,
            f.payer,
            WithdrawAfterTimeoutArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("Withdraw before grace must be rejected");
    }

    #[test]
    fn withdraw_rejects_when_no_request_pending() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(700);
        // No close_requested_at_ms set.
        let mut store = make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_withdraw_after_timeout(
            &mut store,
            f.payer,
            WithdrawAfterTimeoutArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("Withdraw without prior RequestClose must be rejected");
    }

    #[test]
    fn withdraw_rejects_non_payer() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(700);
        channel.close_requested_at_ms = Some(0);
        let mut store =
            make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_withdraw_after_timeout(
            &mut store,
            f.payee, // not the payer
            WithdrawAfterTimeoutArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("non-payer must be rejected");
    }

    // ---------------------------------------------------------------
    // TopUp tests
    // ---------------------------------------------------------------

    /// Happy path: payer adds funds; channel deposit grows by the
    /// amount; payer's accumulator is split (debited) by the same.
    #[test]
    fn top_up_happy_path() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payer,
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Usdc, amount: 500 },
            TransactionDigest::default(),
        )
        .expect("TopUp succeeds");

        let updated = store.read_object(&f.channel_id).unwrap().as_channel().unwrap();
        assert_eq!(updated.deposit, 1_500);
        assert_eq!(updated.settled_amount, 0);
        assert_eq!(updated.close_requested_at_ms, None);

        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::AccumulatorOperation;
        let writes = store.accumulator_writes();
        let payer_id = BalanceAccumulator::derive_id(f.payer, CoinType::Usdc);
        let w = writes.get(&payer_id).expect("payer's withdraw must exist");
        assert_eq!(w.operation, AccumulatorOperation::Split);
        assert_eq!(w.value.as_u64(), 500);
    }

    /// TopUp clears any pending close-timer — the renewal-after-
    /// requested-close path. Payer effectively cancels their close.
    #[test]
    fn top_up_clears_close_request() {
        let f = Fixture::new();
        let mut channel = f.channel_with_deposit(1_000);
        channel.close_requested_at_ms = Some(42_000);
        let mut store = make_store(50_000, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payer,
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Usdc, amount: 100 },
            TransactionDigest::default(),
        )
        .expect("TopUp succeeds even when close was pending");

        let updated = store.read_object(&f.channel_id).unwrap().as_channel().unwrap();
        assert_eq!(updated.deposit, 1_100);
        assert!(
            updated.close_requested_at_ms.is_none(),
            "TopUp must clear pending close timer"
        );
    }

    /// Zero-amount TopUp is rejected — pure state bloat.
    #[test]
    fn top_up_rejects_zero_amount() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payer,
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Usdc, amount: 0 },
            TransactionDigest::default(),
        )
        .expect_err("zero-amount TopUp must be rejected");
    }

    /// Caller-restriction: only the payer can submit TopUp.
    #[test]
    fn top_up_rejects_non_payer_caller() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payee, // not the payer
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Usdc, amount: 100 },
            TransactionDigest::default(),
        )
        .expect_err("non-payer must be rejected");
    }

    /// Coin-type mismatch is rejected: the channel is USDC but
    /// caller passed SOMA. The reservation pre-pass would have
    /// reserved the wrong accumulator; refusing here is
    /// defense-in-depth.
    #[test]
    fn top_up_rejects_coin_type_mismatch() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)));
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payer,
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Soma, amount: 100 },
            TransactionDigest::default(),
        )
        .expect_err("coin_type mismatch must be rejected");
    }

    /// Channel-not-found is rejected.
    #[test]
    fn top_up_rejects_missing_channel() {
        let f = Fixture::new();
        let mut store = make_store(0, GRACE_PERIOD_MS, None);
        let mut exec = ChannelExecutor::new();
        exec.execute_top_up(
            &mut store,
            f.payer,
            TopUpArgs { channel_id: f.channel_id, coin_type: CoinType::Usdc, amount: 100 },
            TransactionDigest::default(),
        )
        .expect_err("missing channel must be rejected");
    }
}
