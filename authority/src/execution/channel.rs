// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Executor for payment-channel transactions.
//!
//! Phase 1 ops: `OpenChannel`, `Settle`, `RequestClose`,
//! `WithdrawAfterTimeout`. See `types::channel` for the on-chain
//! `Channel` object layout and the `Voucher` signing scheme. The op
//! semantics + access-control rules mirror Tempo's MPP session
//! `TempoStreamChannel` adapted to Soma:
//!
//! | Op                     | Caller   | Purpose                                       |
//! |------------------------|----------|-----------------------------------------------|
//! | OpenChannel            | anyone\* | Creates channel, signer becomes the payer     |
//! | Settle                 | payee    | Pay delta on a voucher; channel stays alive   |
//! | RequestClose           | payer    | Start grace timer for forced close            |
//! | WithdrawAfterTimeout   | payer    | After grace elapses, return remainder, delete |
//!
//! \* OpenChannel has no pre-existing channel to authorize against; any
//! signer paying the deposit becomes the payer.

use types::CLOCK_OBJECT_ID;
use types::SYSTEM_STATE_OBJECT_ID;
use types::base::{SomaAddress, TimestampMs};
use types::channel::{Channel, Voucher};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::intent::{Intent, IntentMessage, IntentScope};
use types::object::{CoinType, Object, ObjectID, Owner};
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::{
    OpenChannelArgs, RequestCloseArgs, SettleArgs, TransactionKind, WithdrawAfterTimeoutArgs,
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
        signature.verify_authenticator(&intent_msg, channel.authorized_signer).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Voucher signature verification failed: {}",
                e
            )))
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
        let channel = object.as_channel().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Object {} is not a Channel",
                channel_id
            )))
        })?;
        Ok((object, channel))
    }

    /// Read the agreed wall-clock timestamp from the Clock object —
    /// requires the caller's tx to have declared
    /// `SharedInputObject::CLOCK_OBJ_READ`.
    fn read_clock_ts(store: &TemporaryStore) -> ExecutionResult<TimestampMs> {
        store.read_clock_timestamp_ms().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Clock object missing from inputs (tx must declare CLOCK_OBJ_READ)".to_string(),
            ))
        })
    }

    /// Execute `OpenChannel`. Signer becomes the payer; deposit is
    /// drawn from `args.deposit_coin`, which must be a USDC coin
    /// owned by the signer.
    fn execute_open(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: OpenChannelArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // 1. Load the deposit coin and verify ownership/type.
        let coin_id = args.deposit_coin.0;
        let deposit_coin = store
            .read_object(&coin_id)
            .ok_or(ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?
            .clone();
        super::object::check_ownership(&deposit_coin, signer)?;

        // Coin type must match the requested channel token.
        if deposit_coin.coin_type() != Some(args.token) {
            return Err(ExecutionFailureStatus::InvalidObjectType {
                object_id: coin_id,
                expected_type: types::object::ObjectType::Coin(args.token),
                actual_type: deposit_coin.type_().clone(),
            }
            .into());
        }
        let coin_balance =
            deposit_coin.as_coin().ok_or(ExecutionFailureStatus::InvalidObjectType {
                object_id: coin_id,
                expected_type: types::object::ObjectType::Coin(args.token),
                actual_type: deposit_coin.type_().clone(),
            })?;

        // Reject zero-deposit channels — they're useless and create state bloat.
        if args.deposit_amount == 0 {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                "Channel deposit must be non-zero".to_string(),
            ))
            .into());
        }

        // 2. Verify the coin can cover the deposit.
        if coin_balance < args.deposit_amount {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        // 3. Build the new Channel with payer = signer.
        let channel = Channel::new(
            signer,
            args.payee,
            args.authorized_signer,
            args.token,
            args.deposit_amount,
        );

        // 4. Deduct the deposit from the source coin (delete if drained,
        //    mutate otherwise).
        let remaining = checked_sub(coin_balance, args.deposit_amount)?;
        if remaining == 0 {
            store.delete_input_object(&coin_id);
        } else {
            let mut updated = deposit_coin.clone();
            updated.update_coin_balance(remaining);
            store.mutate_input_object(updated);
        }

        // 5. Create the Channel shared object.
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
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // 1. Load channel.
        let (channel_object, mut channel) = Self::load_channel(store, args.channel_id)?;

        // 2. Caller-authorization: only the payee can submit Settle.
        if signer != channel.payee {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Settle must be submitted by the channel payee ({}), got {}",
                channel.payee, signer
            )))
            .into());
        }

        // 3. Verify voucher signature.
        let voucher = Voucher::new(args.channel_id, args.cumulative_amount);
        Self::verify_voucher_signature(&channel, voucher, &args.voucher_signature)?;

        // 4. Cumulative-monotonicity (replay protection).
        if args.cumulative_amount <= channel.settled_amount {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Voucher cumulative_amount {} must be > settled_amount {}",
                args.cumulative_amount, channel.settled_amount
            )))
            .into());
        }

        // 5. Overspend check: cumulative_amount must not exceed the
        //    total funds ever escrowed (deposit + already-settled).
        let max_cumulative = channel.max_cumulative_amount();
        if args.cumulative_amount > max_cumulative {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Voucher cumulative_amount {} exceeds available funds {}",
                args.cumulative_amount, max_cumulative
            )))
            .into());
        }

        // 6. Compute and apply delta.
        let delta = checked_sub(args.cumulative_amount, channel.settled_amount)?;
        channel.deposit = checked_sub(channel.deposit, delta)?;
        channel.settled_amount = args.cumulative_amount;

        let mut updated_channel_object = channel_object;
        updated_channel_object.set_channel_data(&channel);
        store.mutate_input_object(updated_channel_object);

        // 7. Mint a new Coin for the payee with the delta.
        let payout = Object::new_coin(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            channel.token,
            delta,
            Owner::AddressOwner(channel.payee),
            tx_digest,
        );
        store.create_object(payout);

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
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "RequestClose must be submitted by the channel payer ({}), got {}",
                channel.payer, signer
            )))
            .into());
        }

        // Idempotent re-request: if a close is already requested,
        // overwriting with the current timestamp would extend the
        // payee's grace window. Reject to keep the original timer
        // intact.
        if channel.close_requested_at_ms.is_some() {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                "RequestClose already pending for this channel".to_string(),
            ))
            .into());
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
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (_channel_object, channel) = Self::load_channel(store, args.channel_id)?;

        if signer != channel.payer {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "WithdrawAfterTimeout must be submitted by the channel payer ({}), got {}",
                channel.payer, signer
            )))
            .into());
        }

        let close_requested_at_ms = channel.close_requested_at_ms.ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "WithdrawAfterTimeout requires a prior RequestClose".to_string(),
            ))
        })?;

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
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Grace period not elapsed: now={}, earliest_withdrawable={}",
                now_ms, earliest_withdrawable
            )))
            .into());
        }

        // Pay the remainder back to the payer and delete the channel.
        let remainder = channel.remainder_to_payer();
        if remainder > 0 {
            let payout = Object::new_coin(
                ObjectID::derive_id(tx_digest, store.next_creation_num()),
                channel.token,
                remainder,
                Owner::AddressOwner(channel.payer),
                tx_digest,
            );
            store.create_object(payout);
        }
        store.delete_input_object(&args.channel_id);

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
    /// preload the channel; pass `with_deposit_coin: Some(...)` to
    /// also preload an owned coin owned by `signer`.
    fn make_store(
        clock_ts: u64,
        grace_ms: u64,
        channel: Option<(ObjectID, Channel)>,
        deposit_coin: Option<(ObjectID, SomaAddress, u64)>,
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

        // Deposit coin (owned) if requested.
        if let Some((id, owner, balance)) = deposit_coin {
            let coin = Object::with_id_owner_coin_for_testing(id, owner, balance);
            let oref = coin.compute_object_reference();
            inputs.push(ObjectReadResult::new(
                InputObjectKind::ImmOrOwnedObject(oref),
                ObjectReadResultKind::Object(coin),
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

    /// Happy path: signer becomes payer, deposit coin is debited,
    /// Channel object is created with the right invariants.
    #[test]
    fn open_channel_happy_path() {
        let f = Fixture::new();
        let coin_id = ObjectID::random();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, Some((coin_id, f.payer, 1_000_000)));
        let coin_ref =
            store.read_object(&coin_id).unwrap().compute_object_reference();

        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 100_000,
            },
            TransactionDigest::default(),
        )
        .expect("OpenChannel succeeds");

        // Coin balance dropped by deposit_amount.
        let coin = store.read_object(&coin_id).expect("coin still present");
        assert_eq!(coin.as_coin().unwrap(), 900_000);

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

    /// Coin drained completely: the coin object is deleted, not mutated.
    #[test]
    fn open_channel_drains_coin_when_balance_equals_deposit() {
        let f = Fixture::new();
        let coin_id = ObjectID::random();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, Some((coin_id, f.payer, 5_000)));
        let coin_ref = store.read_object(&coin_id).unwrap().compute_object_reference();
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 5_000,
            },
            TransactionDigest::default(),
        )
        .expect("OpenChannel drains coin");
        assert!(
            store.execution_results.deleted_object_ids.contains(&coin_id),
            "coin must be deleted when fully drained"
        );
    }

    #[test]
    fn open_channel_rejects_zero_deposit() {
        let f = Fixture::new();
        let coin_id = ObjectID::random();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, Some((coin_id, f.payer, 1_000)));
        let coin_ref = store.read_object(&coin_id).unwrap().compute_object_reference();
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 0,
            },
            TransactionDigest::default(),
        )
        .expect_err("zero-deposit channel must be rejected");
    }

    #[test]
    fn open_channel_rejects_insufficient_balance() {
        let f = Fixture::new();
        let coin_id = ObjectID::random();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, Some((coin_id, f.payer, 100)));
        let coin_ref = store.read_object(&coin_id).unwrap().compute_object_reference();
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 1_000,
            },
            TransactionDigest::default(),
        )
        .expect_err("insufficient balance must be rejected");
    }

    #[test]
    fn open_channel_rejects_non_owner_signer() {
        let f = Fixture::new();
        let other = SomaAddress::random();
        let coin_id = ObjectID::random();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, Some((coin_id, f.payer, 1_000)));
        let coin_ref = store.read_object(&coin_id).unwrap().compute_object_reference();
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            other, // ← not the coin owner
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 100,
            },
            TransactionDigest::default(),
        )
        .expect_err("non-owner of deposit coin must be rejected");
    }

    #[test]
    fn open_channel_rejects_wrong_coin_type() {
        let f = Fixture::new();
        let coin_id = ObjectID::random();
        // Build a SOMA coin (not USDC) and try to open a USDC channel from it.
        let coin = Object::with_id_owner_soma_coin_for_testing(coin_id, f.payer, 1_000);
        let coin_ref = coin.compute_object_reference();
        let mut store = make_store(0, GRACE_PERIOD_MS, None, None);
        // Manually inject the SOMA coin since make_store only handles USDC.
        store.input_objects.insert(coin_id, coin);
        let mut exec = ChannelExecutor::new();
        exec.execute_open(
            &mut store,
            f.payer,
            OpenChannelArgs {
                payee: f.payee,
                authorized_signer: f.signer_addr,
                token: CoinType::Usdc,
                deposit_coin: coin_ref,
                deposit_amount: 100,
            },
            TransactionDigest::default(),
        )
        .expect_err("mismatched coin type must be rejected");
    }

    // ---------------------------------------------------------------
    // Settle tests — happy path + every rejection rule
    // ---------------------------------------------------------------

    /// Happy path: payee submits a valid voucher, channel deposit
    /// drops by delta, settled_amount advances, payee gets a coin.
    #[test]
    fn settle_happy_path() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store =
            make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel.clone())), None);

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

        // Payee received a coin worth 300.
        let payout = store
            .execution_results
            .written_objects
            .values()
            .find(|o| {
                matches!(o.type_(), ObjectType::Coin(_))
                    && o.owner().get_owner_address().ok() == Some(f.payee)
            })
            .expect("payee got a payout coin");
        assert_eq!(payout.as_coin().unwrap(), 300);
    }

    /// Sequential settles update settled_amount monotonically and
    /// pay only the delta each time.
    #[test]
    fn settle_sequential_pays_only_delta() {
        let f = Fixture::new();
        let channel = f.channel_with_deposit(1_000);
        let mut store =
            make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel.clone())), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);

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
        let mut store = make_store(0, GRACE_PERIOD_MS, None, None);
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
            make_store(1_700_000_000_000, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(0, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        let mut store = make_store(100, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
            make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
        // Payer received a coin worth `deposit` (the remainder).
        let payout = store
            .execution_results
            .written_objects
            .values()
            .find(|o| {
                matches!(o.type_(), ObjectType::Coin(_))
                    && o.owner().get_owner_address().ok() == Some(f.payer)
            })
            .expect("payer got remainder coin");
        assert_eq!(payout.as_coin().unwrap(), 700);
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
            None,
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
        let mut store = make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
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
            make_store(GRACE_PERIOD_MS, GRACE_PERIOD_MS, Some((f.channel_id, channel)), None);
        let mut exec = ChannelExecutor::new();
        exec.execute_withdraw_after_timeout(
            &mut store,
            f.payee, // not the payer
            WithdrawAfterTimeoutArgs { channel_id: f.channel_id },
            TransactionDigest::default(),
        )
        .expect_err("non-payer must be rejected");
    }
}
