// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bridge executor — modeled after Sui's native bridge verification pattern:
//   1. Decode signer_bitmap to identify which committee members signed
//   2. For each signer, ecrecover the public key from the ECDSA signature
//   3. Verify the recovered key matches the committee member's registered key
//   4. Sum voting power of verified signing members
//   5. Verify total voting power >= threshold for the action type
//   6. Execute the bridge action (mint/burn USDC, toggle pause, etc.)

use std::collections::HashSet;

use fastcrypto::hash::Keccak256;
use fastcrypto::secp256k1::Secp256k1PublicKey;
use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
use fastcrypto::traits::{RecoverableSignature, ToFromBytes};
use types::SYSTEM_STATE_OBJECT_ID;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::bridge::{BridgeCommittee, BridgeRecord, PendingWithdrawal, WithdrawalCertificate};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, Object, ObjectData, ObjectID, ObjectType, Owner, Version};
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::{
    BridgeAttachWithdrawalSignaturesArgs, BridgeDepositArgs, BridgeEmergencyPauseArgs,
    BridgeEmergencyUnpauseArgs, BridgeRegisterBridgeKeyArgs,
    BridgeUpdateCommitteeBlocklistArgs, BridgeWithdrawArgs, TransactionKind,
};

use super::TransactionExecutor;

pub struct BridgeExecutor;

/// Hard cap on the number of Eth addresses a single
/// `BridgeUpdateCommitteeBlocklist` payload can carry. Mirrors Sui's gRPC
/// `validate_list_size(255)` defense — keeps the executor's O(N×M) match
/// loop bounded in the worst case.
pub const MAX_BLOCKLIST_ENTRIES_PER_TX: usize = 255;

/// Per-registration cap on the `http_url` field stored on `BridgeState`.
/// Bounds SystemState bloat: an active validator could otherwise pre-register
/// with a multi-MB URL, ballooning every system tx's BCS size.
pub const MAX_BRIDGE_HTTP_URL_LEN: usize = 2048;

impl BridgeExecutor {
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn read_system_state(store: &TemporaryStore) -> ExecutionResult<(Object, SystemState)> {
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();
        let state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;
        Ok((state_object, state))
    }

    fn commit_system_state(
        store: &mut TemporaryStore,
        state_object: Object,
        state: &SystemState,
    ) -> ExecutionResult<()> {
        let state_bytes = bcs::to_bytes(state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize system state: {}",
                e
            )))
        })?;
        let mut updated = state_object;
        updated.data.update_contents(state_bytes);
        store.mutate_input_object(updated);
        Ok(())
    }

    /// Size of a single secp256k1 recoverable signature: 64 bytes (r,s) + 1 byte (v).
    const ECDSA_SIG_SIZE: usize = 65;

    /// Verify a list of independent recoverable ECDSA signatures over
    /// `message`, summing the voting power of unique non-blocklisted
    /// signers and asserting it meets `threshold`.
    ///
    /// Mirrors Sui's `committee::verify_signatures`:
    ///   1. ecrecover the pubkey from each 65-byte sig
    ///   2. dedupe by recovered pubkey (`seen_pubkeys`)
    ///   3. assert pubkey is in the committee map (else `EInvalidSignature`)
    ///   4. accumulate voting power, skipping blocklisted members
    ///   5. assert total ≥ threshold
    ///
    /// The cert is a labeled-pubkey envelope (`BTreeMap<BridgePubkey, BridgeSignature>`).
    /// For each entry: ecrecover the signature, assert the recovered pubkey
    /// matches the labeled key, look up the member, accumulate non-blocklisted
    /// voting power, assert total ≥ threshold. Map keys give cheap dedup
    /// (insertion already enforces unique pubkeys).
    fn verify_committee_signatures(
        committee: &BridgeCommittee,
        message: &[u8],
        signatures: &std::collections::BTreeMap<
            types::bridge::BridgePubkey,
            types::bridge::BridgeSignature,
        >,
        threshold: u64,
    ) -> ExecutionResult<()> {
        let mut total_stake: u64 = 0;

        for (claimed_pubkey, sig) in signatures {
            let recoverable_sig =
                Secp256k1RecoverableSignature::from_bytes(sig.as_slice()).map_err(|_| {
                    ExecutionFailureStatus::SomaError(SomaError::from(
                        "Invalid ECDSA recoverable signature".to_string(),
                    ))
                })?;

            let recovered_pubkey: Secp256k1PublicKey = recoverable_sig
                .recover_with_hash::<Keccak256>(message)
                .map_err(|_| {
                    ExecutionFailureStatus::SomaError(SomaError::from(
                        "ECDSA signature recovery failed".to_string(),
                    ))
                })?;

            // Confirm the labeled pubkey matches what ecrecover produces.
            // Otherwise an attacker could mislabel a sig to attribute it to
            // a higher-stake member.
            if recovered_pubkey.as_ref() != claimed_pubkey.as_ref() {
                return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                    "ECDSA signature does not recover to the labeled pubkey".to_string(),
                )));
            }

            // Membership lookup — Sui's `self.members.contains(&pubkey)` assert.
            let member = committee.members.get(claimed_pubkey).ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(
                    "ECDSA signature recovers to a non-committee pubkey".to_string(),
                ))
            })?;

            // Blocklisted members' sigs are valid-but-zero-weight. Sui parity.
            if !member.is_blocklisted {
                total_stake = total_stake
                    .checked_add(member.voting_power)
                    .ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;
            }
        }

        if total_stake < threshold {
            return Err(ExecutionFailureStatus::BridgeInsufficientSignatureStake);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeDeposit — mint USDC on Soma after Ethereum deposit
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's approve_token_transfer + claim flow, but atomic:
    //   1. Verify bridge not paused
    //   2. Verify nonce not replayed (like Sui's sequence_nums / EVM isTransferProcessed)
    //   3. Verify committee signatures meet threshold_deposit
    //   4. Mint CoinType::Usdc to recipient
    //   5. Record nonce to prevent replay

    fn execute_bridge_deposit(
        &self,
        store: &mut TemporaryStore,
        args: BridgeDepositArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;
        let bridge = state.bridge_state_mut();

        // Check not paused (mirrors Sui's `claim_token_internal` paused assertion).
        if bridge.paused {
            return Err(ExecutionFailureStatus::BridgePaused);
        }

        // L1: zero-amount deposits would consume a nonce + create a
        // BridgeRecord but mint nothing. Sui rejects these at the abi.rs
        // conversion boundary; we reject here for defense in depth.
        if args.amount == 0 {
            return Err(ExecutionFailureStatus::BridgeAmountZero);
        }

        // Replay defense: a single-u64 watermark on BridgeState plus an
        // auto-draining set for out-of-order tolerance. In steady-state
        // in-order delivery, the set is empty and SystemState only holds
        // the watermark. The queryable per-deposit history lives on the
        // separate BridgeRecord objects we create below, *not* in BridgeState.
        if bridge.is_deposit_nonce_processed(args.nonce) {
            return Err(ExecutionFailureStatus::BridgeNonceAlreadyProcessed);
        }

        // Defense in depth: the off-chain bridge nodes only sign
        // USDC transfers today, so a non-USDC token_type slipping in
        // here is either a wire-format bug or a malicious tx — reject
        // before reconstructing the message bytes (which would also
        // fail signature verification, but failing here gives a
        // clearer error).
        if args.token_type != types::bridge::USDC_TOKEN_TYPE {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "BridgeDeposit: unsupported token_type {} (only USDC supported)",
                args.token_type
            ))));
        }

        // Build the canonical V2 message bytes the committee signed over.
        // Must match what the off-chain bridge nodes hashed in
        // bridge-node/src/types.rs::BridgeAction::Deposit::to_message_bytes.
        let payload = types::bridge::encode_deposit_payload(
            &args.sender_eth_address,
            args.target_chain,
            &args.recipient,
            args.token_type,
            args.amount,
            args.timestamp_ms,
        );
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::UsdcDeposit,
            types::bridge::TOKEN_TRANSFER_MESSAGE_VERSION_V2,
            args.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        // Verify ECDSA signatures meet deposit threshold.
        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.signatures,
            bridge.bridge_committee.threshold_deposit,
        )?;

        // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
        // The off-chain bridge has already locked the corresponding
        // ETH-side amount, so this Merge completes the bridge transfer
        // with no coin object materialized; the per-cp settlement
        // applies the aggregated delta.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(
                args.recipient,
                CoinType::Usdc,
            ),
            types::effects::object_change::AccumulatorOperation::Merge,
            args.amount,
        );

        // Create the immutable per-deposit audit record. The replay marker is
        // the bounded set above; this object is the queryable history a user
        // or watchdog can consult to verify "deposit nonce N from chain C was
        // processed at epoch E with this Eth tx hash".
        let record_id = types::bridge::derive_bridge_record_id(
            types::bridge::ETH_BRIDGE_CHAIN_ID,
            types::bridge::BridgeMessageType::UsdcDeposit,
            args.nonce,
        );
        let record = BridgeRecord {
            id: record_id,
            source_chain_id: types::bridge::ETH_BRIDGE_CHAIN_ID,
            nonce: args.nonce,
            eth_tx_hash: args.eth_tx_hash,
            recipient: args.recipient,
            amount: args.amount,
            created_at_epoch: state.epoch(),
            // Eth-side block timestamp from the V2 deposit event. This is
            // the timestamp the committee signed over, *not* Soma's
            // observation time — they may differ.
            created_at_ms: args.timestamp_ms,
        };
        let record_object = Object::new(
            ObjectData::new_with_id(
                record_id,
                ObjectType::BridgeRecord,
                Version::MIN,
                bcs::to_bytes(&record).unwrap(),
            ),
            Owner::Immutable, // permanent audit trail; never mutated.
            tx_digest,
        );
        store.create_object(record_object);

        // Record the nonce in the bounded set (with eviction past the
        // retention cap). Then bump the conservation-invariant supply
        // counter that the bridge watchdog reads via RPC. Overflow at
        // u64::MAX of USDC raw units is impossible at any realistic
        // bridge scale (u64::MAX ≈ 1.8e13 USDC ≈ 18 trillion), but
        // checked_add anyway since this is custody-critical bookkeeping.
        let bridge = state.bridge_state_mut();
        bridge.record_processed_deposit_nonce(args.nonce);
        bridge.total_usdc_supply = bridge
            .total_usdc_supply
            .checked_add(args.amount)
            .ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;
        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeWithdraw — burn USDC, create PendingWithdrawal for bridge nodes
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's send_token: burn the asset on-chain, emit a record that
    // bridge node watchers observe in checkpoints and sign for Ethereum release.

    fn execute_bridge_withdraw(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: BridgeWithdrawArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;

        // Check not paused
        if state.bridge_state().paused {
            return Err(ExecutionFailureStatus::BridgePaused);
        }

        if args.amount == 0 {
            return Err(ExecutionFailureStatus::BridgeAmountZero);
        }

        // Stage 14c.6 (SIP-58 cutover): only AccumulatorWriteV1.
        // The reservation pre-pass verifies the sender has the funds;
        // per-cp settlement applies the aggregated delta atomically
        // with the PendingWithdrawal object's creation. Bridge nodes
        // observe the PendingWithdrawal in checkpoints and sign for
        // the Ethereum-side release.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(signer, CoinType::Usdc),
            types::effects::object_change::AccumulatorOperation::Split,
            args.amount,
        );

        // Create PendingWithdrawal — bridge nodes observe this in checkpoints
        // and begin off-chain signing for Ethereum release
        let bridge = state.bridge_state_mut();
        let nonce = bridge.next_withdrawal_nonce;
        bridge.next_withdrawal_nonce = nonce
            .checked_add(1)
            .ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;

        // Debit the conservation-invariant supply counter. A
        // `checked_sub` failure here is a *real* bug, not just an
        // overflow guard — it means we're trying to burn USDC the
        // chain doesn't think it minted, which implies state-machine
        // corruption. Halting execution is the right call: the
        // accumulator pre-pass should have caught insufficient-funds
        // long before this code runs, and the only way to land here
        // with a stale supply counter is a serialization/migration bug.
        bridge.total_usdc_supply = bridge
            .total_usdc_supply
            .checked_sub(args.amount)
            .ok_or(ExecutionFailureStatus::BridgeSupplyUnderflow)?;

        // Deterministic ID derived from (chain, msg_type, nonce) so any actor
        // off-chain can compute the withdrawal's ObjectID locally and look it
        // up to attach a cert (mirrors Sui's `BridgeMessageKey`).
        let withdrawal_id = types::bridge::derive_bridge_record_id(
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            types::bridge::BridgeMessageType::UsdcWithdraw,
            nonce,
        );
        // V2 token transfer carries `timestamp_ms` in the signed payload.
        // For Soma-side withdrawals we use `epoch_start_timestamp_ms` —
        // deterministic across all validators within the epoch.
        let timestamp_ms = state.epoch_start_timestamp_ms();
        let pending = PendingWithdrawal {
            id: withdrawal_id,
            nonce,
            sender: signer,
            recipient_eth_address: args.recipient_eth_address,
            amount: args.amount,
            created_at_ms: timestamp_ms,
            target_chain: args.target_chain,
            // Cert attached later via `BridgeAttachWithdrawalSignatures`.
            verified_signatures: None,
        };

        // Mutable shared so a follow-up `BridgeAttachWithdrawalSignatures`
        // tx can attach the committee cert. After attachment the cert is
        // permanently on-chain — anyone can fetch and submit to Eth.
        // Pattern matches `Object::new_channel`: `Version::MIN` for the
        // object's own version (gets bumped on first mutation),
        // `OBJECT_START_VERSION` as the predictable shared-version key.
        let withdrawal_object = Object::new(
            ObjectData::new_with_id(
                withdrawal_id,
                ObjectType::PendingWithdrawal,
                Version::MIN,
                bcs::to_bytes(&pending).unwrap(),
            ),
            Owner::Shared {
                initial_shared_version: types::object::OBJECT_START_VERSION,
            },
            tx_digest,
        );
        store.create_object(withdrawal_object);

        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeEmergencyPause — low threshold (~5% stake)
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's executeEmergencyOpWithSignatures (EVM) and
    // execute_system_message with emergency_op type (Move).
    // Pause is intentionally cheap — safety over liveness.

    fn execute_bridge_emergency_pause(
        &self,
        store: &mut TemporaryStore,
        args: BridgeEmergencyPauseArgs,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;
        let bridge = state.bridge_state_mut();

        // H1: refuse pause-when-paused. Sui parity:
        // `bridge.move::execute_emergency_op` aborts with EBridgeAlreadyPaused.
        // Without this, a no-op pause silently consumes a seq num — an
        // attacker with collusion could burn nonces to invalidate
        // pre-signed unpause certs.
        if bridge.paused {
            return Err(ExecutionFailureStatus::BridgeAlreadyPaused);
        }

        // Per-message-type seq num replay defense. Pause and unpause share
        // the EmergencyOp counter — mirrors Sui's `execute_system_message`
        // which dispatches by message_type byte and increments the per-type
        // seq num.
        let expected = bridge.expected_system_message_seq(
            types::bridge::BridgeMessageType::EmergencyOp,
        );
        if args.nonce != expected {
            return Err(ExecutionFailureStatus::BridgeSystemMessageSeqMismatch {
                expected,
                actual: args.nonce,
            });
        }

        let payload =
            types::bridge::encode_emergency_payload(types::bridge::EmergencyOpCode::Freeze);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::EmergencyOp,
            types::bridge::BRIDGE_MESSAGE_VERSION,
            args.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.signatures,
            bridge.bridge_committee.threshold_pause,
        )?;

        // Sigs ok — commit state changes (flip paused, consume seq num).
        bridge.paused = true;
        bridge.consume_system_message_seq(types::bridge::BridgeMessageType::EmergencyOp);

        Self::commit_system_state(store, state_object, &state)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeEmergencyUnpause — high threshold (2/3 stake)
    // -----------------------------------------------------------------------

    fn execute_bridge_emergency_unpause(
        &self,
        store: &mut TemporaryStore,
        args: BridgeEmergencyUnpauseArgs,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;
        let bridge = state.bridge_state_mut();

        // H1: refuse unpause-when-not-paused. Sui parity:
        // `bridge.move::execute_emergency_op` aborts with EBridgeNotPaused.
        if !bridge.paused {
            return Err(ExecutionFailureStatus::BridgeNotPaused);
        }

        // Per-message-type seq num replay defense — shared with pause.
        let expected = bridge.expected_system_message_seq(
            types::bridge::BridgeMessageType::EmergencyOp,
        );
        if args.nonce != expected {
            return Err(ExecutionFailureStatus::BridgeSystemMessageSeqMismatch {
                expected,
                actual: args.nonce,
            });
        }

        let payload =
            types::bridge::encode_emergency_payload(types::bridge::EmergencyOpCode::Unfreeze);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::EmergencyOp,
            types::bridge::BRIDGE_MESSAGE_VERSION,
            args.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.signatures,
            bridge.bridge_committee.threshold_unpause,
        )?;

        bridge.paused = false;
        bridge.consume_system_message_seq(types::bridge::BridgeMessageType::EmergencyOp);

        Self::commit_system_state(store, state_object, &state)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeRegisterBridgeKey — pre-register a validator's bridge keypair
    // so they're picked up at the next committee rotation.
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's `committee::register`:
    //   - Signer must be in the active validator set
    //   - Pubkey must be a valid 33-byte compressed secp256k1 point
    //   - No two registrations may share the same pubkey (rejected to
    //     prevent griefing the rotation step)
    //   - Writes/overwrites the validator's registration (re-registering
    //     with a fresh URL or pubkey is allowed — only the latest survives
    //     to the next rotation)

    fn execute_bridge_register_bridge_key(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: BridgeRegisterBridgeKeyArgs,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;

        // 1. Signer must be an active validator.
        if !state.validators().is_active_validator(signer) {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                "BridgeRegisterBridgeKey: signer is not an active validator".to_string(),
            )));
        }

        // 2. Pubkey is already validated at the type boundary
        // (BridgePubkey only constructs from a valid secp256k1 point).

        // L2: cap http_url length to bound SystemState bloat. An active
        // validator could otherwise pre-register a multi-MB URL string,
        // ballooning every system tx that re-serializes SystemState.
        if args.http_url.len() > MAX_BRIDGE_HTTP_URL_LEN {
            return Err(ExecutionFailureStatus::BridgeUrlTooLong {
                got: args.http_url.len() as u64,
                max: MAX_BRIDGE_HTTP_URL_LEN as u64,
            });
        }

        let bridge = state.bridge_state_mut();

        // 3. Pubkey uniqueness across registrations (Sui's
        //    `check_uniqueness_bridge_keys`). Allow this validator to
        //    overwrite their own registration; reject if the pubkey is
        //    already claimed by a different validator.
        for (existing_addr, existing_reg) in &bridge.bridge_registrations {
            if existing_reg.bridge_pubkey == args.bridge_pubkey && *existing_addr != signer {
                return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                    "BridgeRegisterBridgeKey: pubkey already registered by a different validator".to_string(),
                )));
            }
        }

        bridge.bridge_registrations.insert(
            signer,
            types::bridge::BridgeRegistration {
                bridge_pubkey: args.bridge_pubkey,
                http_url: args.http_url,
            },
        );

        Self::commit_system_state(store, state_object, &state)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeUpdateCommitteeBlocklist — flip is_blocklisted on members.
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's `committee::execute_blocklist`:
    //   - Match members by their derived 20-byte Eth address
    //   - Abort if any address isn't in the committee (no silent ignores)
    //   - Set or clear `is_blocklisted` on each match
    // Uses per-type seq num for replay defense (Stage 3 contract).

    fn execute_bridge_update_committee_blocklist(
        &self,
        store: &mut TemporaryStore,
        args: BridgeUpdateCommitteeBlocklistArgs,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;

        // M1: cap payload size. The match loop is O(N×M) where N is the
        // payload list and M is committee size; without a cap a quorum-
        // signed payload with collusion could DoS the chain. Mirrors Sui's
        // gRPC `validate_list_size(255)`.
        if args.eth_addresses.len() > MAX_BLOCKLIST_ENTRIES_PER_TX {
            return Err(ExecutionFailureStatus::BridgeBlocklistPayloadTooLarge {
                got: args.eth_addresses.len() as u64,
                max: MAX_BLOCKLIST_ENTRIES_PER_TX as u64,
            });
        }

        let bridge = state.bridge_state_mut();

        // Per-type seq replay defense.
        let expected = bridge.expected_system_message_seq(
            types::bridge::BridgeMessageType::UpdateCommitteeBlocklist,
        );
        if args.nonce != expected {
            return Err(ExecutionFailureStatus::BridgeSystemMessageSeqMismatch {
                expected,
                actual: args.nonce,
            });
        }

        // Reconstruct the canonical signed bytes from authoritative on-chain
        // intent (`is_blocklist` and `eth_addresses`) — the executor never
        // verifies sigs against caller-provided message bytes.
        let blocklist_type = if args.is_blocklist {
            types::bridge::BlocklistType::Blocklist
        } else {
            types::bridge::BlocklistType::Unblocklist
        };
        let payload =
            types::bridge::encode_blocklist_payload(blocklist_type, &args.eth_addresses);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::UpdateCommitteeBlocklist,
            types::bridge::BRIDGE_MESSAGE_VERSION,
            args.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.signatures,
            bridge.bridge_committee.threshold_blocklist,
        )?;

        // Apply the flag flip. Sui aborts (`EValidatorBlocklistContainsUnknownKey`)
        // if any payload address doesn't match a committee member; we mirror.
        // Members are keyed by pubkey in the BTreeMap, so we iterate and
        // derive the eth address from each pubkey to look for a match.
        // BridgePubkey is already validated at construction time, so eth
        // address derivation is infallible.
        for target in &args.eth_addresses {
            let mut found = false;
            for (pubkey, member) in bridge.bridge_committee.members.iter_mut() {
                let derived = types::bridge::derive_eth_address(pubkey);
                if &derived == target {
                    member.is_blocklisted = args.is_blocklist;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                    format!(
                        "Blocklist payload contains unknown committee eth address 0x{}",
                        hex::encode(target)
                    ),
                )));
            }
        }

        bridge.consume_system_message_seq(
            types::bridge::BridgeMessageType::UpdateCommitteeBlocklist,
        );

        Self::commit_system_state(store, state_object, &state)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // BridgeAttachWithdrawalSignatures — attach committee cert to a
    // PendingWithdrawal so any actor can submit it to the Eth-side bridge
    // contract to release the locked USDC.
    // -----------------------------------------------------------------------
    //
    // Mirrors Sui's `bridge::approve_token_transfer` for outbound (Sui→Eth)
    // records: the BridgeRecord must already exist (created at withdraw
    // time); attaching sigs is idempotent (no-op if cert is already present);
    // verification uses the per-message-type threshold (here `threshold_withdraw`).

    fn execute_bridge_attach_withdrawal_signatures(
        &self,
        store: &mut TemporaryStore,
        args: BridgeAttachWithdrawalSignaturesArgs,
    ) -> ExecutionResult<()> {
        // Sui parity: `approve_token_transfer` asserts !paused. Mirror that
        // here so a paused bridge cannot have new certs attached, even
        // though attachment itself moves no funds (it just publishes
        // a quorum-signed authorization).
        let (state_object, state) = Self::read_system_state(store)?;
        if state.bridge_state().paused {
            return Err(ExecutionFailureStatus::BridgePaused);
        }
        // Read once for committee + epoch.
        let committee = state.bridge_state().bridge_committee.clone();
        let threshold_withdraw = committee.threshold_withdraw;
        let attached_at_epoch = state.epoch();

        // Look up the PendingWithdrawal by its deterministic ID.
        let withdrawal_id = types::bridge::derive_bridge_record_id(
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            types::bridge::BridgeMessageType::UsdcWithdraw,
            args.nonce,
        );
        let withdrawal_object = store
            .read_object(&withdrawal_id)
            .ok_or(ExecutionFailureStatus::ObjectNotFound {
                object_id: withdrawal_id,
            })?
            .clone();
        let mut pending: PendingWithdrawal =
            bcs::from_bytes(withdrawal_object.as_inner().data.contents()).map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize PendingWithdrawal: {}",
                    e
                )))
            })?;

        // Idempotency: if the cert is already present, the tx is a no-op.
        // Sui's `approve_token_transfer` emits `TokenTransferAlreadyApproved`
        // and returns; we do the same — skip rather than error so resubmits
        // from racing relayers don't fail loudly.
        if pending.verified_signatures.is_some() {
            return Ok(());
        }

        // Reconstruct the canonical V2 message bytes from the on-chain
        // object's fields. Critical: do NOT trust any caller-provided
        // message — only on-chain authoritative state. This prevents an
        // attacker from smuggling a cert for one withdrawal into another's
        // slot.
        let payload = types::bridge::encode_withdraw_payload(
            &pending.sender,
            pending.target_chain,
            &pending.recipient_eth_address,
            types::bridge::USDC_TOKEN_TYPE,
            pending.amount,
            pending.created_at_ms,
        );
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::UsdcWithdraw,
            types::bridge::TOKEN_TRANSFER_MESSAGE_VERSION_V2,
            pending.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        // Verify quorum at withdraw threshold (3334 BPS = f+1).
        Self::verify_committee_signatures(
            &committee,
            &message,
            &args.signatures,
            threshold_withdraw,
        )?;

        // Attach cert and persist the mutated object.
        pending.verified_signatures = Some(WithdrawalCertificate {
            signatures: args.signatures,
            attached_at_epoch,
        });
        let mut updated = withdrawal_object;
        updated.data.update_contents(bcs::to_bytes(&pending).unwrap());
        store.mutate_input_object(updated);

        // SystemState is declared mutable by `is_bridge_tx`; write back unchanged.
        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }
}

impl TransactionExecutor for BridgeExecutor {
    fn fee_units(&self, _store: &TemporaryStore, kind: &TransactionKind) -> u32 {
        match kind {
            // BridgeDeposit / Pause / Unpause are gasless system txs (skipped before
            // fee_units is even called via is_system_tx). BridgeWithdraw is the only
            // user-paid bridge op; charge a small fixed amount.
            TransactionKind::BridgeWithdraw(_) => 2,
            _ => 0,
        }
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::BridgeDeposit(args) => {
                self.execute_bridge_deposit(store, args, tx_digest)
            }
            TransactionKind::BridgeWithdraw(args) => {
                self.execute_bridge_withdraw(store, signer, args, tx_digest)
            }
            TransactionKind::BridgeEmergencyPause(args) => {
                self.execute_bridge_emergency_pause(store, args)
            }
            TransactionKind::BridgeEmergencyUnpause(args) => {
                self.execute_bridge_emergency_unpause(store, args)
            }
            TransactionKind::BridgeAttachWithdrawalSignatures(args) => {
                self.execute_bridge_attach_withdrawal_signatures(store, args)
            }
            TransactionKind::BridgeUpdateCommitteeBlocklist(args) => {
                self.execute_bridge_update_committee_blocklist(store, args)
            }
            TransactionKind::BridgeRegisterBridgeKey(args) => {
                self.execute_bridge_register_bridge_key(store, signer, args)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}
