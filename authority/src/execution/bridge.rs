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

use fastcrypto::hash::Keccak256;
use fastcrypto::secp256k1::Secp256k1PublicKey;
use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
use fastcrypto::traits::{RecoverableSignature, ToFromBytes};
use types::SYSTEM_STATE_OBJECT_ID;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::bridge::{BridgeCommittee, PendingWithdrawal};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, Object, ObjectData, ObjectID, ObjectType, Owner, Version};
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::{
    BridgeDepositArgs, BridgeEmergencyPauseArgs, BridgeEmergencyUnpauseArgs, BridgeWithdrawArgs,
    TransactionKind,
};

use super::TransactionExecutor;

pub struct BridgeExecutor;

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

    /// Verify that the signer bitmap represents sufficient stake for the
    /// given threshold, and that each indicated signer produced a valid
    /// ECDSA signature over `message`.
    ///
    /// The bitmap encodes which committee members signed: bit i set means
    /// the i-th member (in BTreeMap iteration order) signed, and the i-th
    /// 65-byte slice of `aggregated_signature` is their recoverable signature.
    ///
    /// For each bit set:
    ///   1. Extract the 65-byte recoverable signature from aggregated_signature
    ///   2. Ecrecover the public key using Keccak256 hash of `message`
    ///   3. Verify the recovered key matches the committee member's registered key
    ///   4. Accumulate the member's voting_power
    ///
    /// Reject if total accumulated stake < threshold.
    fn verify_committee_signatures(
        committee: &BridgeCommittee,
        message: &[u8],
        aggregated_signature: &[u8],
        signer_bitmap: &[u8],
        threshold: u64,
    ) -> ExecutionResult<()> {
        let members: Vec<_> = committee.members.values().collect();
        let mut total_stake: u64 = 0;
        let mut sig_offset: usize = 0;

        for (i, member) in members.iter().enumerate() {
            let byte_index = i / 8;
            let bit_index = i % 8;
            if byte_index < signer_bitmap.len() && (signer_bitmap[byte_index] >> bit_index) & 1 == 1
            {
                // Extract 65-byte recoverable signature for this signer
                if sig_offset + Self::ECDSA_SIG_SIZE > aggregated_signature.len() {
                    return Err(ExecutionFailureStatus::BridgeInsufficientSignatureStake);
                }
                let sig_bytes =
                    &aggregated_signature[sig_offset..sig_offset + Self::ECDSA_SIG_SIZE];
                sig_offset += Self::ECDSA_SIG_SIZE;

                // Parse the recoverable signature
                let recoverable_sig =
                    Secp256k1RecoverableSignature::from_bytes(sig_bytes).map_err(|_| {
                        ExecutionFailureStatus::SomaError(SomaError::from(
                            "Invalid ECDSA recoverable signature".to_string(),
                        ))
                    })?;

                // Ecrecover: recover the public key from the signature + message
                let recovered_pubkey: Secp256k1PublicKey = recoverable_sig
                    .recover_with_hash::<Keccak256>(message)
                    .map_err(|_| {
                        ExecutionFailureStatus::SomaError(SomaError::from(
                            "ECDSA signature recovery failed".to_string(),
                        ))
                    })?;

                // Verify the recovered key matches the committee member's registered key
                let expected_pubkey =
                    Secp256k1PublicKey::from_bytes(&member.ecdsa_pubkey).map_err(|_| {
                        ExecutionFailureStatus::SomaError(SomaError::from(
                            "Invalid committee member public key".to_string(),
                        ))
                    })?;

                if recovered_pubkey != expected_pubkey {
                    return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                        "ECDSA signature does not match committee member".to_string(),
                    )));
                }

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
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;
        let bridge = state.bridge_state_mut();

        // Check not paused
        if bridge.paused {
            return Err(ExecutionFailureStatus::BridgePaused);
        }

        // Check nonce not replayed (mirrors EVM isTransferProcessed + Sui sequence_nums)
        if bridge.processed_deposit_nonces.contains(&args.nonce) {
            return Err(ExecutionFailureStatus::BridgeNonceAlreadyProcessed);
        }

        // Build the message that was signed
        let payload =
            types::bridge::encode_deposit_payload(&args.recipient, args.amount);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::UsdcDeposit,
            args.nonce,
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        // Verify ECDSA signatures meet deposit threshold
        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.aggregated_signature,
            &args.signer_bitmap,
            bridge.bridge_committee.threshold_deposit,
        )?;

        // Stage 12: credit recipient's USDC accumulator. The off-chain
        // bridge has already locked the corresponding ETH-side amount,
        // so this Deposit completes the bridge transfer with no coin
        // object materialized.
        store.emit_balance_event(BalanceEvent::deposit(args.recipient, CoinType::Usdc, args.amount));

        // Record nonce and update total
        bridge.processed_deposit_nonces.insert(args.nonce);
        bridge.total_bridged_usdc = bridge
            .total_bridged_usdc
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
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                "BridgeWithdraw amount must be non-zero".to_string(),
            )));
        }

        // Stage 12: debit sender's USDC accumulator. The reservation
        // pre-pass already verified the sender has the funds; the
        // settlement pipeline applies the delta atomically with the
        // PendingWithdrawal object's creation. Bridge nodes observe
        // the PendingWithdrawal in checkpoints and sign for the
        // Ethereum-side release.
        store.emit_balance_event(BalanceEvent::withdraw(signer, CoinType::Usdc, args.amount));

        // Create PendingWithdrawal — bridge nodes observe this in checkpoints
        // and begin off-chain signing for Ethereum release
        let bridge = state.bridge_state_mut();
        let nonce = bridge.next_withdrawal_nonce;
        bridge.next_withdrawal_nonce = nonce
            .checked_add(1)
            .ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;

        // saturating_sub: total_bridged_usdc is a best-effort tracking counter.
        // USDC can exist on-chain from genesis/testing without a deposit, so this
        // may underflow. The actual balance check is on the coin itself.
        bridge.total_bridged_usdc = bridge.total_bridged_usdc.saturating_sub(args.amount);

        let withdrawal_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let pending = PendingWithdrawal {
            id: withdrawal_id,
            nonce,
            sender: signer,
            recipient_eth_address: args.recipient_eth_address,
            amount: args.amount,
            created_at_ms: state.epoch_start_timestamp_ms(),
        };

        let withdrawal_object = Object::new(
            ObjectData::new_with_id(
                withdrawal_id,
                ObjectType::PendingWithdrawal,
                Version::MIN,
                bcs::to_bytes(&pending).unwrap(),
            ),
            Owner::Immutable, // read-only after creation — bridge nodes observe via checkpoints
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

        // Build the emergency pause message (nonce=0 for emergency ops)
        let payload =
            types::bridge::encode_emergency_payload(types::bridge::EmergencyOpCode::Freeze);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::EmergencyOp,
            0, // emergency ops use nonce=0
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.aggregated_signature,
            &args.signer_bitmap,
            bridge.bridge_committee.threshold_pause,
        )?;

        bridge.paused = true;

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

        // Build the emergency unpause message (nonce=0 for emergency ops)
        let payload =
            types::bridge::encode_emergency_payload(types::bridge::EmergencyOpCode::Unfreeze);
        let message = types::bridge::encode_bridge_message(
            types::bridge::BridgeMessageType::EmergencyOp,
            0, // emergency ops use nonce=0
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );

        Self::verify_committee_signatures(
            &bridge.bridge_committee,
            &message,
            &args.aggregated_signature,
            &args.signer_bitmap,
            bridge.bridge_committee.threshold_unpause,
        )?;

        bridge.paused = false;

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
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}
