// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::ask::{Ask, AskStatus};
use types::base::SomaAddress;
use types::bid::{Bid, BidStatus};
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::settlement::{SellerRating, Settlement};
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::{AcceptBidArgs, CreateAskArgs, CreateBidArgs, TransactionKind};
use types::vault::SellerVault;

use super::object::check_ownership;
use super::{BPS_DENOMINATOR, FeeCalculator, TransactionExecutor, bps_mul, checked_add, checked_sub};

pub struct MarketplaceExecutor;

impl MarketplaceExecutor {
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Read and deserialize the system state.
    fn read_system_state(
        store: &TemporaryStore,
    ) -> ExecutionResult<(Object, SystemState)> {
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

    /// Serialize system state back and commit to store.
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

    /// Read an object and deserialize it as the given type.
    fn read_typed_object<T: serde::de::DeserializeOwned>(
        store: &TemporaryStore,
        id: &ObjectID,
        expected_type: ObjectType,
        not_found_err: ExecutionFailureStatus,
    ) -> ExecutionResult<(Object, T)> {
        let object = store
            .read_object(id)
            .ok_or(not_found_err.clone())?
            .clone();
        let inner: T = object
            .deserialize_contents(expected_type.clone())
            .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                object_id: *id,
                expected_type,
                actual_type: object.type_().clone(),
            })?;
        Ok((object, inner))
    }

    // -----------------------------------------------------------------------
    // CreateAsk
    // -----------------------------------------------------------------------

    fn execute_create_ask(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: CreateAskArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, state) = Self::read_system_state(store)?;
        let params = &state.marketplace_params();

        // Validate timeout bounds
        if args.timeout_ms < params.min_ask_timeout_ms
            || args.timeout_ms > params.max_ask_timeout_ms
        {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "timeout_ms {} not in [{}, {}]",
                    args.timeout_ms, params.min_ask_timeout_ms, params.max_ask_timeout_ms
                ),
            });
        }

        // Validate price and bid count
        if args.max_price_per_bid == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "max_price_per_bid must be > 0".into(),
            });
        }
        if args.num_bids_wanted == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "num_bids_wanted must be > 0".into(),
            });
        }

        let ask_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let ask = Ask {
            id: ask_id,
            buyer: signer,
            task_digest: args.task_digest,
            max_price_per_bid: args.max_price_per_bid,
            num_bids_wanted: args.num_bids_wanted,
            timeout_ms: args.timeout_ms,
            created_at_ms: state.epoch_start_timestamp_ms(),
            status: AskStatus::Open,
            accepted_bid_count: 0,
        };

        let ask_object = Object::new_marketplace_object(
            ask_id,
            ObjectType::Ask,
            &ask,
            Owner::Shared { initial_shared_version: types::object::OBJECT_START_VERSION },
            tx_digest,
        );
        store.create_object(ask_object);

        // System state was read but not mutated — still need to write it back
        // because it's a shared input that the framework expects to be mutated.
        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // CancelAsk
    // -----------------------------------------------------------------------

    fn execute_cancel_ask(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        ask_id: ObjectID,
    ) -> ExecutionResult<()> {
        let (state_object, state) = Self::read_system_state(store)?;

        let (ask_object, mut ask) = Self::read_typed_object::<Ask>(
            store,
            &ask_id,
            ObjectType::Ask,
            ExecutionFailureStatus::AskNotFound,
        )?;

        // Validate sender is the buyer
        if ask.buyer != signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: ask_id,
                expected_owner: ask.buyer,
                actual_owner: Some(signer),
            });
        }

        // Must be open
        if ask.status != AskStatus::Open {
            return Err(ExecutionFailureStatus::AskNotOpen);
        }

        // Cannot cancel after accepting bids
        if ask.accepted_bid_count > 0 {
            return Err(ExecutionFailureStatus::AskHasAcceptedBids);
        }

        ask.status = AskStatus::Cancelled;

        let mut updated_ask = ask_object;
        updated_ask.update_contents(&ask);
        store.mutate_input_object(updated_ask);

        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // CreateBid
    // -----------------------------------------------------------------------

    fn execute_create_bid(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: CreateBidArgs,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, state) = Self::read_system_state(store)?;

        let (_ask_object, ask) = Self::read_typed_object::<Ask>(
            store,
            &args.ask_id,
            ObjectType::Ask,
            ExecutionFailureStatus::AskNotFound,
        )?;

        // Ask must be open
        if ask.status != AskStatus::Open {
            return Err(ExecutionFailureStatus::AskNotOpen);
        }

        // Check timeout (using epoch_start_timestamp as current time proxy)
        let current_time = state.epoch_start_timestamp_ms();
        if current_time >= ask.created_at_ms + ask.timeout_ms {
            return Err(ExecutionFailureStatus::AskExpired);
        }

        // Validate price
        if args.price == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "bid price must be > 0".into(),
            });
        }
        if args.price > ask.max_price_per_bid {
            return Err(ExecutionFailureStatus::BidPriceTooHigh);
        }

        // Seller cannot bid on own ask
        if signer == ask.buyer {
            return Err(ExecutionFailureStatus::SellerCannotBidOnOwnAsk);
        }

        let bid_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let bid = Bid {
            id: bid_id,
            ask_id: args.ask_id,
            seller: signer,
            price: args.price,
            response_digest: args.response_digest,
            created_at_ms: current_time,
            status: BidStatus::Pending,
        };

        let bid_object = Object::new_marketplace_object(
            bid_id,
            ObjectType::Bid,
            &bid,
            Owner::Shared { initial_shared_version: types::object::OBJECT_START_VERSION },
            tx_digest,
        );
        store.create_object(bid_object);

        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // AcceptBid — the core settlement action
    // -----------------------------------------------------------------------

    fn execute_accept_bid(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: AcceptBidArgs,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::read_system_state(store)?;
        let fee_bps = state.marketplace_params().marketplace_fee_bps;
        let rating_window_ms = state.marketplace_params().rating_window_ms;
        let current_time = state.epoch_start_timestamp_ms();

        // --- Validate ask ---
        let (ask_object, mut ask) = Self::read_typed_object::<Ask>(
            store,
            &args.ask_id,
            ObjectType::Ask,
            ExecutionFailureStatus::AskNotFound,
        )?;

        if ask.buyer != signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: args.ask_id,
                expected_owner: ask.buyer,
                actual_owner: Some(signer),
            });
        }

        if ask.status != AskStatus::Open {
            return Err(ExecutionFailureStatus::AskNotOpen);
        }

        if ask.accepted_bid_count >= ask.num_bids_wanted {
            return Err(ExecutionFailureStatus::AskAlreadyFilled);
        }

        // --- Validate bid ---
        let (bid_object, mut bid) = Self::read_typed_object::<Bid>(
            store,
            &args.bid_id,
            ObjectType::Bid,
            ExecutionFailureStatus::BidNotFound,
        )?;

        if bid.ask_id != args.ask_id {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "bid.ask_id does not match provided ask_id".into(),
            });
        }

        if bid.status != BidStatus::Pending {
            return Err(ExecutionFailureStatus::BidNotPending);
        }

        // --- Validate payment coin ---
        let coin_id = args.payment_coin.0;
        let payment_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?
            .clone();

        check_ownership(&payment_object, signer)?;

        // Must be USDC
        if payment_object.coin_type() != Some(CoinType::Usdc) {
            return Err(ExecutionFailureStatus::WrongCoinTypeForPayment);
        }

        let coin_balance = payment_object
            .as_coin()
            .ok_or(ExecutionFailureStatus::InsufficientCoinBalance)?;

        if coin_balance < bid.price {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance);
        }

        // --- Compute fee and settle ---
        let marketplace_fee = bps_mul(bid.price, fee_bps);
        let seller_amount = checked_sub(bid.price, marketplace_fee)?;

        // Deduct bid.price from buyer's coin
        let remaining_balance = checked_sub(coin_balance, bid.price)?;
        let mut updated_coin = payment_object.clone();
        updated_coin.update_coin_balance(remaining_balance);
        store.mutate_input_object(updated_coin);

        // Credit fee to protocol fund
        state.add_protocol_fund_balance(marketplace_fee)?;

        // --- Load or create seller vault ---
        // Search for existing vault by iterating input objects.
        // The vault is owned by seller, so it would need to be an input.
        // In practice, vaults are created lazily here if not provided.
        // For now, always create a new vault credit object that the seller
        // can merge later — OR find existing vault in store.
        //
        // Actually per the design: SellerVault is an owned object. The AcceptBid
        // tx only has system_state + ask + bid + payment_coin as inputs.
        // The vault is NOT an input to this tx. So we create a new vault or
        // credit needs to work differently.
        //
        // Design decision: AcceptBid creates a new SellerVault object each time.
        // The seller can merge vaults via WithdrawFromVault. This avoids
        // contention on a single vault object and keeps AcceptBid simple.
        // Alternatively, we just create a USDC coin directly to the seller.
        //
        // Per REFACTOR.md: "Load or create seller's SellerVault" — but since
        // the vault is owned by seller, we can't load it in the buyer's tx.
        // Solution: Create a new SellerVault per settlement. Seller merges
        // via WithdrawFromVault.

        let vault_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let vault = SellerVault {
            id: vault_id,
            owner: bid.seller,
            balance: seller_amount,
        };
        let vault_object = Object::new_marketplace_object(
            vault_id,
            ObjectType::SellerVault,
            &vault,
            Owner::AddressOwner(bid.seller),
            tx_digest,
        );
        store.create_object(vault_object);

        // --- Update bid status ---
        bid.status = BidStatus::Accepted;
        let mut updated_bid = bid_object;
        updated_bid.update_contents(&bid);
        store.mutate_input_object(updated_bid);

        // --- Increment ask accepted count ---
        ask.accepted_bid_count += 1;
        if ask.accepted_bid_count >= ask.num_bids_wanted {
            ask.status = AskStatus::Filled;
        }
        let mut updated_ask = ask_object;
        updated_ask.update_contents(&ask);
        store.mutate_input_object(updated_ask);

        // --- Create Settlement ---
        let settlement_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let settlement = Settlement {
            id: settlement_id,
            ask_id: args.ask_id,
            bid_id: args.bid_id,
            buyer: signer,
            seller: bid.seller,
            amount: seller_amount,
            task_digest: ask.task_digest,
            response_digest: bid.response_digest,
            settled_at_ms: current_time,
            seller_rating: SellerRating::Positive,
            rating_deadline_ms: checked_add(current_time, rating_window_ms)?,
        };
        let settlement_object = Object::new_marketplace_object(
            settlement_id,
            ObjectType::Settlement,
            &settlement,
            Owner::Shared { initial_shared_version: types::object::OBJECT_START_VERSION },
            tx_digest,
        );
        store.create_object(settlement_object);

        // --- Commit system state ---
        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // RateSeller (negative only)
    // -----------------------------------------------------------------------

    fn execute_rate_seller(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        settlement_id: ObjectID,
    ) -> ExecutionResult<()> {
        let (state_object, state) = Self::read_system_state(store)?;
        let current_time = state.epoch_start_timestamp_ms();

        let (settlement_object, mut settlement) = Self::read_typed_object::<Settlement>(
            store,
            &settlement_id,
            ObjectType::Settlement,
            ExecutionFailureStatus::SettlementNotFound,
        )?;

        // Only the buyer can rate
        if settlement.buyer != signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: settlement_id,
                expected_owner: settlement.buyer,
                actual_owner: Some(signer),
            });
        }

        // Cannot rate if already negative
        if settlement.seller_rating == SellerRating::Negative {
            return Err(ExecutionFailureStatus::SettlementAlreadyRatedNegative);
        }

        // Must be within deadline
        if current_time >= settlement.rating_deadline_ms {
            return Err(ExecutionFailureStatus::RatingDeadlinePassed);
        }

        settlement.seller_rating = SellerRating::Negative;

        let mut updated_settlement = settlement_object;
        updated_settlement.update_contents(&settlement);
        store.mutate_input_object(updated_settlement);

        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // WithdrawFromVault
    // -----------------------------------------------------------------------

    fn execute_withdraw_from_vault(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        vault_ref: ObjectRef,
        amount: Option<u64>,
        recipient_coin: Option<ObjectRef>,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let (state_object, state) = Self::read_system_state(store)?;

        let vault_id = vault_ref.0;
        let vault_object = store
            .read_object(&vault_id)
            .ok_or(ExecutionFailureStatus::VaultNotFound)?
            .clone();

        let mut vault: SellerVault = vault_object
            .deserialize_contents(ObjectType::SellerVault)
            .ok_or(ExecutionFailureStatus::VaultNotFound)?;

        // Validate ownership
        if vault.owner != signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: vault_id,
                expected_owner: vault.owner,
                actual_owner: Some(signer),
            });
        }

        let withdraw_amount = match amount {
            Some(amt) => {
                if amt > vault.balance {
                    return Err(ExecutionFailureStatus::InsufficientVaultBalance);
                }
                amt
            }
            None => vault.balance,
        };

        if withdraw_amount == 0 {
            return Err(ExecutionFailureStatus::InsufficientVaultBalance);
        }

        vault.balance = checked_sub(vault.balance, withdraw_amount)?;

        // Credit to recipient coin or create new coin
        match recipient_coin {
            Some(coin_ref) => {
                let coin_id = coin_ref.0;
                let coin_obj = store
                    .read_object(&coin_id)
                    .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                        object_id: coin_id,
                    })?
                    .clone();

                check_ownership(&coin_obj, signer)?;

                if coin_obj.coin_type() != Some(CoinType::Usdc) {
                    return Err(ExecutionFailureStatus::WrongCoinTypeForPayment);
                }

                let old_balance = coin_obj
                    .as_coin()
                    .ok_or(ExecutionFailureStatus::InsufficientCoinBalance)?;
                let new_balance = checked_add(old_balance, withdraw_amount)?;

                let mut updated_coin = coin_obj;
                updated_coin.update_coin_balance(new_balance);
                store.mutate_input_object(updated_coin);
            }
            None => {
                // Create new USDC coin
                let coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    CoinType::Usdc,
                    withdraw_amount,
                    Owner::AddressOwner(signer),
                    tx_digest,
                );
                store.create_object(coin);
            }
        }

        // Update or delete vault
        if vault.balance == 0 {
            store.delete_input_object(&vault_id);
        } else {
            let mut updated_vault = vault_object;
            updated_vault.update_contents(&vault);
            store.mutate_input_object(updated_vault);
        }

        Self::commit_system_state(store, state_object, &state)?;

        Ok(())
    }
}

impl TransactionExecutor for MarketplaceExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::CreateAsk(args) => {
                self.execute_create_ask(store, signer, args, tx_digest)
            }
            TransactionKind::CancelAsk { ask_id } => {
                self.execute_cancel_ask(store, signer, ask_id)
            }
            TransactionKind::CreateBid(args) => {
                self.execute_create_bid(store, signer, args, tx_digest)
            }
            TransactionKind::AcceptBid(args) => {
                self.execute_accept_bid(store, signer, args, tx_digest, value_fee)
            }
            TransactionKind::RateSeller { settlement_id } => {
                self.execute_rate_seller(store, signer, settlement_id)
            }
            TransactionKind::WithdrawFromVault { vault, amount, recipient_coin } => {
                self.execute_withdraw_from_vault(
                    store,
                    signer,
                    vault,
                    amount,
                    recipient_coin,
                    tx_digest,
                )
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for MarketplaceExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        // No value fee on marketplace transactions — the marketplace fee is
        // calculated and deducted inside AcceptBid execution, not via the
        // generic fee pipeline. All other marketplace txs pay only base_fee.
        0
    }
}
