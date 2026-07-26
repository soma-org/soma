# Soma vs Sui: Transaction Differences

Comparison of Soma's transaction definitions against Sui's implementations. Soma is a Rust-native chain (no Move VM), so some Sui patterns don't apply directly.

---

## Coin Transactions

No gaps. Soma's `TransferCoins`, `MergeCoins`, `SplitCoins`, and `TransferObjects` are functionally equivalent to Sui's PTB commands, just expressed as standalone transaction types instead of composable commands within a programmable transaction block.

---

## Validator Management

### AddValidator missing `name`/`description` metadata

Sui's `request_add_validator_candidate` includes `name`, `description`, `image_url`, `project_url`, initial `gas_price`, and initial `commission_rate`. Soma's `AddValidatorArgs` only has keys and network addresses.

Human-readable metadata is useful for explorer UIs and wallet displays. Consider adding at minimum `name: Vec<u8>` and `description: Vec<u8>`.

### No per-validator gas pricing

Sui validators independently set gas prices via `request_set_gas_price`, and the protocol uses a reference gas price derived from stake-weighted median. Soma uses protocol-level fixed fees (`base_fee`, `value_fee_bps`, `write_object_fee`) with no per-validator bidding.

This is a deliberate design choice. Fixed fees are simpler and more predictable for a marketplace chain.

### No ValidatorOperationCap delegation

Sui uses a `ValidatorOperationCap` capability object for validator operations like reporting. This allows validators to delegate operational actions to a separate address (e.g., an operator key) without exposing their staking key.

Soma uses direct sender address checks. Simpler, but means validators must use their primary key for all operations.

### No candidate-to-active two-step validator registration

Sui separates `request_add_validator_candidate` (register) from `request_add_validator` (promote to active). This lets candidates build stake before joining the active set.

Soma's `AddValidator` goes directly to the pending set. Fine for a smaller validator set.

---

## Staking

### No multi-coin staking

Sui has `request_add_stake_mul_coin(Vec<Coin<SUI>>, Option<u64>, validator_address)` that accepts multiple coins in one call. Soma takes a single `coin_ref`. Users can merge first, so this is a convenience gap, not a functional one.

### No fungible staked tokens

Sui supports `convert_to_fungible_staked_sui` and `redeem_fungible_staked_sui` for liquid staking. Soma doesn't have this. Future feature if needed.

---

## Bridge

### No `BridgeCommitteeUpdate` transaction type

Sui rotates the bridge committee on Ethereum at epoch boundaries via committee-signed `updateCommitteeWithSignatures` calls. Soma's bridge-node crate has committee update message encoding (`BridgeMessageType::CommitteeUpdate`) and the Ethereum contract design includes `updateCommittee()`, but there is no explicit on-chain transaction type to trigger this.

Worth verifying: is the Ethereum committee rotation triggered purely by the bridge node observing epoch changes in checkpoints, or does it need an on-chain transaction to authorize the rotation?

### No bridge rate limiting

Sui's `BridgeLimiter` enforces rolling 24h USD-denominated transfer caps per route. Both the EVM contract and Move module enforce limits independently. Soma defers this to a future protocol upgrade per REFACTOR.md.

### No bridge blocklisting

Sui can blocklist compromised validators from bridge signing via `execute_blocklist` (Move) and `updateBlocklistWithSignatures` (EVM). Blocklisted signers' votes don't count toward quorum. Soma defers this.

### No token metadata / price governance

Sui has governance transactions for adding new bridged tokens (`add_tokens_on_sui`, `addTokensWithSignatures`) and updating asset prices (`update_asset_price`). Not needed while Soma is USDC-only.

### No approval + claim two-step process

Sui separates bridge transfers into approval (attaching committee signatures to a record) and claim (actually minting/releasing tokens). This allows the limiter to gate claims independently. Soma's `BridgeDeposit` does verification and minting atomically.

Soma's approach is simpler and fine for v1 without rate limiting. If rate limiting is added later, a two-step process may be needed.

---

## System Transactions

### Simpler ChangeEpoch fee breakdown

Sui's `ChangeEpoch` breaks fees into `storage_charge`, `computation_charge`, `storage_rebate`, and `non_refundable_storage_fee` because Sui has a storage fund mechanism. Soma uses a single `fees` field. Appropriate since Soma doesn't have storage fund economics.

### No system package upgrades in ChangeEpoch

Sui bundles framework upgrades into `ChangeEpoch.system_packages`, allowing Move module upgrades at epoch boundaries. Soma uses `protocol_version` bumps with code changes deployed by validators. Different upgrade mechanism, both valid.

---

## Summary

| Difference | Severity | Action |
|---|---|---|
| AddValidator missing `name`/`description` | Low | Consider adding for explorer UX |
| No per-validator gas pricing | None | Design choice |
| No ValidatorOperationCap delegation | Low | Consider if operator key separation is needed |
| No candidate-to-active two-step | None | Design choice |
| No multi-coin staking | Low | Convenience — users merge first |
| No fungible staked tokens | None | Future feature |
| No BridgeCommitteeUpdate tx type | Medium | Verify epoch-boundary committee rotation flow |
| No bridge rate limiting | None | Deferred per REFACTOR.md |
| No bridge blocklisting | None | Deferred per REFACTOR.md |
| No approval + claim two-step bridge | Low | May need if rate limiting added |
| Simpler ChangeEpoch fees | None | Design choice |
| No system package upgrades in epoch | None | Different upgrade mechanism |
