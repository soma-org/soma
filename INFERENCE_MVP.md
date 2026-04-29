# Soma Inference MVP — Phase 1

## Summary

Soma is a Rust L1 (Sui-lineage, Mysticeti consensus, native USDC via EVM bridge, Postgres-indexed GraphQL). Phase 1 ships a permissionless **spot inference marketplace** on top of it: GPU operators register on-chain, publish content-addressed model manifests, and serve OpenAI-compatible HTTP requests; aggregators discover offerings via GraphQL, route user traffic to providers, and settle in batched USDC payments. The chain stays minimal — four object types, nine handlers, no escrow, no slashing, no on-chain arbitration. Reputation is spend-weighted negative-rating; the 2.5% USDC value-fee on every payment is the anti-sybil anchor. An off-chain points system rewards operators (uptime + serving volume × quality) and buyers (spend × diversity) for conversion to token allocations at TGE.

This repo ships the **chain**, the **bridge** (Rust bridge-node + Solidity contract), and the **daemon** (with a Vast.ai example). The aggregator and points service are closed-source, separate repos.

The chain inherits the agent-marketplace refactor that took it from a model/submission ledger to a USDC-settlement ledger: `CoinType::{Soma, Usdc}`, unified `Transfer` / `MergeCoins`, the SellerVault accumulator, geometric step-decay SOMA emissions, validator Secp256k1 bridge keys, and most of the bridge stack are already built. The pivot to inference replaces ask/bid/settlement with provider/manifest/offering/payment on top of that base; the bridge and deploy work carry over essentially unchanged.

---

## Chain (this repo)

### Carry over from the `market` branch (already built — keep)
- `CoinType::{Soma, Usdc}` discriminator on `Coin` objects; `ObjectType::Coin(CoinType)`.
- Unified `Transfer { coins, amounts?, recipients }` + `MergeCoins { coins }`. Both validate single-CoinType per call.
- `WithdrawFromVault` handler — reused as-is for `ProviderVault`.
- Geometric step-decay SOMA emissions (Sui StakeSubsidy model), 1B total supply, validator-only staking.
- `base_fee` (SOMA) gas infrastructure and `value_fee_bps` (USDC) hook on settlement-style handlers.
- Validator Secp256k1 bridge keys in `ValidatorInfo` + genesis.
- `TestClusterBuilder` scaffolding: USDC-genesis support, ECDSA bridge committee (`with_bridge_committee(n)`), supply-conservation harness that excludes USDC from SOMA totals, RPC field-mask `previous_transaction` fix, secondary-index consistency pattern. Rename `with_marketplace_params()` → `with_inference_params()`.

### Strip from the `market` branch
- Delete `types/src/{ask,bid,settlement}.rs`, `authority/src/execution/marketplace.rs`, `indexer-alt/src/handlers/soma_{asks,bids,settlements}.rs`, and the matching `Ask`/`Bid`/`Settlement` variants in `transaction.rs`, `object.rs`, and `system_state` (replace `marketplace_params` with `inference_params`).
- Drop the ask/bid/settlement RPC endpoints (`GetAsk`, `GetBidsForAsk`, `GetOpenAsks`, `GetSettlement`, `GetSettlements`, `SubscribeAsks`, `SubscribeBids`) and their proto definitions; drop the SDK methods that wrap them.
- Drop the `Ask` / `Bid` / `Settlement` GraphQL types and the `soma asks` / `soma bids` / `soma settlements` / `soma seller listen` / `soma buyer listen` CLI subcommands.
- **Do not finish** the in-flight Phase 5 work on `SubscribeAsks`/`SubscribeBids` real streaming or the seller/buyer-listen E2E flows — the inference MVP polls instead of streaming.
- Rename `SellerVault` → `ProviderVault` in `types/src/vault.rs`.

### New object types (`types/src/inference.rs`)
- **`ProviderRegistration`** — one per operator. Fields: `operator` (cold key, owns stake), `operational_key` (optional hot key on the daemon box, rotatable, cannot move stake), `endpoint` (HTTPS URL), `stake` (SOMA shannons in protocol escrow, sybil gate, not slashable), `last_heartbeat_ms`. Status (`active | stale | deregistered`) is derived at query time, not stored.
- **`Manifest`** — immutable content-addressed model spec. Fields: identity (`name`, `hugging_face_id?`, `description?`), architecture (`input/output_modalities`, `tokenizer`, `context_length`, `max_output_length`, `quantization`), capabilities (`supported_sampling_parameters`, `supported_features`), optional `weights_ref`, and `content_hash = sha256(BCS(content_fields))`. Anyone can publish; aggregators curate display-name → ObjectID mappings.
- **`Offering`** — one per `(provider, manifest)`. Fields: `manifest_id`, `price_per_m_input`, `price_per_m_output` (USDC micro-units per million tokens), `active`. No latency/throughput claims — those belong in observation, not state.
- **`Payment`** — one per settlement event. Fields: `buyer`, `provider`, `offering_id`, `amount` (net of fee), `tokens_in`, `tokens_out`, `rating` (default `Positive`), `rating_deadline_ms`. Default Positive; one-shot Negative within window.
- **`ProviderVault`** — per-provider USDC balance accumulator. Lazily created on first payment; collapses high-frequency batched payments into a single balance.

### Nine transaction handlers (`authority/src/execution/inference.rs`)
- `RegisterProvider` — sender becomes operator; stake_coin moved to escrow; creates `ProviderRegistration`.
- `UpdateProvider` — change `endpoint` and/or `operational_key` (operational key change requires cold-key auth).
- `DeregisterProvider` — operator-only; deactivates all offerings, refunds stake, deletes registration. Vault balance survives.
- `Heartbeat` — operator OR operational key; bumps `last_heartbeat_ms`. Most other handlers also bump it implicitly; only needed when otherwise idle.
- `CreateManifest` — any sender; chain re-verifies `content_hash`. Duplicates allowed.
- `CreateOffering` / `UpdateOffering` — operator OR operational key. `(provider, manifest)` unique on active offerings.
- `PayProvider` — any sender (the buyer). Validates active offering, USDC coin ownership, `buyer != provider`, `tokens_in + tokens_out > 0`. Deducts `value_fee_bps × amount` USDC to protocol fund, credits remainder to `ProviderVault`, creates `Payment` with `rating = Positive` and a `rating_deadline_ms`.
- `RateNegative` — sender must equal `payment.buyer`, before deadline; sets rating to `Negative` (idempotent-once).

### Events
One per handler: `ProviderRegistered`, `ProviderUpdated`, `ProviderDeregistered`, `HeartbeatReceived`, `ManifestCreated`, `OfferingCreated`, `OfferingUpdated`, `PaymentSettled` (carries `amount` AND `value_fee` separately), `RatingNegative`.

### `InferenceParameters` in `SystemState`
- `minimum_provider_stake` (SOMA shannons)
- `rating_window_ms` (default 7d)
- `heartbeat_stale_ms` (default 30m)
- `value_fee_bps` (default 250 = 2.5%)

Hardcode initial values for MVP; defer governance tunability.

### Fees
- `base_fee` (SOMA gas) on every handler.
- `PayProvider` additionally levies `value_fee_bps × amount` in USDC to the protocol fund. This is what anchors `Payment.amount` to reality (fabricating payments costs real USDC).

### Indexer (`indexer-alt`)
- New migration drops ask/bid/settlement tables; adds `inf_providers`, `inf_manifests`, `inf_offerings`, `inf_payments`.
- Four handlers (one per table): upsert from the relevant events; `inf_providers` also bumps `last_heartbeat_ms` on any tx whose sender matches an operator/operational key.
- Compute `spend_weighted_negative_ratio` and `reputation_score` at query time from base tables. Defer materialized views and GraphQL subscriptions until hot-path latency demands them.

### GraphQL (`soma-graphql`)
- Types: `Provider`, `Manifest`, `Offering`, `Payment`.
- Hot-path query for aggregators: `offerings(filter: { manifestId, active, providerStatus, providerMinReputation, maxPriceIn, maxPriceOut }, limit, sortBy: COMPOSITE)` — must return < 100ms; aggregators poll on short TTL.

### CLI (`cli`)
```
soma inference provider {register|update|deregister|heartbeat|status}
soma inference manifest  {publish|show|list|find}
soma inference offering  {create|update|list}
soma inference pay  <offering-id> --coin <ref> --amount <usdc> --tokens-in <n> --tokens-out <n>
soma inference rate <payment-id> --negative
soma inference {reputation|earnings|payment show}
```

`manifest publish` reads a JSON file matching the `Manifest` content schema, BCS-encodes the content fields, computes the hash, submits `CreateManifest`.

---

## Bridge (this repo)

Bidirectional USDC bridge between Ethereum and Soma — validator-operated, ECDSA signatures over `keccak256` for cheap Solidity verification, off-chain signature aggregation via gRPC. Tiered thresholds: deposit/withdraw require f+1 (~33%) stake, emergency pause ~5%, emergency unpause 2/3.

**Critical-path rationale.** The aggregator's user flow is Stripe USD → USDC on Base → bridge → Soma. Without the bridge there is no path for USDC to enter the chain at all, so this is on the inference MVP critical path even though no inference handler touches it.

### Already built (keep)
- **On-chain handlers** (gasless system txs except `BridgeWithdraw`): `BridgeDeposit`, `BridgeWithdraw`, `BridgeEmergencyPause`, `BridgeEmergencyUnpause`. Real ECDSA verification in the executor.
- **`BridgeState` in `SystemState`**: `paused`, `next_withdrawal_nonce`, `processed_deposit_nonces`, `bridge_committee`, `total_bridged_usdc`. `BridgeCommittee` carries `members: BTreeMap<SomaAddress, BridgeMember>` + four threshold values.
- **`PendingWithdrawal` object type** — created on `BridgeWithdraw`, observed by bridge nodes for off-chain signing.
- **`bridge-node` Rust crate**: 10 modules, 34 unit tests passing. Subsystems: Ethereum watcher (alloy-based finalized-block poller + `eth_getLogs` event listener with chunking and `-32005` backoff), gRPC signature exchange, Soma checkpoint watcher. Retry with exponential backoff. wiremock-based integration tests on `EthClient` / `EthSyncer`. Crypto cross-verification with sign→ecrecover roundtrips for all action types and **pinned test vectors in `bridge-node/src/types.rs`** intended as Solidity cross-verification anchors — do not change these.
- **Validator key management**: each validator holds BLS (consensus), Ed25519 (network), and Secp256k1 (bridge) keypairs. Bridge key registered in genesis, rotatable via `UpdateValidatorMetadata` at epoch boundaries.
- **Committee sync**: bridge nodes observe new validator set at epoch boundary, sign update, submit to Ethereum contract when quorum reached.
- **6 e2e bridge tests passing**: `BridgeDeposit` minting USDC, nonce replay rejection, `BridgeWithdraw` (burn + `PendingWithdrawal`), pause-blocks-ops / unpause-resumes, insufficient-stake rejection, full deposit→withdraw→deposit round-trip.

### Still to build
- **Ethereum contract (Solidity, `bridge/evm/`)**:
  - `forge init`, OpenZeppelin UUPS proxy + Pausable.
  - `SomaBridgeMessage.sol` — message encoding matching Rust side; cross-verify against the pinned test vectors in `bridge-node/src/types.rs`.
  - `SomaBridgeCommittee.sol` — committee state, `ecrecover`-based signature verification, bitmap stake summation (port of Sui's `BridgeCommittee.sol` pattern).
  - `SomaBridge.sol` — `deposit(somaRecipient, amount)`, `withdraw(signatures, recipient, amount, nonce)`, `emergencyPause(signatures)`, `emergencyUnpause(signatures)`, `updateCommittee(signatures, members, votingPowers)`. Auto-incrementing deposit nonce; emits `Deposited(nonce, sender, somaRecipient, amount)`.
  - Foundry tests + Sepolia deploy script.
- **bridge-node integration tests**:
  - Full deposit round-trip (mock ETH event → sign → gRPC peer exchange → quorum → `BridgeDeposit` tx).
  - Full withdrawal round-trip (`PendingWithdrawal` observed → sign → quorum → `withdraw()` on Ethereum contract).
  - Committee rotation at epoch boundary.
  - Emergency pause / unpause end-to-end.
  - gRPC peer exchange + quorum-based on-chain submission (orchestrator stub).
- **RPC failure → automatic pause**: bridge nodes pause the bridge if Ethereum RPC failures exceed threshold (Hyperliquid October 2024 lesson). Configure with multiple RPC endpoints (Infura, Alchemy, self-hosted) with fallback.

### Out of scope for Phase 1 bridge
Withdrawal dispute window, governance actions (member blocklist, transfer/rate limits), asset price oracles, additional token support beyond USDC, multi-chain. The signature quorum is the security model in v1.

---

## Daemon + Vast.ai (this repo, `daemon/`)

### What it is
A single Rust binary (`soma-daemon`) the provider runs on their GPU box. **Not in the chain's trust perimeter** — providers can write their own implementation if they want.

### Responsibilities
- Maintain on-chain identity: heartbeat, reconcile offerings on startup (create/update to match TOML config), watch own `PaymentSettled` events for earnings display.
- Accept signed inbound HTTP from aggregators and proxy to a local OpenAI-compatible engine (stock vLLM, SGLang, TGI — operator runs the engine separately, daemon just points `upstream` at it).
- Expose Prometheus metrics on `:9090`.

### Inbound HTTP
- Endpoints: `POST /v1/chat/completions` (streaming + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, plus whatever the upstream engine supports. `GET /v1/models` (derived from configured offerings), `GET /health`, `GET /metrics`.
- **Auth**: every request carries `Authorization: SomaSig <address>:<signature>` over `(method, path, body_hash, timestamp)`. Signatures > 60s old rejected. Daemon verifies the signature but does not gate on identity — open by default.
- **503 + `Retry-After`** when upstream unhealthy or daemon draining.

### Anti-leech (local, no chain state)
- Per-address rate limit (default 120/min).
- Chain-derived unpaid-ratio deprioritization: track served-vs-paid per signer; demote/block addresses past a configurable threshold (default 90% unpaid).
- Defer `min_recent_payment_usdc` and explicit allowlists; off by default.

### Keys
- Recommended: keyfile holds the **operational key** (can heartbeat, manage offerings; cannot touch stake or rotate itself).
- Operator generates the operational key offline, sets it via `soma inference provider update --operational-key <addr>` from the cold key, copies the operational keyfile to the daemon host.

### Capacity mode
- **`standalone` only** for MVP — always-on owned hardware; operator drains manually via `soma inference offering update <id> --active false`.
- Defer the `custom` script-hook variant (`on_capacity_available` / `on_preempt` / `on_capacity_lost`) until needed.

### Config (TOML, ~30 lines)
```toml
[provider]
keyfile  = "/etc/soma/operational.key"
endpoint = "https://my-provider.example.com:8443"

[chain]
rpc                         = "https://rpc.soma.network"
graphql                     = "https://gql.soma.network"
heartbeat_interval_seconds  = 600

[engine]
upstream          = "http://127.0.0.1:8000"
health_check_path = "/health"

[capacity]
mode = "standalone"

[[offerings]]
manifest_id            = "0xabc..."
price_per_m_input_usdc = 0.28
price_per_m_output_usdc = 0.40

[auth]
rate_limit_per_address_per_minute = 120
deprioritize_unpaid_ratio         = 0.9

[telemetry]
prometheus_port = 9090
log_level       = "info"
```

### Startup reconciliation
1. Load config, open keyfile, derive signing address.
2. Query chain for matching `ProviderRegistration` (as operator OR operational key); fail with instruction to run `soma inference provider register` if missing.
3. For each `[[offerings]]`: create if absent, update if prices differ.
4. Start HTTP server, heartbeat timer, and earnings poller.

### Daemon CLI
```
soma-daemon start --config <path>
soma-daemon stop
soma-daemon status
```

### Vast.ai example (`daemon/examples/vast/`)
- `rent.sh` — query Vast API for the cheapest instance matching GPU requirements, rent it, deploy daemon + weights, register endpoint.
- `watch.sh` — poll Vast status, drain the daemon on pre-emption, clean up on instance termination.
- `vast.toml` — example config.
- README walking through end-to-end: rent → install → publish manifest → create offering → serve → get paid. Operators on RunPod/Prime Intellect/AWS Spot write analogous scripts.

---

## Deploy (ops, this repo)

Entirely unbuilt. This is a breaking protocol change — fresh testnet, no live migration. All validators come up simultaneously on a coordinated genesis.

### Sepolia (bridge first, gates everything else)
- Deploy `SomaBridge` UUPS proxy to Sepolia after Foundry tests pass.
- Wire the test bridge committee (validator Secp256k1 pubkeys) into the contract via `updateCommittee` from the deployer key.
- Smoke-test deposit/withdraw round-trip against a long-running local Soma node.

### Fresh-testnet genesis
- Bump protocol version in `protocol-config`.
- New `GenesisConfig`: `inference_params` (`minimum_provider_stake`, `rating_window_ms`, `heartbeat_stale_ms`, `value_fee_bps`), bridge committee config (validator ECDSA keys + voting powers), emission decay params, `TOTAL_SUPPLY_SOMA = 1B`.
- **No USDC at genesis** — every USDC enters via the bridge (the aggregator bridges its own float in like any other user).
- SOMA faucet for gas tokens during testnet.

### K8s
- Manifests for: validators (with bridge-node sidecar per validator), indexer-alt, GraphQL, Postgres.
- Health-checks on validator + bridge-node + indexer + GraphQL.
- Multiple Ethereum RPC endpoints configured with fallback per bridge-node.

### Smoke tests (run post-deploy)
1. Bridge USDC in (Ethereum → Soma).
2. Provider: `register` → `manifest publish` → `offering create` → daemon starts heartbeating.
3. Buyer: signs HTTP request → daemon proxies to upstream engine → buyer submits `pay` batch.
4. Buyer: `rate --negative` within window → reputation score shifts.
5. Provider: `withdraw vault` → `bridge withdraw` (Soma → Ethereum) → USDC lands on Sepolia.

---

## API / Aggregator (separate repo, closed-source)

### Role
The OpenAI-compatible front door for end users. Holds Stripe + KYC, owns user accounts and balances, runs routing policy, settles to chain in batches. In Phase 1 there is one (the protocol operator's); the chain has no privileged "aggregator" concept — any address that submits `PayProvider` is one.

### External API
- OpenAI-compatible: `POST /v1/chat/completions` (streaming + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `GET /v1/models`, `GET /v1/models/{id}`.
- Routing extensions via headers/body: `provider` (pin), `route` (lowest_cost | lowest_latency | highest_reputation | balanced), `min_reputation`, `max_price_per_m_tokens`.

### Routing per request
1. Map user model name → one or more acceptable Manifest IDs (aggregator's curated namespace).
2. GraphQL query for active offerings filtered by price caps + reputation + provider status.
3. Score: `w1/effective_price + w2·reputation + w3/observed_p95 + w4·heartbeat_liveness`.
4. Take primary + 2 fallbacks; fail over on timeout / 5xx / malformed stream.
5. Sign outbound request with `SomaSig` and proxy/stream to provider's `endpoint`.
6. Stream response back to user; enqueue settlement metadata.

### Settlement
- Batched: one `PayProvider` per `(aggregator, provider)` per 30–60s window. `amount` = net USDC owed; `tokens_in/out` summed across the window.
- Aggregator's Soma address holds USDC float bridged in via the existing EVM bridge. Top-up flow: Stripe USD → USDC on Base → bridge to Soma.
- `RateNegative` fired when verification verdicts or hard signals (timeouts, malformed streams, ops review) accumulate against a recent payment.

### User-facing payments
1. Account creation; KYC above threshold (~$1K/month).
2. Stripe USD → USDC on Base → bridge to Soma → aggregator's Soma address.
3. Charge user's USD balance per call; settle on chain in batches.
4. USD withdrawals: bridge USDC out, refund via Stripe.

### Legal / entity stack (not engineering)
Cayman Foundation owns the protocol + token. Delaware C-corp operates the aggregator (Stripe, MSB/MTL). Commercial agreement between them; protocol fees flow through `value_fee_bps` automatically.

---

## Points system (separate repo, closed-source)

### What it is
Off-chain scoring service that reads only the indexer (zero chain changes). Hourly batch updates cumulative points per address. Converts to token allocations at TGE.

### Operator points (running the daemon)
```
uptime_points   = active_hours × reputation_multiplier      # 0.5–1.5 from spend_weighted_negative_ratio
serving_points  = total_usdc_earned × quality_multiplier    # 1.2 / 1.0 / 0.7 / 0.3 by neg-rating ratio
diversity_bonus = 1.0 + 0.1 × distinct_manifests_served     # capped at 1.5
early_bird      = 3.0 → 1.0 linear over first 6 months

total = (uptime_points + serving_points) × diversity_bonus × early_bird
anti-gaming cap: total ≤ min(uptime_points × 3, serving_points × 2)
```

### Buyer points (spending through the aggregator)
```
usage_points    = usd_spent
diversity_bonus = 1.0 + 0.1 × distinct_models_used (cap 1.5)
early_adopter   = 3.0 → 1.0 linear over first 6 months

total = usage_points × diversity_bonus × early_adopter
```
KYC above ~$500/month spend for full points; below that, 1× without multipliers.

### Data sources (all from indexer)
- `inf_providers.last_heartbeat_ms` over time → uptime.
- `inf_payments` → volume + buyer spend.
- Joined `inf_offerings.manifest_id` → diversity.
- Aggregator's billing DB → USD-equivalent + KYC tier for buyers.

### Anti-gaming
- The 2.5% `value_fee_bps` is the load-bearing on-chain anchor — fabricating payments to inflate reputation costs real USDC proportional to fake spend.
- Off-chain backstop: wallet-cluster detection (shared funding sources, correlated timing, overlapping IPs) — flagged clusters capped or disqualified.
- Self-dealing detection: buyer earning points from spend to their own provider identity is zero-rated.
- Manual top-100 review pre-airdrop.

### Public leaderboard
Top operators / buyers by points / volume / uptime / diversity. Opt-in display names.
