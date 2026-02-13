# Sync Crate — Comprehensive Testing Plan

Testing plan for `sync/src/` achieving high parity with Sui's `crates/sui-network/src/`. Covers file-by-file mapping, attribution requirements, test infrastructure needed, and every test needed for parity plus Soma-specific coverage.

**Sui reference**: `MystenLabs/sui` — `crates/sui-network/src/`
**Soma crate**: `sync/src/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #5 of 7 plans — important for operational resilience but not consensus-critical. State sync failures cause fullnode desync, not chain halts.

**Accuracy**: The claim of 0 tests is **confirmed** — all test modules are commented out. The `test_utils.rs` file exists but contains only commented-out code. The Sui file path reference `crates/sui-network/src/state_sync/tests.rs` may be inaccurate — the verification agent could not confirm a standalone `tests.rs` file in that location; tests may be embedded in `mod.rs` or structured differently in the current Sui codebase.

**Key Concerns**:
1. **`CommitteeFixture` is the critical infrastructure blocker** — this is shared with the consensus testing plan. Without the ability to generate certified checkpoint chains, nearly no state sync test is possible. This should be the first infrastructure investment.
2. **Feature gaps vs Sui are well-documented** — the "Gaps Between Soma & Sui" table (Priority 14) is valuable. Missing features like `PeerScore`, adaptive timeout, and rate limiting are operational risks but not mainnet blockers.
3. **Integration tests (Priority 6) are the highest value** — verifying multi-node checkpoint sync across epoch boundaries is more important than unit-testing PeerHeights data structure methods.
4. **E2E resilience tests already provide partial coverage** — `full_node_tests.rs` (7 tests) and `checkpoint_tests.rs` (2 tests) in e2e-tests provide some sync coverage. The gap is in isolated unit testing of sync internals.
5. **Missing: network partition recovery** — no test for a fullnode recovering after extended network partition, which is a common operational scenario.

**Estimated Effort**: ~10 engineering days as planned, heavily front-loaded with CommitteeFixture infrastructure.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Cross-Reference](#file-by-file-cross-reference)
3. [Attribution Requirements](#attribution-requirements)
4. [Test Infrastructure](#test-infrastructure)
5. [Priority 1: State Sync Unit Tests](#priority-1-state-sync-unit-tests)
6. [Priority 2: PeerHeights Unit Tests](#priority-2-peerheights-unit-tests)
7. [Priority 3: PeerBalancer Unit Tests](#priority-3-peerbalancer-unit-tests)
8. [Priority 4: Discovery Unit Tests](#priority-4-discovery-unit-tests)
9. [Priority 5: P2P Server Tests](#priority-5-p2p-server-tests)
10. [Priority 6: State Sync Integration Tests](#priority-6-state-sync-integration-tests)
11. [Priority 7: Discovery Integration Tests](#priority-7-discovery-integration-tests)
12. [Priority 8: E2E State Sync Resilience Tests (msim)](#priority-8-e2e-state-sync-resilience-tests-msim)
13. [Priority 9: Worker / Archive Sync Tests](#priority-9-worker--archive-sync-tests)
14. [Gaps Between Soma & Sui](#gaps-between-soma--sui)
15. [Implementation Order](#implementation-order)
16. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State
- **0 tests** in the sync crate — test modules are commented out in both `state_sync/mod.rs` and `discovery/mod.rs`
- `test_utils.rs` exists but contains only commented-out helper code
- All current sync testing is indirect via E2E tests in `e2e-tests/` (fullnode sync, checkpoint tests)

### Target State
- **~50-60 unit/integration tests** matching Sui's state_sync (12 tests) + discovery (10 tests) coverage, plus Soma-specific tests
- Functional `test_utils.rs` with helper infrastructure
- `CommitteeFixture` equivalent for generating certified checkpoint chains
- 1+ msim resilience test for degraded peer scenarios

### Test Count Summary

| Category | Sui Tests | Soma Existing | Soma Target | Gap |
|----------|-----------|---------------|-------------|-----|
| State sync server handlers | 2 | 0 | 2 | 2 |
| State sync integration (multi-node) | 5 | 0 | 5 | 5 |
| State sync archive | 1 | 0 | 1 | 1 |
| State sync peer scoring | 4 | 0 | 0* | N/A |
| PeerHeights unit tests | 0 (inline) | 0 | 8 | 8 |
| PeerBalancer unit tests | 1 | 0 | 3 | 3 |
| Discovery server | 1 | 0 | 1 | 1 |
| Discovery unit tests | 5 | 0 | 5 | 5 |
| Discovery integration (multi-node) | 4 | 0 | 3 | 3 |
| P2P builder tests | 0 | 0 | 2 | 2 |
| E2E resilience (msim) | 1 | 0 | 1 | 1 |
| Soma-specific tests | N/A | 0 | ~10 | 10 |
| **Total** | **~23** | **0** | **~45** | **~45** |

\* Sui's `PeerScore` (throughput tracking) does not exist in Soma — peer scoring is a feature gap.

---

## File-by-File Cross-Reference

### Legend
- **Heavy** = Direct port/fork, needs full attribution
- **Moderate** = Significant shared patterns, needs attribution
- **Soma-only** = Original Soma code, no attribution needed

### Sync Crate Files

| Soma File | Sui File | Derivation | Inline Tests | Notes |
|-----------|----------|------------|--------------|-------|
| `sync/src/lib.rs` | `crates/sui-network/src/lib.rs` | Moderate | 0 | Module exports |
| `sync/src/builder.rs` | `crates/sui-network/src/state_sync/builder.rs` + `crates/sui-network/src/discovery/builder.rs` | Heavy | 0 | Combined P2pBuilder for discovery + state sync |
| `sync/src/server.rs` | `crates/sui-network/src/state_sync/server.rs` + `crates/sui-network/src/discovery/server.rs` | Heavy | 0 | Combined P2pService for all P2P RPCs |
| `sync/src/state_sync/mod.rs` | `crates/sui-network/src/state_sync/mod.rs` | Heavy | 0 | PeerHeights, PeerBalancer, StateSyncEventLoop, all sync logic |
| `sync/src/state_sync/worker.rs` | `crates/sui-network/src/state_sync/worker.rs` | Heavy | 0 | Archive-based checkpoint sync worker |
| `sync/src/discovery/mod.rs` | `crates/sui-network/src/discovery/mod.rs` | Heavy | 0 | DiscoveryEventLoop, update_known_peers, peer gossip |
| `sync/src/test_utils.rs` | `crates/sui-network/src/utils.rs` | Moderate | 0 | Test-only channel manager setup (currently commented out) |
| `sync/src/proto/p2p.P2p.rs` | Generated from `crates/sui-network/build.rs` | Heavy | 0 | Auto-generated tonic stubs |
| `sync/build.rs` | `crates/sui-network/build.rs` | Heavy | 0 | Proto generation with BcsCodec |

### Supporting Type Files (in `types/` crate)

| Soma File | Sui File | Derivation | Notes |
|-----------|----------|------------|-------|
| `types/src/sync/mod.rs` | `crates/sui-network/src/discovery/mod.rs` (NodeInfo types) + anemo types | Heavy | P2P request/response types, NodeInfo, PeerEvent |
| `types/src/sync/active_peers.rs` | anemo `NetworkRef` internals | Moderate | ActivePeers, PeerState with broadcast events |
| `types/src/sync/channel_manager.rs` | Not direct — Sui uses anemo natively | Moderate | gRPC/tonic channel management (Soma-specific abstraction replacing anemo) |
| `types/src/storage/shared_in_memory_store.rs` | `crates/sui-types/src/storage/shared_in_memory_store.rs` | Heavy | Mock store for testing |
| `types/src/storage/write_store.rs` | `crates/sui-types/src/storage/write_store.rs` | Heavy | WriteStore trait |
| `types/src/storage/read_store.rs` | `crates/sui-types/src/storage/read_store.rs` | Heavy | ReadStore trait |
| `types/src/config/p2p_config.rs` | `crates/sui-config/src/p2p.rs` | Heavy | P2pConfig, DiscoveryConfig, StateSyncConfig |
| `types/src/config/state_sync_config.rs` | `crates/sui-config/src/p2p.rs` (StateSyncConfig section) | Heavy | StateSyncConfig |

---

## Attribution Requirements

All files below need the following header added (if not already present):

```
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-network/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

### Files Requiring Attribution (Heavy Derivation — 9 files)

**Sync Crate (7):**
- `sync/src/lib.rs`
- `sync/src/builder.rs`
- `sync/src/server.rs`
- `sync/src/state_sync/mod.rs`
- `sync/src/state_sync/worker.rs`
- `sync/src/discovery/mod.rs`
- `sync/build.rs`

**Proto (1 — auto-generated, attribute the build.rs):**
- `sync/src/proto/p2p.P2p.rs`

**Test Utils (1):**
- `sync/src/test_utils.rs`

### Supporting Type Files Requiring Attribution (6 files)

- `types/src/sync/mod.rs`
- `types/src/sync/active_peers.rs`
- `types/src/storage/shared_in_memory_store.rs`
- `types/src/storage/write_store.rs`
- `types/src/storage/read_store.rs`
- `types/src/config/p2p_config.rs`

### Files NOT Requiring Attribution (Soma-only)

- `types/src/sync/channel_manager.rs` — Soma-specific tonic/gRPC channel management (Sui uses anemo natively)

---

## Test Infrastructure

### Existing Infrastructure (Ready to Use)

**`types/src/storage/shared_in_memory_store.rs`** — Mock store (already exists, fully functional):
```rust
SharedInMemoryStore(Arc<std::sync::RwLock<InMemoryStore>>)
// Implements ReadStore + WriteStore
// Methods: insert_genesis_state(), insert_checkpoint(), insert_certified_checkpoint(),
//          insert_checkpoint_contents(), delete_checkpoint_content_test_only(),
//          update_highest_synced_checkpoint(), update_highest_verified_checkpoint()
```

**`types/src/sync/active_peers.rs`** — ActivePeers for simulating peer connections:
```rust
ActivePeers::new(broadcast_capacity)
// Methods: subscribe(), insert(), remove(), peers(), get(), get_state(), contains()
// Emits PeerEvent::NewPeer / PeerEvent::LostPeer via broadcast
```

### Infrastructure to Create

#### 1. `CommitteeFixture` — Checkpoint Chain Generator

**Sui equivalent:** `crates/sui-swarm-config/src/test_utils.rs` — `CommitteeFixture`

Soma needs a test fixture that can generate certified checkpoint chains. This is the **most critical** infrastructure gap. Without it, almost no state sync test is possible.

**File to create:** `sync/src/test_utils.rs` (uncomment and extend existing stub)

```rust
pub struct CommitteeFixture {
    pub epoch: EpochId,
    pub validators: Vec<(AuthorityKeyPair, StakeUnit)>,
    pub committee: Committee,
}

impl CommitteeFixture {
    /// Generate a committee with `committee_size` validators
    pub fn generate(rng: &mut StdRng, epoch: EpochId, committee_size: usize) -> Self;

    /// Create a chain of empty checkpoints (headers only, no transaction content)
    pub fn make_empty_checkpoints(
        &self,
        count: usize,
        previous: Option<&VerifiedCheckpoint>,
    ) -> (Vec<VerifiedCheckpoint>, SequenceNumber);

    /// Create a chain of checkpoints with random transaction content
    pub fn make_random_checkpoints(
        &self,
        count: usize,
        previous: Option<&VerifiedCheckpoint>,
    ) -> (Vec<VerifiedCheckpoint>, SequenceNumber);

    /// Create an end-of-epoch checkpoint with next committee info
    pub fn make_end_of_epoch_checkpoint(
        &self,
        previous: &VerifiedCheckpoint,
        next_committee: Option<Committee>,
    ) -> VerifiedCheckpoint;

    /// Sign a checkpoint summary with all validators to create a CertifiedCheckpointSummary
    pub fn create_certified_checkpoint(&self, summary: CheckpointSummary) -> VerifiedCheckpoint;
}
```

**Implementation notes:**
- Generates `AuthorityKeyPair` for each validator
- Creates `Committee` from the keypairs with equal stake
- `make_empty_checkpoints` creates headers with correct `previous_digest` chain
- `create_certified_checkpoint` signs with all validator keys to produce quorum signature
- Each checkpoint gets a unique, deterministic digest

#### 2. Channel Manager Test Helper

**File:** `sync/src/test_utils.rs` (extend)

Uncomment and fix the existing `create_test_channel_manager` function:

```rust
pub(crate) async fn create_test_channel_manager<S>(
    own_address: Multiaddr,
    server: P2pServer<P2pService<S>>,
) -> (
    mpsc::Sender<ChannelManagerRequest>,
    broadcast::Receiver<PeerEvent>,
    ActivePeers,
    NetworkKeyPair,
)
where
    S: WriteStore + Clone + Send + Sync + 'static;
```

This is needed for multi-node integration tests where nodes communicate via actual gRPC channels.

#### 3. Builder Test Helpers

Utility functions for quickly creating test state sync and discovery instances:

```rust
/// Create an UnstartedStateSync with a SharedInMemoryStore populated with genesis
pub fn create_test_state_sync(
    config: StateSyncConfig,
    fixture: &CommitteeFixture,
) -> (UnstartedStateSync<SharedInMemoryStore>, StateSyncHandle, SharedInMemoryStore);

/// Create an UnstartedDiscovery with default config
pub fn create_test_discovery(
    config: P2pConfig,
) -> (UnstartedDiscovery, NetworkKeyPair);
```

---

## Priority 1: State Sync Unit Tests

### 1A. Server Push Checkpoint

**Sui equivalent:** `server_push_checkpoint` in `crates/sui-network/src/state_sync/tests.rs`

**Soma file to create:** `sync/src/state_sync/tests.rs`

**Test:** `test_server_push_checkpoint`

**Description:** When a peer pushes a checkpoint summary to our P2P server, verify:
1. `peer_heights` is updated with the peer's reported height
2. The checkpoint is stored in `unprocessed_checkpoints`
3. The highest known sequence number is updated
4. A `StartSyncJob` message is sent to the state sync mailbox (when checkpoint is higher than our highest verified)

**Implementation pattern:**
```rust
#[tokio::test]
async fn test_server_push_checkpoint() {
    // 1. Use P2pBuilder::build_internal() to get raw P2pService + mailbox
    // 2. Create CommitteeFixture, generate genesis + 10 checkpoints
    // 3. Populate store with genesis
    // 4. Manually insert peer into peer_heights with genesis info
    // 5. Call server.push_checkpoint_summary() with a checkpoint
    // 6. Assert: peer_heights.peers[peer_id].height == checkpoint.sequence_number()
    // 7. Assert: peer_heights.unprocessed_checkpoints contains the checkpoint
    // 8. Assert: mailbox received StateSyncMessage::StartSyncJob
}
```

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/server.rs:40-68` (push_checkpoint_summary) | `crates/sui-network/src/state_sync/server.rs` |
| `sync/src/state_sync/mod.rs:136-156` (update_peer_info) | Same |

### 1B. Server Get Checkpoint Summary

**Sui equivalent:** `server_get_checkpoint` in `crates/sui-network/src/state_sync/tests.rs`

**Test:** `test_server_get_checkpoint_summary`

**Description:** Test the `get_checkpoint_summary` RPC for all three request variants:
1. `Latest` — returns genesis initially, then latest after insertion
2. `BySequenceNumber` — returns `None` for missing, correct checkpoint when present
3. `ByDigest` — same behavior

**Implementation pattern:**
```rust
#[tokio::test]
async fn test_server_get_checkpoint_summary() {
    // 1. Build server with genesis in store
    // 2. Test Latest: returns genesis
    // 3. Insert 5 checkpoints into store
    // 4. Test BySequenceNumber(3): returns correct checkpoint
    // 5. Test BySequenceNumber(99): returns None
    // 6. Test ByDigest(checkpoint_3_digest): returns correct checkpoint
    // 7. Test ByDigest(random_digest): returns None
    // 8. Test Latest again: returns checkpoint 5
}
```

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/server.rs:71-91` (get_checkpoint_summary) | `crates/sui-network/src/state_sync/server.rs` |

---

## Priority 2: PeerHeights Unit Tests

**Soma file:** `sync/src/state_sync/tests.rs` (inline tests section)

These are unit tests for the `PeerHeights` struct, which Sui tests inline and via integration tests. Soma should have dedicated unit coverage for this critical data structure.

| # | Test | Description |
|---|------|-------------|
| 1 | `test_peer_heights_insert_peer_info` | Insert new peer, verify fields stored correctly |
| 2 | `test_peer_heights_insert_peer_info_update_height` | Insert same peer with higher height, verify max wins |
| 3 | `test_peer_heights_insert_peer_info_different_genesis` | Insert same peer with different genesis digest, verify overwrite |
| 4 | `test_peer_heights_update_peer_info_same_chain` | Update peer on same chain, verify height ratcheted |
| 5 | `test_peer_heights_update_peer_info_not_on_same_chain` | Update peer not on same chain, verify returns false |
| 6 | `test_peer_heights_highest_known_checkpoint` | Multiple peers, verify correct highest checkpoint returned |
| 7 | `test_peer_heights_cleanup_old_checkpoints` | Insert checkpoints, cleanup below threshold, verify removed |
| 8 | `test_peer_heights_mark_peer_as_not_on_same_chain` | Mark peer, verify excluded from peers_on_same_chain() |

**Implementation pattern:**
```rust
#[test]
fn test_peer_heights_insert_peer_info() {
    let mut heights = PeerHeights {
        peers: HashMap::new(),
        unprocessed_checkpoints: HashMap::new(),
        sequence_number_to_digest: HashMap::new(),
        wait_interval_when_no_peer_to_sync_content: Duration::from_secs(10),
    };

    let peer_id = PeerId::from([1u8; 32]);
    let info = PeerStateSyncInfo {
        genesis_checkpoint_digest: CheckpointDigest::default(),
        on_same_chain_as_us: true,
        height: 42,
        lowest: 0,
    };

    heights.insert_peer_info(peer_id, info);
    assert_eq!(heights.peers.get(&peer_id).unwrap().height, 42);
}
```

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/state_sync/mod.rs:94-227` (PeerHeights) | `crates/sui-network/src/state_sync/mod.rs` |

---

## Priority 3: PeerBalancer Unit Tests

**Soma file:** `sync/src/state_sync/tests.rs`

| # | Test | Description |
|---|------|-------------|
| 1 | `test_peer_balancer_filters_by_height_summary` | Summary mode: only returns peers with `height >= requested` |
| 2 | `test_peer_balancer_filters_by_height_and_lowest_content` | Content mode: only returns peers with `height >= requested && lowest <= requested` |
| 3 | `test_peer_balancer_empty_when_no_eligible_peers` | Returns None when no peers match criteria |

**Sui equivalent:** `test_peer_balancer_sorts_by_throughput` — Sui's version also tests throughput-based sorting, but Soma doesn't have PeerScore, so we test the filtering logic instead.

**Implementation notes:**
- Need to create `ActivePeers` with mock `PeerState` entries
- Insert peers with known heights/lowest values into `PeerHeights`
- Call `PeerBalancer::new().with_checkpoint(N)` and iterate, asserting eligible peers returned

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/state_sync/mod.rs:229-305` (PeerBalancer) | `crates/sui-network/src/state_sync/mod.rs` |

---

## Priority 4: Discovery Unit Tests

**Soma file to create:** `sync/src/discovery/tests.rs`

### 4A. Known Peers Update Logic

**Sui equivalent:** `get_known_peers` in `crates/sui-network/src/discovery/tests.rs`

| # | Test | Description |
|---|------|-------------|
| 1 | `test_update_known_peers_valid_signature` | Valid signed NodeInfo is accepted and stored in known_peers |
| 2 | `test_update_known_peers_invalid_signature` | NodeInfo with wrong/invalid signature is rejected |
| 3 | `test_update_known_peers_future_timestamp_rejected` | NodeInfo with timestamp >30s in the future is rejected |
| 4 | `test_update_known_peers_old_timestamp_rejected` | NodeInfo older than 1 day is rejected |
| 5 | `test_update_known_peers_own_peer_id_skipped` | Our own NodeInfo is not stored in known_peers |

**Implementation pattern:**
```rust
#[test]
fn test_update_known_peers_valid_signature() {
    let mut rng = StdRng::from_seed([0; 32]);
    let our_keypair = NetworkKeyPair::generate(&mut rng);
    let peer_keypair = NetworkKeyPair::generate(&mut rng);

    let our_info = NodeInfo {
        peer_id: PeerId::from(our_keypair.public()),
        address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
        timestamp_ms: now_unix(),
    }.sign(&our_keypair);

    let state = Arc::new(RwLock::new(DiscoveryState {
        our_info: Some(our_info),
        known_peers: HashMap::new(),
    }));

    let peer_info = NodeInfo {
        peer_id: PeerId::from(peer_keypair.public()),
        address: "/ip4/127.0.0.1/tcp/9090".parse().unwrap(),
        timestamp_ms: now_unix(),
    }.sign(&peer_keypair);

    update_known_peers(state.clone(), vec![peer_info.clone()], Arc::new(HashMap::new()));
    assert!(state.read().known_peers.contains_key(&peer_info.peer_id));
}
```

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/discovery/mod.rs:428-491` (update_known_peers) | `crates/sui-network/src/discovery/mod.rs` |

### 4B. Known Peers Replacement Logic

| # | Test | Description |
|---|------|-------------|
| 6 | `test_update_known_peers_newer_replaces_older` | Peer with newer timestamp replaces older entry |
| 7 | `test_update_known_peers_older_does_not_replace_newer` | Peer with older timestamp does NOT replace newer entry |

### 4C. Peer Culling Logic

| # | Test | Description |
|---|------|-------------|
| 8 | `test_peer_culling_removes_old_peers` | Peers older than 1 day (ONE_DAY_MILLISECONDS) are removed during tick |

---

## Priority 5: P2P Server Tests

**Soma file:** `sync/src/state_sync/tests.rs`

### 5A. Get Checkpoint Availability

| # | Test | Description |
|---|------|-------------|
| 1 | `test_server_get_checkpoint_availability` | Returns highest_synced_checkpoint and lowest_available_checkpoint from store |

### 5B. Get Checkpoint Contents

| # | Test | Description |
|---|------|-------------|
| 2 | `test_server_get_checkpoint_contents` | Returns full checkpoint contents by digest, None when missing |

### 5C. Get Known Peers

| # | Test | Description |
|---|------|-------------|
| 3 | `test_server_get_known_peers` | Returns own_info and known_peers list, limited to MAX_PEERS_TO_SEND |
| 4 | `test_server_get_known_peers_no_own_info` | Returns error when own_info not yet initialized |

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/server.rs:93-177` (all server handlers) | `crates/sui-network/src/state_sync/server.rs` + `crates/sui-network/src/discovery/server.rs` |

---

## Priority 6: State Sync Integration Tests

These require the full `CommitteeFixture` infrastructure and either direct event loop manipulation or multi-node setups with `ChannelManager`.

### 6A. Isolated Sync Job

**Sui equivalent:** `isolated_sync_job`

**Test:** `test_isolated_checkpoint_summary_sync`

**Description:** Two nodes — Node 2 has 100 checkpoints, Node 1 knows about them via `peer_heights`. Trigger sync on Node 1, verify it downloads all 100 checkpoint summaries.

**Implementation:**
1. Create two `SharedInMemoryStore` instances with genesis
2. Populate Store 2 with 100 checkpoints via `CommitteeFixture::make_empty_checkpoints()`
3. Create two P2P nodes with `ChannelManager` + `P2pBuilder`
4. Connect Node 1 to Node 2 via `ChannelManagerRequest::Connect`
5. Manually insert Node 2's height into Node 1's `peer_heights`
6. Call `event_loop.maybe_start_checkpoint_summary_sync_task()` on Node 1
7. Wait for task completion
8. Assert: Node 1's store has all 100 checkpoints with correct digests

### 6B. Live Checkpoint Sync

**Sui equivalent:** `sync_with_checkpoints_being_inserted`

**Test:** `test_sync_with_checkpoints_being_inserted`

**Description:** Checkpoints are produced one-by-one on Node 1 (simulating consensus). Node 2 should receive each checkpoint via state sync push notifications and download contents.

**Implementation:**
1. Create two connected nodes
2. Spawn both event loops as tokio tasks
3. Use `handle.send_checkpoint()` to inject checkpoints on Node 1
4. Use `handle.subscribe_to_synced_checkpoints()` on Node 2 to verify arrival
5. Assert: Both stores have identical checkpoints at the end

### 6C. Watermark-Based Content Sync

**Sui equivalent:** `sync_with_checkpoints_watermark`

**Test:** `test_sync_with_checkpoint_watermarks`

**Description:** Tests the low-watermark (pruning) mechanism. Peer 1 has pruned old checkpoints (high `lowest`). Peer 2 must get content from Peer 3 (the only node with full content).

**Implementation:**
1. Create 3 connected nodes
2. Node 1: all checkpoints but `lowest=100` (simulating pruning)
3. Node 2: needs to sync
4. Node 3: all checkpoints with `lowest=0`
5. Verify Node 2 syncs from Node 3 (not Node 1) for content below watermark

### 6D. Epoch Boundary Sync

**Soma-specific test** (no direct Sui equivalent — Sui's test is implicit)

**Test:** `test_sync_across_epoch_boundary`

**Description:** Verify state sync correctly handles end-of-epoch checkpoints:
1. Generate checkpoints for epoch 0, ending with an end-of-epoch checkpoint containing the next committee
2. Generate checkpoints for epoch 1 signed by the new committee
3. Syncing node should verify epoch 0 checkpoints with epoch 0 committee and epoch 1 checkpoints with epoch 1 committee

### 6E. Handle Checkpoint From Consensus

**Test:** `test_handle_checkpoint_from_consensus`

**Description:** Verify that `handle_checkpoint_from_consensus()`:
1. Panics if `previous_digest` doesn't match the preceding checkpoint
2. Skips checkpoints older than `highest_verified_checkpoint`
3. Inserts end-of-epoch committee for next epoch
4. Updates both `highest_verified_checkpoint` and `highest_synced_checkpoint`
5. Broadcasts the checkpoint to subscribers
6. Triggers `spawn_notify_peers_of_checkpoint`

---

## Priority 7: Discovery Integration Tests

### 7A. Seed Peer Connection

**Sui equivalent:** `make_connection_to_seed_peer`

**Test:** `test_seed_peer_connection`

**Description:** Two nodes — Node 2 is configured as a seed peer for Node 1. After discovery starts, Node 1 should connect to Node 2 via `ChannelManagerRequest::Connect`.

### 7B. Three-Node Gossip Discovery

**Sui equivalent:** `three_nodes_can_connect_via_discovery`

**Test:** `test_three_nodes_discover_via_gossip`

**Description:** Node 1 is seed for Nodes 2 and 3. After gossip exchange, Node 2 should discover Node 3 (and vice versa) via Node 1's `get_known_peers` response.

### 7C. Peer Lost Cleanup

**Test:** `test_peer_lost_removes_from_known_peers`

**Description:** When a `PeerEvent::LostPeer` is received, the peer is removed from `known_peers` in `DiscoveryState`.

---

## Priority 8: E2E State Sync Resilience Tests (msim)

**Sui equivalent:** `test_state_sync_with_degraded_peers` in `crates/sui-e2e-tests/tests/state_sync_resilience_tests.rs`

**Soma file:** `e2e-tests/tests/state_sync_resilience_tests.rs` (new file)

### 8A. State Sync With Degraded Peers

**Test:** `test_state_sync_with_degraded_peers`

**Description:** Full msim test with 1 validator + multiple fullnodes. After an initial sync period, inject high latency between the syncing node and most peers. Verify the syncing node eventually catches up despite degraded network conditions.

**Implementation pattern:**
```rust
#[sim_test]
async fn test_state_sync_with_degraded_peers() {
    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(10000)
        .build()
        .await;

    // Let network run and produce checkpoints
    sleep(Duration::from_secs(30)).await;

    // Spawn a new fullnode
    let fullnode = test_cluster.spawn_new_fullnode().await;

    // Verify fullnode catches up to validators
    let timeout = Duration::from_secs(120);
    let start = Instant::now();
    loop {
        let validator_height = test_cluster.fullnode_handle.soma_node.with(|node| {
            *node.state().get_highest_synced_checkpoint().unwrap().sequence_number()
        });
        let fullnode_height = fullnode.soma_node.with(|node| {
            *node.state().get_highest_synced_checkpoint().unwrap().sequence_number()
        });
        if fullnode_height >= validator_height.saturating_sub(5) {
            break;
        }
        if start.elapsed() > timeout {
            panic!("Fullnode failed to catch up: validator={}, fullnode={}", validator_height, fullnode_height);
        }
        sleep(Duration::from_secs(1)).await;
    }
}
```

### 8B. State Sync Catch-Up After Partition

**Soma-specific test**

**Test:** `test_state_sync_catch_up_after_partition`

**Description:** Fullnode syncs normally, then is partitioned (killed) for several epochs. After restart, it should catch up via state sync from peers.

---

## Priority 9: Worker / Archive Sync Tests

### 9A. Archive-Based Checkpoint Sync

**Sui equivalent:** `test_state_sync_using_archive`

**Test:** `test_state_sync_using_archive`

**Description:** When peers have pruned old checkpoints (high `lowest` watermark), verify the archive fallback path:
1. Node 2 has 100 checkpoints but has pruned contents for 0-9
2. Node 1 is configured with archive URL pointing to local temp directory
3. Node 1 syncs: checkpoints 0-9 from archive, 10-99 from Node 2

**Implementation notes:**
- Requires `data-ingestion` crate for archive setup
- Use `tempdir()` with encoded checkpoint blobs
- Set `ArchiveReaderConfig` with `file://` URL

### 9B. Worker Process Checkpoint

**Test:** `test_worker_process_checkpoint`

**Description:** Unit test for `StateSyncWorker::process_checkpoint()`:
1. Create a `CheckpointData` with summary, contents, and transactions
2. Call `worker.process_checkpoint(&data)`
3. Verify store contains the checkpoint, contents, and `highest_synced_checkpoint` is updated

**Cross-reference files:**
| Soma | Sui |
|------|-----|
| `sync/src/state_sync/worker.rs:13-36` (StateSyncWorker) | `crates/sui-network/src/state_sync/worker.rs` |

### 9C. Worker Verify-and-Insert Checkpoint

**Test:** `test_get_or_insert_verified_checkpoint`

**Description:** Unit test for `get_or_insert_verified_checkpoint()`:
1. Returns existing checkpoint from store without re-verification
2. Verifies and inserts a new checkpoint when not in store
3. Fails when previous checkpoint is missing
4. Inserts without verification when `verify=false`

---

## Gaps Between Soma & Sui

These are features present in Sui's state sync/discovery but missing or commented out in Soma. They should be noted in the testing plan but may not need tests until the features are implemented.

| Feature | Sui | Soma | Test Impact |
|---------|-----|------|-------------|
| **PeerScore / throughput tracking** | Full `PeerScore` struct with throughput, failure rate, min samples | Not present | Skip throughput-sorting tests |
| **Adaptive timeout** | `compute_adaptive_timeout()` scales timeout by tx count | Not present | Skip adaptive timeout tests |
| **max_checkpoint_lookahead** | Server rejects pushes >N ahead of verified | Not present | Skip lookahead test (but note as future improvement) |
| **CheckpointContentsDownloadLimitLayer** | Per-digest semaphore rate limiting | Not present | Skip rate limit tests |
| **Per-RPC rate limits** | Governor-based rate limiting per endpoint | Not present | Skip |
| **Discovery AccessType** | Public/Private/Trusted peer visibility | Not present (commented out) | Skip access type tests |
| **EndpointManager** | Dynamic address updates (Admin > Config > Committee) | Not present | Skip |
| **Discovery metrics** | `num_peers_with_external_address` gauge | Not present | Skip |
| **State sync metrics** | `highest_known`, `highest_verified`, `highest_synced` gauges | Not present | Skip |
| **connected_peers tracking** | In `State.connected_peers` | Commented out | Discovery tick filtering is impaired |
| **Shutdown handle** | `oneshot::Receiver<()>` in discovery event loop | Commented out | Discovery runs until task cancelled |

---

## Implementation Order

### Phase 1: Test Infrastructure (Day 1-2)
1. **Create `CommitteeFixture`** in `sync/src/test_utils.rs`
   - Generate validator committees with keypairs
   - `make_empty_checkpoints()` — chain of certified headers
   - `make_random_checkpoints()` — chain with transaction content
   - `make_end_of_epoch_checkpoint()` — epoch boundary
2. **Uncomment and fix `create_test_channel_manager`**
3. **Add test module declarations** to `state_sync/mod.rs` and `discovery/mod.rs`

### Phase 2: PeerHeights Unit Tests (Day 2-3)
4. **8 PeerHeights tests** (Priority 2) — pure unit tests, no async
5. **3 PeerBalancer tests** (Priority 3) — need mock ActivePeers

### Phase 3: Server Handler Tests (Day 3-4)
6. **2 state sync server tests** (Priority 1A, 1B) — async, use `P2pBuilder::build_internal()`
7. **4 P2P server tests** (Priority 5) — get_checkpoint_availability, get_checkpoint_contents, get_known_peers

### Phase 4: Discovery Unit Tests (Day 4-5)
8. **8 discovery tests** (Priority 4) — update_known_peers, signature validation, timestamp filtering

### Phase 5: Integration Tests (Day 5-8)
9. **5 state sync integration tests** (Priority 6) — multi-node sync via ChannelManager
10. **3 discovery integration tests** (Priority 7) — seed peers, gossip discovery

### Phase 6: Archive & E2E (Day 8-10)
11. **3 worker/archive tests** (Priority 9) — worker process, verify-and-insert, archive sync
12. **2 E2E resilience tests** (Priority 8) — msim degraded peers, catch-up after partition

### Phase 7: Attribution (Day 10)
13. **Add attribution headers** to all 15 derived files
14. **Verify all tests pass**: `cargo test -p sync` and msim variants

---

## Build & Run Commands

```bash
# Run all sync unit tests
cargo test -p sync

# Run specific test file
cargo test -p sync -- state_sync::tests
cargo test -p sync -- discovery::tests

# Run a single test
cargo test -p sync -- test_server_push_checkpoint

# Build for msim (if any tests use msim)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p sync

# Run E2E resilience tests (msim required)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test state_sync_resilience_tests

# Check compilation only
PYO3_PYTHON=python3 cargo check -p sync
```

---

## Summary of New Files to Create

| File | Tests | Priority |
|------|-------|----------|
| `sync/src/state_sync/tests.rs` | ~20 | P1-P3, P5-P6 |
| `sync/src/discovery/tests.rs` | ~11 | P4, P7 |
| `sync/src/test_utils.rs` (rewrite) | 0 (infrastructure) | P0 |
| `e2e-tests/tests/state_sync_resilience_tests.rs` | 2 | P8 |
| **Total new files** | **4** | |
| **Total new tests** | **~45** | |

---

## Sui Cross-Reference URLs

Key Sui files referenced in this plan:

| Category | Sui File Path |
|----------|--------------|
| State sync tests | `crates/sui-network/src/state_sync/tests.rs` |
| State sync module | `crates/sui-network/src/state_sync/mod.rs` |
| State sync server | `crates/sui-network/src/state_sync/server.rs` |
| State sync builder | `crates/sui-network/src/state_sync/builder.rs` |
| State sync worker | `crates/sui-network/src/state_sync/worker.rs` |
| State sync metrics | `crates/sui-network/src/state_sync/metrics.rs` |
| Discovery tests | `crates/sui-network/src/discovery/tests.rs` |
| Discovery module | `crates/sui-network/src/discovery/mod.rs` |
| Discovery server | `crates/sui-network/src/discovery/server.rs` |
| Discovery builder | `crates/sui-network/src/discovery/builder.rs` |
| E2E resilience | `crates/sui-e2e-tests/tests/state_sync_resilience_tests.rs` |
| SharedInMemoryStore | `crates/sui-types/src/storage/shared_in_memory_store.rs` |
| CommitteeFixture | `crates/sui-swarm-config/src/test_utils.rs` |
| Network utils | `crates/sui-network/src/utils.rs` |
| build.rs (proto gen) | `crates/sui-network/build.rs` |
