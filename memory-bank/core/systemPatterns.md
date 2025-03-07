# Soma System Patterns

## Architectural Patterns

### Component Architecture
The system is organized into primary modules that interact with clear boundaries:
- **Authority**: State management, transaction validation and execution
- **Consensus**: BFT agreement protocol implementation (Mysticeti)
- **Node**: Lifecycle and orchestration 
- **P2P**: Network discovery and state synchronization

```rust
// From node/lib.rs - Component organization example
pub struct SomaNode {
    config: NodeConfig,
    validator_components: Mutex<Option<ValidatorComponents>>,
    end_of_epoch_channel: broadcast::Sender<SystemState>,
    state: Arc<AuthorityState>,
    transaction_orchestrator: Option<Arc<TransactiondOrchestrator<NetworkAuthorityClient>>>,
    state_sync_handle: StateSyncHandle,
    commit_store: Arc<CommitStore>,
    accumulator: Mutex<Option<Arc<StateAccumulator>>>,
    consensus_store: Arc<dyn ConsensusStore>,
    auth_agg: Arc<ArcSwap<AuthorityAggregator<NetworkAuthorityClient>>>,
}
```

### State Management
- The primary state management is through `AuthorityState` and epoch-specific `AuthorityPerEpochStore`
- Clear separation of per-epoch data to handle reconfiguration
- Thread-safe state access via Arc<RwLock<>> pattern

```rust
// From authority/state.rs
pub struct AuthorityState {
    pub name: AuthorityName,
    pub secret: StableSyncAuthoritySigner,
    epoch_store: ArcSwap<AuthorityPerEpochStore>,
    execution_lock: RwLock<EpochId>,
    committee_store: Arc<CommitteeStore>,
    transaction_manager: Arc<TransactionManager>,
    // Additional fields...
}
```

### Event-Loop and Handler Pattern
Central event loops for message processing with handler functions for specific message types:

```rust
// From p2p/state_sync/mod.rs
pub async fn start(mut self) {
    info!("State-Synchronizer started");

    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    loop {
        tokio::select! {
            now = interval.tick() => {
                self.handle_tick(now.into_std());
            },
            maybe_message = self.mailbox.recv() => {
                if let Some(message) = maybe_message {
                    self.handle_message(message);
                } else {
                    break;
                }
            },
            peer_event = self.peer_event_receiver.recv() => {
                self.handle_peer_event(peer_event);
            },
            Some(task_result) = self.tasks.join_next() => {
                // Handle task result
            },
        }
    }
}
```

### Task Spawning and Management
The codebase uses JoinSet for tracking related tasks with proper error handling:

```rust
// Pattern observed in multiple components
let mut tasks = JoinSet::new();
tasks.spawn(async move {
    // Task logic here
});

while let Some(result) = tasks.join_next().await {
    match result {
        Ok(value) => println!("Task completed with: {value:?}"),
        Err(e) => {
            if e.is_cancelled() {
                // Handle cancellation
            } else if e.is_panic() {
                // Propagate panics
                std::panic::resume_unwind(e.into_panic());
            } else {
                // Handle other errors
            }
        }
    };
}
```

## Blockchain-Specific Patterns

### Consensus Engine Structure
The consensus mechanism implements a Core-based design with round management:

```rust
// From consensus/authority.rs
pub struct ConsensusAuthority {
    context: Arc<Context>,
    transaction_client: Arc<TransactionClient>,
    synchronizer: Arc<SynchronizerHandle>,
    commit_syncer: CommitSyncer<TonicClient>,
    core_thread_handle: CoreThreadHandle,
    // Additional fields...
}
```

### Transaction Processing Flow
The system follows a clear transaction process:
1. **Validation**: Verify transaction signatures and inputs
2. **Execution**: Process the transaction in a temporary store
3. **Commit**: Write results to permanent storage
4. **Notification**: Signal execution completion
5. **State Update**: Update global state with transaction effects

```rust
// From authority/state.rs
pub async fn try_execute_immediately(
    &self,
    certificate: &VerifiedExecutableTransaction,
    expected_effects_digest: Option<TransactionEffectsDigest>,
    commit: Option<CommitIndex>,
    epoch_store: &Arc<AuthorityPerEpochStore>,
) -> SomaResult<(TransactionEffects, Option<ExecutionError>)> {
    // Transaction execution logic
}
```

### Epoch Management and Reconfiguration
Clean epoch transition with state and validator set changes:

```rust
// From node/lib.rs
pub async fn reconfigure(
    self: Arc<Self>,
    cur_epoch_store: &AuthorityPerEpochStore,
    new_committee: Committee,
    epoch_start_configuration: EpochStartConfiguration,
    epoch_last_commit: CommitIndex,
) -> SomaResult<Arc<AuthorityPerEpochStore>> {
    // Reconfiguration logic
}
```

### State Protection Patterns
The codebase uses various synchronization primitives for state protection:
- `RwLock` for shared state that needs concurrent readers
- `Mutex` or `parking_lot::RwLock` for non-async contexts
- `tokio::sync::Mutex` for async contexts requiring exclusive access
- `ArcSwap` for hot-swappable components like epoch store

```rust
// State protection example from authority/epoch_store.rs
pub struct AuthorityPerEpochStore {
    pub(crate) name: AuthorityName,
    committee: Arc<Committee>,
    tables: ArcSwapOption<AuthorityEpochTables>,
    consensus_quarantine: RwLock<ConsensusOutputQuarantine>,
    consensus_output_cache: ConsensusOutputCache,
    protocol_config: ProtocolConfig,
    reconfig_state_mem: RwLock<ReconfigState>,
    // Additional fields...
}
```

## Error Handling

### Error Type Hierarchy
The code uses module-specific error types with thiserror:

```rust
// Error pattern observed in codebase
#[derive(Debug, thiserror::Error)]
pub enum SomaError {
    #[error("epoch has ended: {0}")]
    EpochEnded(EpochId),
    
    #[error("wrong epoch, expected {expected_epoch}, actual {actual_epoch}")]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },
    
    #[error("validator halted at epoch end")]
    ValidatorHaltedAtEpochEnd,
    
    #[error("database error: {0}")]
    DatabaseError(#[from] std::io::Error),
    
    #[error("internal error: {0}")]
    InternalError(String),
}
```

### Error Propagation
Errors are consistently propagated with the `?` operator and context is added:

```rust
// Error propagation pattern
fn process_operation() -> SomaResult<Output> {
    let data = fetch_data()
        .map_err(|e| SomaError::InternalError(format!("Failed to fetch data: {}", e)))?;
    
    let result = validate_data(data)?;
    
    Ok(process_result(result)?)
}
```

## Testing Patterns

### Async Testing
The project uses tokio::test for asynchronous test cases:

```rust
#[tokio::test]
async fn test_consensus_authority_operation() {
    // Setup test environment with mock components
    let authority = setup_test_authority().await;
    
    // Execute test operations
    let result = authority.process_transaction(test_tx).await;
    
    // Verify results
    assert!(result.is_ok());
}
```

### Test Fixtures
Centralized test fixtures are used for common test scenarios:

```rust
// From consensus/core.rs tests
#[cfg(test)]
pub(crate) struct CoreTextFixture {
    pub core: Core,
    pub signal_receivers: CoreSignalsReceivers,
    pub block_receiver: broadcast::Receiver<VerifiedBlock>,
    pub commit_receiver: UnboundedReceiver<CommittedSubDag>,
    pub store: Arc<MemStore>,
}

#[cfg(test)]
impl CoreTextFixture {
    fn new(context: Context, authorities: Vec<Stake>, own_index: AuthorityIndex) -> Self {
        // Setup logic for test fixture
    }
}
```

## P2P Network Patterns

### Peer Discovery
The P2P module implements peer discovery with signed node information exchange:

```rust
// From p2p/discovery/mod.rs
fn update_known_peers(
    state: Arc<RwLock<DiscoveryState>>,
    found_peers: Vec<SignedNodeInfo>,
    allowlisted_peers: Arc<HashMap<PeerId, Option<Multiaddr>>>,
) {
    // Peer discovery logic
}
```

### State Synchronization
The system implements state sync to catch up with latest chain state:

```rust
// From p2p/state_sync/mod.rs
async fn sync_from_peer<S>(
    active_peers: ActivePeers,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit_event_sender: broadcast::Sender<CommittedSubDag>,
    weak_sender: mpsc::WeakSender<StateSyncMessage>,
    block_verifier: Arc<SignedBlockVerifier>,
    timeout: Duration,
    target_commit: CommitIndex,
) {
    // State sync implementation
}
