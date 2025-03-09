# Soma Blockchain System Patterns

## Purpose and Scope
This document outlines the key architectural and implementation patterns used throughout the Soma blockchain. It serves as a guide for understanding how components are designed and how they interact, as well as providing a consistent framework for implementing new features.

## Core Architectural Patterns

### Module-Based Architecture
Soma is organized into distinct modules with clear responsibilities and boundaries:

- **Authority Module**: State management, transaction validation and execution
- **Consensus Module**: Byzantine fault tolerant agreement protocol
- **Node Module**: Component lifecycle and service orchestration
- **P2P Module**: Network discovery and state synchronization

**When to use**:
- When designing new components that need to interact with existing modules
- For understanding system organization and relationships
- When planning cross-module features and integration points

### Actor Model Adaptation
Soma adapts aspects of the actor model, with components that:
- Maintain private internal state
- Communicate through message passing via channels
- Handle requests asynchronously through mailboxes
- Manage their own lifecycle

```rust
pub async fn start(mut self) {
    loop {
        tokio::select! {
            maybe_message = self.mailbox.recv() => {
                if let Some(message) = maybe_message {
                    self.handle_message(message).await;
                } else {
                    break;
                }
            }
            // Additional message sources
        }
    }
}
```

**When to use**:
- For long-running components that need to process multiple message types
- When state isolation and message passing is preferred over shared mutable state
- For components that need to manage concurrent operations

### State Hierarchy Pattern
Soma organizes state in a hierarchical pattern with decreasing scope:

- **Global State**: System-wide state (Committee, Genesis configuration)
- **Epoch State**: State valid for current epoch (AuthorityPerEpochStore)
- **Transaction State**: State scoped to transaction execution (TemporaryStore)
- **Operation State**: State specific to an operation (single transaction processing)

**When to use**:
- For organizing state with different validity periods
- When different components need access to state at different levels
- To provide clear boundaries for state transitions and scope

### Checkpoint and Reconfiguration Pattern
Soma uses a checkpoint-based approach to handle reconfiguration:

- Critical operations complete before reconfiguration
- State is saved at checkpoint boundaries
- New configuration takes effect atomically
- Recovery is possible from checkpoint state

**When to use**:
- When implementing features that span epoch boundaries
- For ensuring atomic updates to system configuration
- To provide clean recovery points for system state

## Concurrency Patterns

### RwLock Pattern
Soma uses `RwLock` extensively for shared state that allows multiple concurrent readers with exclusive writer access.

```rust
pub struct AuthorityState {
    // State fields
    execution_lock: RwLock<EpochId>
}
```

**When to use**:
- For state that is frequently read but less frequently written
- When you need to ensure readers have a consistent view of the state
- For coordinating access across multiple async tasks

### ArcSwap Pattern
For hot-swappable components like epoch stores, Soma uses `ArcSwap` to atomically replace entire data structures.

```rust
pub struct AuthorityState {
    epoch_store: ArcSwap<AuthorityPerEpochStore>
}
```

**When to use**:
- For epoch transitions where an entire state component needs to be replaced
- When you need atomic replacement without disrupting ongoing operations
- For components where readers should always see a consistent version

### Mutex Pattern
For simple exclusive access, Soma uses `tokio::sync::Mutex` for async contexts and `std::sync::Mutex` or `parking_lot::Mutex` for non-async contexts.

**When to use**:
- When exclusive access is required and multiple readers aren't needed
- For protecting critical sections that modify multiple fields atomically
- When simplicity is more important than read concurrency

### Lock Hierarchies Pattern
Soma enforces strict lock hierarchies to prevent deadlocks:

```
ExecutionLock -> EpochStoreLock -> TransactionLock -> ObjectLock
```

**When to use**:
- When implementing features that require multiple locks
- To maintain system safety with complex locking requirements
- For ensuring deterministic lock acquisition across the codebase

### JoinSet Pattern
For managing groups of related tasks, Soma uses `tokio::task::JoinSet`:

```rust
let mut tasks = JoinSet::new();
tasks.spawn(async move {
    // Task logic here
});

while let Some(result) = tasks.join_next().await {
    match result {
        Ok(value) => {
            // Handle successful completion
        },
        Err(e) => {
            if e.is_cancelled() {
                // Handle cancellation
            } else if e.is_panic() {
                // Handle panic
                std::panic::resume_unwind(e.into_panic());
            }
        }
    }
}
```

**When to use**:
- For spawning and tracking multiple related background tasks
- When you need to gracefully handle task completion, cancellation, and panics
- For implementing proper task supervision patterns

### Notification Pattern
Soma uses broadcasting channels to notify interested components about state changes:

```rust
pub struct NotifyRead<T> {
    value: RwLock<Option<T>>,
    notify: Notify,
}
```

**When to use**:
- When multiple components need to be notified of the same event
- For implementing waiting conditions with timeouts
- To decouple event producers from consumers

## Error Handling Patterns

### Module-Specific Error Types
Each module defines its own error type with `thiserror::Error`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum SomaError {
    #[error("epoch has ended: {0}")]
    EpochEnded(EpochId),
    
    #[error("wrong epoch, expected {expected_epoch}, actual {actual_epoch}")]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },
    
    #[error("database error: {0}")]
    DatabaseError(#[from] std::io::Error),
    
    #[error("internal error: {0}")]
    InternalError(String),
}
```

**When to use**:
- For module-specific errors that need detailed context
- When errors need to be categorized by type (validation, internal, network, etc.)
- For providing clear error messages with structured data

### Error Categorization Pattern
Soma categorizes errors to determine appropriate handling:

- **Permanent Errors**: Indicate a request that can never succeed (Invalid signature)
- **Transient Errors**: Temporary failures that might succeed with retry (Network timeout)
- **Recoverable Errors**: Errors that require explicit recovery action (Database corruption)
- **Fatal Errors**: Errors that require component restart (Unrecoverable state inconsistency)

**When to use**:
- When designing error handling for operation reliability
- For implementing retry mechanisms with appropriate backoff
- To provide clear indicators for operational monitoring

### Error Propagation with Context
Errors are propagated using the `?` operator with context addition:

```rust
fn process_operation() -> SomaResult<Output> {
    let data = fetch_data()
        .map_err(|e| SomaError::InternalError(format!("Failed to fetch data: {}", e)))?;
    
    let result = validate_data(data)?;
    
    Ok(process_result(result)?)
}
```

**When to use**:
- For propagating errors up the call stack
- When additional context needs to be added
- For converting between error types

### Result Type Aliases
Soma uses module-specific `Result` type aliases for clearer function signatures:

```rust
pub type SomaResult<T> = Result<T, SomaError>;
```

**When to use**:
- For consistent return types across a module
- To avoid repeating complex error types
- For readability and maintainability

## State Management Patterns

### Epoch-Based State Isolation
Soma isolates state by epoch, with dedicated storage for each epoch:

```rust
pub struct AuthorityState {
    epoch_store: ArcSwap<AuthorityPerEpochStore>
}

pub struct AuthorityPerEpochStore {
    // Epoch-specific fields
    committee: Committee,
    database: Arc<AuthorityStore>,
    // More epoch-specific fields
}
```

**When to use**:
- For data that has different validity periods
- When you need clean reconfiguration boundaries
- For isolating failures between epochs

### Temporary Store Pattern
Soma uses temporary stores for transaction execution:

```rust
pub struct TemporaryStore {
    input_objects: BTreeMap<ObjectID, Object>,
    written_objects: BTreeMap<ObjectID, (Object, WriteKind)>,
    events: Vec<Event>,
    max_binary_format_version: u64,
}
```

**When to use**:
- For speculative execution before commitment
- When operations need to be validated before permanent state changes
- For transactional semantics with abort capability

### Arc for Shared Ownership
Immutable shared components use `Arc` for reference counting:

```rust
pub struct AuthorityState {
    committee_store: Arc<CommitteeStore>,
    transaction_manager: Arc<TransactionManager>
}
```

**When to use**:
- For components that are shared between multiple owners
- When the component doesn't change after initialization
- For efficient passing of complex structures

### Versioned Object Pattern
Soma uses versioned objects to track state changes:

```rust
pub struct Object {
    // Object fields
    previous_transaction: TransactionDigest,
    version: SequenceNumber,
    // More fields
}
```

**When to use**:
- For tracking object lineage and history
- When concurrency control requires versioning
- For enabling deterministic execution verification

### Thread-Safe Caching
For performance-critical components, Soma implements thread-safe caches:

```rust
pub struct WritebackCache<K, V> {
    inner: Arc<RwLock<HashMap<K, V>>>,
    // More fields
}
```

**When to use**:
- For frequently accessed data that's expensive to compute
- When you need to coordinate access across threads
- For implementing complex locking patterns
- For improving performance of hot paths

## Asynchronous Programming Patterns

### Event Loop Pattern
Many components use event loops to process messages asynchronously:

```rust
pub async fn start(mut self) {
    info!("Service started");
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    loop {
        tokio::select! {
            now = interval.tick() => {
                self.handle_tick(now.into_std()).await;
            },
            maybe_message = self.mailbox.recv() => {
                if let Some(message) = maybe_message {
                    self.handle_message(message).await;
                } else {
                    break;
                }
            },
            peer_event = self.peer_event_receiver.recv() => {
                self.handle_peer_event(peer_event).await;
            },
            Some(task_result) = self.tasks.join_next() => {
                self.handle_task_result(task_result).await;
            },
        }
    }
    
    info!("Service stopped");
}
```

**When to use**:
- For components that need to handle multiple message types
- When you need to coordinate between multiple channels
- For long-running background services
- For components that need periodic operations

### Cancellation Propagation
Critical operations handle cancellation gracefully:

```rust
tokio::select! {
    result = async_operation() => {
        // Handle result
    }
    _ = tokio::time::sleep(Duration::from_secs(timeout)) => {
        // Handle timeout
    }
}
```

**When to use**:
- For operations that might take too long
- When tasks need to be cancelled cleanly
- For implementing proper timeout handling
- For responsive shutdown of background operations

### Channel-Based Communication
Components communicate using channels to decouple producers and consumers:

```rust
let (tx, rx) = tokio::sync::mpsc::channel(32);
```

**When to use**:
- For communicating between async tasks
- When you need to decouple components
- For implementing backpressure
- To avoid shared mutable state

### Structured Concurrency Pattern
Soma uses structured concurrency to manage task lifetimes:

```rust
async fn process_with_dependencies() -> Result<()> {
    let mut tasks = JoinSet::new();
    
    // Spawn dependent tasks
    tasks.spawn(async move { process_part_1().await });
    tasks.spawn(async move { process_part_2().await });
    
    // Wait for all tasks to complete
    while let Some(result) = tasks.join_next().await {
        result??; // Propagate errors
    }
    
    Ok(())
}
```

**When to use**:
- When tasks have a clear parent-child relationship
- For operations that spawn multiple sub-tasks
- To ensure proper cleanup of resources
- For propagating cancellation to child tasks

## Testing Patterns

### Async Testing
Tests use `tokio::test` for asynchronous test cases:

```rust
#[tokio::test]
async fn test_consensus_authority_operation() {
    // Test setup and assertions
    let authority = setup_test_authority().await;
    
    let result = authority.process_transaction(test_tx).await;
    
    assert!(result.is_ok());
    // Additional assertions
}
```

**When to use**:
- For testing async functions and flows
- When you need a runtime for async operations
- For integration tests that span multiple components

### Mock Components
Critical interfaces have mock implementations for testing:

```rust
#[derive(Default)]
struct MockConsensusClient {
    transactions: Mutex<Vec<VerifiedTransaction>>,
    handle_certificate_calls: AtomicUsize,
}

impl ConsensusClient for MockConsensusClient {
    async fn handle_certificate(&self, certificate: VerifiedCertificate) -> SomaResult<()> {
        self.handle_certificate_calls.fetch_add(1, Ordering::SeqCst);
        let tx = self.transactions.lock().unwrap();
        tx.push(certificate.into_inner());
        Ok(())
    }
    
    // Other method implementations
}
```

**When to use**:
- For isolating components during testing
- When external dependencies need to be controlled
- For simulating error conditions
- To verify correct interaction between components

### Property-Based Testing
For critical algorithms, Soma uses property-based testing to verify invariants:

```rust
proptest! {
    #[test]
    fn test_transaction_execution_determinism(
        txs in vec(any::<SignedTransaction>(), 1..10),
        seed in any::<u64>(),
    ) {
        let mut store1 = TemporaryStore::new(seed);
        let mut store2 = TemporaryStore::new(seed);
        
        let result1 = execute_transactions(&mut store1, &txs);
        let result2 = execute_transactions(&mut store2, &txs);
        
        prop_assert_eq!(result1, result2);
        prop_assert_eq!(store1.objects(), store2.objects());
    }
}
```

**When to use**:
- For testing algorithm correctness with randomized inputs
- When verifying invariants across many possible states
- For discovering edge cases in complex logic
- To test determinism and reproducibility

### Test Fixtures
Common test scenarios use fixtures for setup:

```rust
fn setup_test_authority() -> AuthorityState {
    // Setup code to create a test authority
    let committee = Committee::new_for_testing();
    let keypair = generate_test_keypair();
    
    AuthorityState::new_for_testing(committee, keypair)
}
```

**When to use**:
- For complex setup code shared between tests
- When you need consistent test environments
- For improving test readability
- To avoid duplication in test code

### Randomized Integration Testing
Soma uses randomized integration tests for Byzantine behavior:

```rust
#[tokio::test]
async fn test_consensus_with_byzantine_nodes() {
    // Create a test network with some Byzantine nodes
    let mut network = TestNetwork::new(7, 2); // 7 nodes, 2 Byzantine
    
    // Generate test transactions
    let txs = generate_test_transactions(100);
    
    // Submit transactions and verify consensus
    let results = network.process_transactions(txs).await;
    
    // Verify safety properties
    assert!(network.verify_safety_properties());
    
    // Verify liveness properties
    assert!(network.verify_liveness_properties());
}
```

**When to use**:
- For testing consensus safety and liveness
- When simulating Byzantine behavior
- For verifying system properties under stress
- To test recovery from adverse conditions

## Cross-Module Integration Patterns

### Dependency Injection Pattern
Soma uses dependency injection for module integration:

```rust
pub struct ConsensusAuthority {
    context: Arc<Context>,
    transaction_client: Arc<dyn TransactionClient>,
    synchronizer: Arc<SynchronizerHandle>,
    // More dependencies
}

impl ConsensusAuthority {
    pub fn new(
        context: Arc<Context>,
        transaction_client: Arc<dyn TransactionClient>,
        synchronizer: Arc<SynchronizerHandle>,
        // More parameters
    ) -> Self {
        Self {
            context,
            transaction_client,
            synchronizer,
            // Initialize other fields
        }
    }
}
```

**When to use**:
- For creating testable components
- When components need configurable dependencies
- For flexible module composition
- To enable mock implementations in tests

### Handler Registration Pattern
Soma uses handler registration for extensible event processing:

```rust
pub struct Node {
    handlers: HashMap<HandlerType, Box<dyn Handler>>,
}

impl Node {
    pub fn register_handler(&mut self, handler_type: HandlerType, handler: Box<dyn Handler>) {
        self.handlers.insert(handler_type, handler);
    }
    
    pub fn handle_event(&self, event: Event) {
        if let Some(handler) = self.handlers.get(&event.handler_type) {
            handler.handle(event);
        }
    }
}
```

**When to use**:
- For creating extensible event handling
- When components need to be dynamically reconfigured
- For implementing plugin architectures
- To support multiple protocol versions

### Service Interface Pattern
Soma defines clear service interfaces between modules:

```rust
#[async_trait]
pub trait AuthorityService {
    async fn handle_transaction(&self, transaction: SignedTransaction) 
        -> SomaResult<TransactionResponse>;
        
    async fn handle_certificate(&self, certificate: VerifiedCertificate)
        -> SomaResult<TransactionEffects>;
    
    // More methods
}
```

**When to use**:
- For defining clear module boundaries
- When implementing RPC or service endpoints
- For enabling alternative implementations
- To document cross-module contracts

## Confidence: 9/10
This document provides a comprehensive overview of the key patterns used in the Soma blockchain. The patterns have been verified against the codebase and reflect the current implementation. As the system evolves, additional patterns may emerge, but the core architectural approaches documented here are stable.

## Last Updated: 2025-03-08 by Cline
