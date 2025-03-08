# Soma Blockchain System Patterns

## Purpose and Scope
This document outlines the key architectural and implementation patterns used throughout the Soma blockchain. It serves as a guide for understanding how components are designed and how they interact, as well as providing a consistent framework for implementing new features.

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

### JoinSet Pattern
For managing groups of related tasks, Soma uses `tokio::task::JoinSet`:

```rust
let mut tasks = JoinSet::new();
tasks.spawn(async move {
    // Task logic here
});

while let Some(result) = tasks.join_next().await {
    // Handle task completion
}
```

**When to use**:
- For spawning and tracking multiple related background tasks
- When you need to gracefully handle task completion
- For implementing proper task supervision patterns

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
- When errors need to be categorized by type
- For providing clear error messages with structured data

### Error Propagation
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
}
```

**When to use**:
- For data that has different validity periods
- When you need clean reconfiguration boundaries
- For isolating failures between epochs

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

### Thread-Safe Caching
For performance-critical components, Soma implements thread-safe caches:

```rust
pub struct ObjectLockTable {
    // Lock table implementation
}
```

**When to use**:
- For frequently accessed data that's expensive to compute
- When you need to coordinate access across threads
- For implementing complex locking patterns

## Asynchronous Programming Patterns

### Event Loop Pattern
Many components use event loops to process messages asynchronously:

```rust
pub async fn start(mut self) {
    loop {
        tokio::select! {
            maybe_message = self.mailbox.recv() => {
                if let Some(message) = maybe_message {
                    self.handle_message(message);
                } else {
                    break;
                }
            },
            event = self.event_receiver.recv() => {
                self.handle_event(event);
            },
            Some(task_result) = self.tasks.join_next() => {
                // Handle task result
            },
        }
    }
}
```

**When to use**:
- For components that need to handle multiple message types
- When you need to coordinate between multiple channels
- For long-running background services

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

### Channel-Based Communication
Components communicate using channels to decouple producers and consumers:

```rust
let (tx, rx) = tokio::sync::mpsc::channel(32);
```

**When to use**:
- For communicating between async tasks
- When you need to decouple components
- For implementing backpressure

## Testing Patterns

### Async Testing
Tests use `tokio::test` for asynchronous test cases:

```rust
#[tokio::test]
async fn test_consensus_authority_operation() {
    // Test setup and assertions
}
```

**When to use**:
- For testing async functions and flows
- When you need a runtime for async operations
- For integration tests that span multiple components

### Mock Components
Critical interfaces have mock implementations for testing:

```rust
struct MockConsensusClient {
    // Mock implementation
}

impl ConsensusClient for MockConsensusClient {
    // Method implementations
}
```

**When to use**:
- For isolating components during testing
- When external dependencies need to be controlled
- For simulating error conditions

### Test Fixtures
Common test scenarios use fixtures for setup:

```rust
fn setup_test_authority() -> AuthorityState {
    // Setup code
}
```

**When to use**:
- For complex setup code shared between tests
- When you need consistent test environments
- For improving test readability

## Confidence: 8/10
This document provides a high-level overview of the key patterns used in Soma blockchain. While comprehensive, some patterns may evolve as the system develops, and additional patterns may emerge.

## Last Updated: 2025-03-08
