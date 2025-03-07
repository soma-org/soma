# Soma Technical Context

## Technology Stack

### Programming Language
- **Rust**: Primary language for all components
  - Strong memory safety guarantees
  - Zero-cost abstractions
  - Trait-based system
  - Pattern matching
  - Type inference
  - Minimal runtime

### Core Libraries & Dependencies
- **Tokio**: Async runtime for concurrent operations
  - Task spawning and management with JoinSet
  - I/O multiplexing with tokio::select!
  - Synchronization primitives for async contexts
  - Testing utilities for async code

- **Tonic**: gRPC framework based on HTTP/2
  - Service definition and implementation for inter-node communication
  - Client and server code generation
  - Channel-based communication

- **Parking_lot**: Efficient synchronization primitives
  - RwLock for concurrent reads with exclusive writes
  - Mutex for exclusive access
  - Performance improvements over standard library versions

- **Arc-swap**: Hot-swappable atomic reference counting
  - Used for swapping epoch stores during reconfiguration
  - Thread-safe updates of shared state

- **BCS**: Binary Canonical Serialization
  - Deterministic serialization for blockchain data
  - Used for transaction and block serialization

- **Fastcrypto**: Cryptographic library
  - Verification of signatures
  - Key management for authority operations
  - Used for consensus signatures and transaction validation

- **Tracing**: Structured, event-based diagnostic information
  - Hierarchical spans for tracking operations
  - Detailed logging across components

- **Supporting Libraries**:
  - **thiserror**: Error type definitions and propagation
  - **anyhow**: Flexible error handling
  - **futures**: Asynchronous operation composition
  - **itertools**: Advanced iterator operations
  - **bytes**: Efficient byte buffer handling
  - **lru**: Least-recently-used cache implementation
  - **tap**: Functional programming utilities

### Project Structure
- **/authority**: Validator and authority management
  - **state.rs**: Central authority state management
  - **epoch_store.rs**: Per-epoch state and storage
  - **handler.rs**: Consensus transaction handling
  - **commit/**: Transaction commit and execution
  - **manager/**: Consensus manager implementation
  - **cache/**: Object and transaction caching

- **/node**: Core blockchain node implementation
  - **lib.rs**: Main SomaNode implementation
  - **handle.rs**: Node handle for external interaction

- **/consensus**: Mysticeti consensus implementation
  - **authority.rs**: ConsensusAuthority implementation
  - **core.rs**: Core consensus logic
  - **core_thread.rs**: Consensus thread management
  - **committer/**: Transaction commitment to state
  - **network/**: Consensus network communication

- **/p2p**: Peer-to-peer networking layer
  - **discovery/**: Peer discovery implementation
  - **state_sync/**: State synchronization between nodes
  - **builder.rs**: P2P network builder

- **/types**: Core data types and structures
  - **base.rs**: Fundamental type definitions
  - **committee.rs**: Validator committee management
  - **transaction.rs**: Transaction structure and validation
  - **consensus/**: Consensus-specific types
  - **effects/**: Transaction effects definitions
  - **storage/**: Storage interface definitions

- **/utils**: Shared utilities and helpers
  - **notify_once.rs**: One-time notification utility
  - **notify_read.rs**: Notification for reads
  - **agg.rs**: Aggregation utilities

## System Components

### Authority Components
- **AuthorityState**: Central state manager for a validator node
  - Transaction processing and execution
  - Certificate handling and verification
  - State management and transitions

- **AuthorityPerEpochStore**: Per-epoch storage and state
  - Epoch-specific transaction handling
  - Consensus interaction
  - Reconfiguration state management

- **TransactionManager**: Transaction handling and scheduling
  - Enqueues transactions for execution
  - Manages pending certificate execution
  - Tracks transaction dependencies

### Consensus Components
- **ConsensusAuthority**: Main consensus implementation
  - Block production and verification
  - Leader selection and scheduling
  - Commit management
  - Network communication

- **Core**: Consensus core logic
  - ThresholdClock for round management
  - Block creation and verification
  - Leader block handling

- **CommitSyncer**: Synchronizes commits between nodes
  - Fetches missing commits
  - Verifies commit validity
  - Updates local state

### P2P Components
- **DiscoveryEventLoop**: Peer discovery and management
  - Finds and connects to new peers
  - Maintains peer information
  - Manages connections to peers

- **StateSyncEventLoop**: State synchronization
  - Fetches missing state from peers
  - Verifies fetched state
  - Updates local state

### Node Components
- **SomaNode**: Main node implementation
  - Component lifecycle management
  - Reconfiguration handling
  - Epoch transition coordination

## Development Environment
- **Cargo**: Rust package manager
- **Rustfmt & Clippy**: Code formatting and linting
- **Github Actions**: CI/CD pipeline
  - Automated testing
  - Linting and formatting checks
  - Deployment automation

## Technical Constraints
- Byzantine fault tolerance requirements:
  - System functions correctly with up to f faulty nodes out of 3f+1 total
  - Validators must reach consensus despite node failures
  - System must detect and handle malicious behavior

- Performance considerations:
  - Efficient transaction processing
  - Minimized consensus latency
  - Optimized state synchronization

- Safety guarantees:
  - No double-spending or invalid state transitions
  - Consistent state across honest nodes
  - Deterministic transaction execution

- Liveness guarantees:
  - System continues to make progress despite failures
  - Reconfiguration completes successfully
  - Transactions are eventually processed
