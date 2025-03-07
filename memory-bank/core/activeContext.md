# Soma Active Context

## Current Development Focus
Based on the milestones document, the current development priorities are:

1. **Transaction Types Implementation**
   - Balances and transfers
     - Transaction conflict resolution
     - Transaction ordering by dependencies
   - Staking mechanisms
     - Separation between Encoder and Validator Stake
     - Leader Scoring and Tally Rules
   - Shard-related transactions
     - Escrow payment management
     - Result submission and verification

2. **Node and Shard Integration**
   - Encoder committee management
     - Committee selection algorithms
     - Rotation and handoff procedures
   - Probe hashes in EndOfEpochData
     - Aggregate signature validation
     - Threshold verification
   - Transaction proof serving
     - Merkle proof generation
     - Verification mechanisms

3. **Core Infrastructure**
   - State synchronization
     - Incremental state sync
     - Snapshot-based recovery
   - P2P networking
     - Peer discovery and connection management
     - Message propagation optimizations
   - Consensus mechanism
     - Integration with Mysticeti
     - Epoch boundary handling

## Immediate Next Steps
1. Implement transaction type structures and validation
   - Define core transaction traits
   - Implement specific transaction types
   - Create comprehensive validation rules
   - Design serialization format

2. Develop core state transition logic
   - Create atomic state update mechanism
   - Implement rollback capability
   - Design deterministic execution flow
   - Add comprehensive error handling

3. Integrate encoder committee management
   - Implement stake-weighted selection
   - Create committee rotation logic
   - Design Byzantine fault tolerance mechanisms
   - Build verification for committee proofs

4. Implement basic RPC endpoints for transaction submission
   - Define API interface
   - Create request validation
   - Implement response formatting
   - Add authentication and rate limiting

## Current Challenges
1. Ensuring atomic state transitions across distributed components
   - Current approach: Transaction validation before processing
   - Alternative: Two-phase commit protocol
   - Exploring: Optimistic execution with rollback

2. Maintaining performance with complex validation logic
   - Current approach: Parallel validation with Tokio
   - Bottleneck: Cryptographic operations
   - Investigation: Batched signature verification

3. Designing a flexible yet secure staking mechanism
   - Current approach: Separate stake pools with delegation
   - Challenge: Ensuring fair committee selection
   - Trade-off: Security vs. participation barriers

4. Balancing between development speed and code quality
   - Current approach: Test-driven development
   - Challenge: Comprehensive testing of async code
   - Strategy: Focused end-to-end tests for critical paths

## Recent Decisions
1. Use Tokio for all async operations
   - Rationale: Comprehensive ecosystem and tooling
   - Alternatives considered: async-std, smol
   - Implementation: Standardize on tokio::spawn pattern

2. Implement a modular transaction processing pipeline
   - Design: Separation of validation and execution
   - Benefit: Better testability and error handling
   - Implementation: Trait-based plugin architecture

3. Follow a staged deployment strategy for features
   - Approach: Core consensus first, then economic model
   - Rationale: Ensure stability before complexity
   - Timeline: Align with milestone document

4. Prioritize testing infrastructure early in development
   - Strategy: Develop test-cluster alongside core code
   - Implementation: Docker-based multi-node testing
   - Metrics: Coverage and scenario-based testing
