# Soma Development Index

This document serves as the main entry point to the Soma memory bank, which is organized into three primary sections:

## Core Files
- [Project Brief](core/projectbrief.md) - Overall project scope and goals
- [Technical Context](core/techContext.md) - Technology stack and environment
- [System Patterns](core/systemPatterns.md) - Architectural and coding patterns
- [Product Context](core/productContext.md) - Product vision and user personas
- [Active Context](core/activeContext.md) - Current development focus
- [Progress](core/progress.md) - Development status and milestone tracking
- [Documentation](core/documentation.md) - Documentation overview and best practices

## Knowledge Base
- [Cross-Module Communication](knowledge/cross_module_communication.md) - Inter-module interaction patterns
- [Data Flow](knowledge/data_flow.md) - Overall data flow architecture
  - [Checkpoint Processing](knowledge/data_flow/checkpoint_processing.md)
  - [Concurrency Model](knowledge/data_flow/concurrency_model.md)
  - [Cross-Module Relationships](knowledge/data_flow/cross_module_relationships.md)
  - [Dependency Management](knowledge/data_flow/dependency_management.md)
  - [Object Model](knowledge/data_flow/object_model.md)
  - [Shared Object Processing](knowledge/data_flow/shared_object_processing.md)
  - [Transaction Lifecycle](knowledge/data_flow/transaction_lifecycle.md)
- [Epoch Management](knowledge/epoch_management.md) - Epoch transitions and boundaries
- [Error Handling](knowledge/error_handling.md) - Error handling patterns
- [Security Model](knowledge/security_model.md) - Security considerations and patterns
- [Storage Model](knowledge/storage_model.md) - Storage architecture and patterns
- [Thread Safety Patterns](knowledge/thread_safety_patterns.md) - Thread safety approaches
- [Type System](knowledge/type_system.md) - Type system design and usage

## Module Documentation
- [Authority](modules/authority.md) - State management and transaction processing
  - [Index](modules/authority/index.md) - Module overview
  - [Module Structure](modules/authority/module_structure.md) - Component organization
  - [Aggregator & Quorum Driver](modules/authority/aggregator_quorum_driver.md)
  - [Caching Layer](modules/authority/caching_layer.md)
  - [Checkpoint Processing](modules/authority/checkpoint_processing.md)
  - [Commit Processing](modules/authority/commit_processing.md)
  - [Consensus Handler](modules/authority/consensus_handler.md)
  - [Consensus Integration](modules/authority/consensus_integration.md)
  - [Consensus Quarantine](modules/authority/consensus_quarantine.md)
  - [Execution Driver](modules/authority/execution_driver.md)
  - [Mysticeti Integration](modules/authority/mysticeti_integration.md)
  - [Orchestrator](modules/authority/orchestrator.md)
  - [Reconfiguration](modules/authority/reconfiguration.md)
  - [Server Components](modules/authority/server_components.md)
  - [Service Implementation](modules/authority/service_implementation.md)
  - [State Accumulator](modules/authority/state_accumulator.md)
  - [State Management](modules/authority/state_management.md)
  - [State Sync Store](modules/authority/state_sync_store.md)
  - [Thread Safety](modules/authority/thread_safety.md)
  - [Transaction Processing](modules/authority/transaction_processing.md)
- [Consensus](modules/consensus.md) - Byzantine Fault Tolerant agreement protocol
  - [Index](modules/consensus/index.md) - Module overview
  - [Module Structure](modules/consensus/module_structure.md) - Component organization
  - [Block Processing](modules/consensus/block_processing.md)
  - [Commit Observer](modules/consensus/commit_observer.md)
  - [Consensus Workflow](modules/consensus/consensus_workflow.md)
  - [Thread Safety](modules/consensus/thread_safety.md)
- [Node](modules/node.md) - Lifecycle and service orchestration
  - [Index](modules/node/index.md) - Module overview
  - [Module Structure](modules/node/module_structure.md) - Component organization
  - [Lifecycle Management](modules/node/lifecycle_management.md)
  - [Reconfiguration](modules/node/reconfiguration.md)
  - [Service Orchestration](modules/node/service_orchestration.md)
  - [Thread Safety](modules/node/thread_safety.md)
- [P2P](modules/p2p.md) - Network discovery and state synchronization
  - [Index](modules/p2p/index.md) - Module overview
  - [Module Structure](modules/p2p/module_structure.md) - Component organization
  - [Checkpoint Processing](modules/p2p/checkpoint_processing.md)
  - [Discovery](modules/p2p/discovery.md)
  - [State Sync](modules/p2p/state_sync.md)
  - [Thread Safety](modules/p2p/thread_safety.md)
