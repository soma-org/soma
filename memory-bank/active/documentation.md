# Memory Bank Documentation Strategy

## Overview
The Memory Bank is Soma blockchain's knowledge repository designed specifically for efficient agent collaboration. Rather than documenting every file exhaustively, we maintain a structured knowledge system that agents can efficiently consume before starting tasks. This approach reduces context window overhead, improves cross-component understanding, and facilitates smoother collaboration between multiple AI agents.

## Memory Bank Philosophy

### Core Principles
1. **Knowledge Over Completeness**: Focus on helping agents understand how things work, not documenting every line of code
2. **Workflow Documentation**: Explain processes that span multiple files to capture cross-component relationships
3. **Component Interaction**: Document how modules interact rather than isolating documentation by file
4. **Reduced Context Overhead**: Organize information efficiently to minimize token usage during agent context loading
5. **Practical Over Exhaustive**: Prioritize practical understanding that enables implementation work

### Memory Bank Structure

```
memory-bank/
├── core/                         # Essential project context
│   ├── projectbrief.md           # High-level project overview
│   ├── productContext.md         # Why Soma exists and what problems it solves
│   ├── systemPatterns.md         # Key architectural patterns
│   ├── techContext.md            # Technology stack and constraints
│   ├── activeContext.md          # Current development focus
│   └── progress.md               # Project status and milestones
│
├── knowledge/                    # Cross-component knowledge
│   ├── data_flow.md              # Transaction lifecycle, object model, processing
│   ├── system_coordination.md    # Consensus, committee, epoch reconfiguration
│   ├── deployment.md             # Node types, genesis, lifecycle
│   ├── type_system.md            # Core types and their relationships
│   └── error_handling.md         # Error propagation and recovery patterns
│
├── modules/                      # Module-specific knowledge
│   ├── authority.md              # Authority module structure and concepts
│   ├── consensus.md              # Consensus module structure and concepts
│   ├── node.md                   # Node module structure and concepts
│   └── p2p.md                    # P2P module structure and concepts
│
├── active/                       # Current development context
│   ├── current_tasks.md          # Tasks currently in progress
│   └── module_updates/           # Per-module recent changes
│
└── reference/                    # Supporting resources
    ├── glossary.md               # Terminology definitions
    ├── agent_workflows.md        # Agent collaboration guides
    └── confidence_log.md         # Historical confidence ratings
```

## Three Pillars Knowledge Framework
Our documentation strategy focuses on three core knowledge pillars that provide maximum understanding with minimum overhead:

### Pillar 1: Core Data Flow
- Combines transaction lifecycle, object model/ownership, and commit processing
- Focuses on the complete data path through the system
- Primary concern for developers implementing features or fixes

### Pillar 2: System Coordination 
- Combines consensus workflow, validator committee, and epoch reconfiguration
- Explains how the system maintains agreement and evolves over time
- Critical for understanding BFT properties and system governance

### Pillar 3: Deployment & Operations
- Combines node types (fullnode vs. validator), genesis/bootstrap, and node lifecycle
- Focuses on how the system is deployed and maintained
- Essential for operators and SREs managing the network

## Memory Bank Document Templates

### Core Document Template
```markdown
# [Document Title]

## Purpose and Scope
[Brief explanation of what this document covers and why it's important]

## Key Components
- **Component A**: [Brief description]
- **Component B**: [Brief description]
- **Component C**: [Brief description]

## Component Relationships Diagram
[Mermaid diagram showing relationships]

## Detailed Information
[Main content sections]

## Confidence: [1-10]
[Statement about the confidence level and any caveats or limitations]

## Last Updated: [Date]
```

### Knowledge Document Template
```markdown
# [Knowledge Area]

## Purpose and Scope
[Brief explanation of this knowledge area and why it matters]

## Key Components
[List of relevant components and brief descriptions]

## Component Relationships
[Mermaid diagram showing relationships]

## Detailed Explanations
[Multiple sections explaining concepts, workflows, etc.]

## Usage Examples
[Example code or scenarios showing concepts in action]

## Relationship to Other Components
[How this knowledge area connects to other parts of the system]

## Key Insights for Developers
[Bullet points with critical insights]

## Confidence: [1-10]
[Statement about confidence level]

## Last Updated: [Date]
```

### Module Document Template
```markdown
# [Module Name]

## Purpose and Scope
[Brief explanation of what this module does]

## Key Components
[List major components and their roles]

## Architecture Diagram
[Mermaid diagram showing module structure]

## Primary Workflows
[Explain main processes implemented by this module]

## Key Interfaces
[Document important interfaces with other modules]

## Design Patterns
[Explain patterns used in this module]

## Confidence: [1-10]
[Statement about confidence level]

## Last Updated: [Date]
```

## Memory Bank Maintenance Guidelines

### When to Update Memory Bank Documents
Documents should be updated when:
1. Implementing significant new features
2. Making architectural changes
3. Refactoring core components
4. Discovering better ways to explain concepts
5. Updating module interfaces
6. Changing workflow processes
7. Before transitioning between AI agents (Cursor to Cline)

### Document Update Process
1. Review existing document for accuracy
2. Update technical content as needed
3. Update component relationship diagrams 
4. Update confidence rating (1-10 scale)
5. Update "Last Updated" timestamp
6. Add note to `memory-bank/active/current_tasks.md` about the update

### Best Practices for Memory Bank Documents
1. **Focus on Knowledge Transfer**: Prioritize information that helps agents understand critical concepts
2. **Explain Why, Not Just How**: Include rationale for design decisions
3. **Use Visual Diagrams**: Mermaid diagrams help clarify complex relationships
4. **Include Examples**: Concrete examples make abstract concepts clearer
5. **Cross-Reference**: Link to related documents to help navigate the knowledge base
6. **Confidence Rating**: Always include a 1-10 confidence rating that honestly assesses completeness
7. **Agent Handoff Ready**: Ensure documents support smooth transitions between agents

## Integration with Code Documentation
The Memory Bank complements, rather than replaces, inline code documentation:

1. **Memory Bank**: Cross-component knowledge, architectural patterns, workflows spanning multiple files
2. **Inline Documentation**: 
   - Module-level documentation (//! comments)
   - Function-level documentation (/// comments) 
   - Implementation details specific to a file

When adding inline documentation, reference relevant Memory Bank documents when discussing cross-component aspects.

## Current Focus and Progress
We are currently implementing the Memory Bank in phases:

### Phase 1: Foundation (Immediate Focus)
- Create Memory Bank directory structure
- Extract knowledge from existing documentation:
  - Core: projectbrief.md, systemPatterns.md
  - Knowledge: type_system.md, error_handling.md
- Update documentation strategy files

### Phase 2: Key Knowledge Documents
- Create data_flow.md focusing on transaction lifecycle
- Create authority.md from existing module documentation
- Create progress.md documenting current milestone status

### Phase 3: Completion
- Create remaining essential documents
- Ensure all documents have confidence ratings
- Complete module-specific documents for all four modules

## Recent Updates
*2025-03-08*: Created Memory Bank structure and established documentation strategy with three knowledge pillars - Cline (Confidence: 9/10)

*2025-03-08*: Extracted systemPatterns.md and type_system.md from existing documentation - Cline (Confidence: 8/10)

*2025-03-08*: Created error_handling.md knowledge document - Cline (Confidence: 9/10)

*2025-03-07*: Completed documentation for authority/src/tx_manager.rs with comprehensive coverage of transaction management - Cline (Confidence: 9/10)

*2025-03-07*: Completed Phase 1 documentation with 100% coverage of high-priority Types module files - Cline (Confidence: 9/10)
