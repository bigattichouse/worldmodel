# WorldModel: BluePrint Reasoning for LLMs

A system for training small language models to develop structured reasoning capabilities through **BluePrint methodology** - progressing from abstract thinking to formal algorithmic design.

## Overview

This project implements a **two-phase approach** to world modeling:

**Phase 1 (Current)**: `<thinking>` → `<blueprint>` reasoning chains  
**Phase 2 (Future)**: Add `<computation>` token for executable code

### What Makes This Different

Instead of jumping directly from natural language to code, the model learns to:
1. **Think strategically** using `<thinking>` tags for problem understanding
2. **Design formally** using `<blueprint>` tags with structured specifications
3. **Build incrementally** toward computational capabilities

## Quick Start

### Prerequisites
- Python 3.8+
- ROCm 7.1.1+ (for AMD GPU training)
- 8-16GB GPU memory recommended

### Installation
```bash
git clone [repository]
cd worldmodel
pip install -r requirements.txt
```

### Training a BluePrint Model
```bash
# Set up ROCm environment (for AMD GPUs)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0

# Train with BluePrint examples
python train_blueprint_model.py \
  --epochs 15 \
  --batch-size 2 \
  --model-name Qwen3-0.6B
```

## How It Works

### Training Format
The model learns to generate structured reasoning:

```
User: Design a system to track family relationships

<thinking>
PROBLEM UNDERSTANDING:
- Need to model family relationships between people
- Should support queries about relatives
- Must handle multiple relationship types

STRATEGIC APPROACH:
- Create service to manage family data  
- Define relationships as explicit entities
- Support querying by relationship type
- Consider bidirectional relationships

DESIGN PREPARATION:
- Family member entities with basic info
- Relationship types (parent, child, spouse, etc.)
- Query interfaces for finding relatives
- Data persistence requirements
</thinking>

<blueprint>
Service FamilyTracker {
  description: "Tracks family relationships and enables relative queries",
  
  methods: {
    addPerson(name: string, birthDate: date) -> PersonId,
    addRelationship(person1: PersonId, person2: PersonId, type: RelationType) -> void,
    findRelatives(person: PersonId, relationType: RelationType) -> PersonId[]
  }
}

Database FamilyDatabase {
  Table People {
    columns: {
      id: {type: UUID, primaryKey: true},
      name: {type: String, required: true},
      birthDate: {type: Date}
    }
  },
  
  Table Relationships {
    columns: {
      person1_id: {type: UUID, foreignKey: "People.id"},
      person2_id: {type: UUID, foreignKey: "People.id"},
      relationship_type: {type: String, required: true}
    }
  }
}
</blueprint>
```

### Why BluePrint First?

**Traditional Approach Problems:**
- Direct natural language → code is too big a leap
- Models struggle with structured reasoning
- High failure rates on complex problems

**BluePrint Approach Benefits:**
- **Incremental learning**: Understanding → Design → Implementation
- **Traceable reasoning**: Each step can be reviewed and debugged
- **Higher success rates**: Smaller, more manageable cognitive steps
- **Foundation building**: Creates base for future computational layers

## Project Structure

```
worldmodel/
├── docs/
│   ├── worldmodel-blueprint-plan.md    # 🎯 Main implementation plan
│   ├── worldmodel-bytelogic.md         # Phase 2 roadmap  
│   ├── blueprint-prompt.md             # BluePrint methodology
│   ├── rocm/                          # ROCm GPU setup guides
│   └── archive/                       # Phase 2 materials
├── src/                               # Core implementation (coming)
├── training/                          # Training datasets  
└── README.md                          # This file
```

## Current Status

**✅ Completed:**
- BluePrint methodology defined
- Phase 1 implementation plan created
- ROCm training infrastructure validated
- Documentation organized

**🚧 In Progress:**
- Training dataset creation (Phase 1)
- BluePrint model training pipeline

**📋 Next Steps:**
1. Create 200+ high-quality thinking→blueprint examples
2. Train and validate BluePrint model
3. Establish quality evaluation metrics
4. Plan Phase 2 (computational execution)

## Phase Roadmap

### Phase 1: BluePrint Reasoning (Weeks 1-6)
- **Goal**: Reliable `<thinking>` → `<blueprint>` generation
- **Success Criteria**: >90% valid BluePrint specifications
- **Foundation**: Structured reasoning without execution

### Phase 2: Computational Execution (Future)
- **Goal**: Add `<computation>` token for executable code
- **Prerequisite**: Phase 1 success
- **Integration**: BluePrint designs → executable implementations

## Success Metrics

**Phase 1 Targets:**
- BluePrint syntax accuracy: >90%
- Token generation consistency: >95%
- Reasoning chain coherence: Human evaluation
- Generalization: Novel problem handling

## Contributing

This project focuses on **research and experimentation** in structured reasoning for LLMs. The BluePrint approach is designed to be:

- **Simple enough** for small models to learn reliably
- **Expressive enough** for practical problem-solving  
- **Structured enough** to enable systematic evaluation
- **Extensible enough** to support future computational layers

## Documentation

- 📖 [Implementation Plan](docs/worldmodel-blueprint-plan.md) - Complete Phase 1 roadmap
- 🔧 [BluePrint Methodology](docs/blueprint-prompt.md) - Core design principles
- 🚀 [Phase 2 Vision](docs/worldmodel-bytelogic.md) - Computational execution roadmap
- ⚡ [ROCm Setup](docs/rocm/) - AMD GPU training guides

---

**Vision**: Train small models to think systematically and design formally, creating a foundation for reliable computational reasoning capabilities.