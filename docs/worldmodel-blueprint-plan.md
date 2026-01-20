# WorldModel BluePrint Plan

**Version:** 2.0  
**Date:** 2026-01-20  
**Purpose:** Consolidated plan for training a world model using BluePrint methodology as stepping stone

---

## 1. Vision and Goals

### 1.1 Core Objective
Train Qwen3-0.6B to develop structured reasoning capabilities through a two-token system:
- `<thinking>` - Abstract problem understanding and strategic planning  
- `<blueprint>` - Formal algorithmic design using BluePrint notation

This approach provides a foundation for eventual computational execution while establishing reliable structured reasoning patterns.

### 1.2 Strategic Direction
**Phase 1 (Current Focus)**: `<thinking>` → `<blueprint>` reasoning chains  
**Phase 2 (Future)**: Add `<computation>` token for executable code generation

### 1.3 Success Criteria
- Model consistently generates both thinking and blueprint tokens
- BluePrint specifications are syntactically correct and implementable
- Reasoning chains demonstrate logical progression from understanding to design
- Model generalizes to novel problem types

---

## 2. Token System Design

### 2.1 Thinking Token (`<thinking>`)
**Purpose**: Abstract problem understanding and strategic planning

**Structure**:
```
<thinking>
PROBLEM UNDERSTANDING:
- What is being asked?
- What information is provided/missing?
- What type of solution approach is needed?

STRATEGIC APPROACH:
- High-level strategy without implementation details
- Key challenges and considerations
- Connection to BluePrint concepts

DESIGN PREPARATION:
- What components will be needed?
- What relationships need to be modeled?
- What are the key constraints?
</thinking>
```

**Quality Standards**:
- Focus on understanding rather than implementation
- Identify core logical challenges
- Map problem to appropriate design patterns
- Remain conceptual and strategic

### 2.2 Blueprint Token (`<blueprint>`)
**Purpose**: Formal algorithmic design using BluePrint notation

**Structure**:
```
<blueprint>
[Valid BluePrint specification following the methodology]
</blueprint>
```

**Content Requirements**:
- Valid BluePrint syntax per docs/blueprint-prompt.md
- Complete component definitions with interfaces
- Clear data models and relationships
- Specified behaviors with preconditions/postconditions
- Error handling and edge cases

**Quality Standards**:
- Syntactically correct BluePrint notation
- Implementable design specifications
- Comprehensive coverage of the problem
- Proper abstraction levels

---

## 3. Training Approach

### 3.1 Dataset Strategy
**Current Challenge**: Previous ByteLogic training showed zero loss, suggesting data format or learning issues

**New Approach**:
1. **Curated Examples**: Hand-craft high-quality thinking→blueprint pairs
2. **Progressive Complexity**: Start simple, gradually increase sophistication
3. **Domain Diversity**: Cover multiple problem types to ensure generalization
4. **Quality over Quantity**: Focus on 1000-2000 excellent examples vs 10k mediocre ones

### 3.2 Training Methodology
Based on lessons learned from previous attempts:

**Model Configuration**:
- Base Model: Qwen3-0.6B (proven to work with your infrastructure)
- Training Method: LoRA fine-tuning (what you've already validated)
- Context Length: 768 tokens (sufficient for thinking+blueprint)

**Training Parameters** (refined from TRAINING_COMPARISON.md):
- Epochs: 15 (proven effective from previous experiments)
- Batch Size: 2 (stable on your ROCm MI50)
- Gradient Accumulation: 8 (effective batch size 16)
- Learning Rate: 2e-4 with cosine scheduling
- Warmup: 10% of total steps

**Key Improvements**:
- **Loss Function**: Monitor both overall loss AND token-specific loss for thinking/blueprint regions
- **Validation**: Real-time evaluation of BluePrint syntax correctness
- **Early Stopping**: Stop if BluePrint generation quality plateaus

### 3.3 Curriculum Design

**Stage 1: Simple Problem-Solution Pairs (300 examples)**
- Basic logical relationships
- Simple data modeling
- Single-component designs
- Focus: Establish thinking→blueprint pattern

**Stage 2: Structured Design Problems (400 examples)**
- Multi-component systems
- Interface definitions
- Basic error handling
- Focus: BluePrint syntax mastery

**Stage 3: Complex Reasoning (400 examples)**
- System architectures
- Advanced patterns from BluePrint methodology
- Real-world scenarios
- Focus: Sophisticated design thinking

**Stage 4: Novel Generalization (200 examples)**
- Edge cases and unusual problems
- Cross-domain challenges
- Creative problem-solving
- Focus: Robust generalization

---

## 4. Implementation Plan

### 4.1 Phase 1: Dataset Creation (Weeks 1-2)

**Week 1**:
- Create 50 high-quality thinking→blueprint example pairs
- Validate BluePrint syntax correctness
- Test training pipeline with small dataset
- Debug any remaining training issues

**Week 2**:
- Expand to 200 examples across multiple domains
- Implement automated BluePrint validation
- Establish quality metrics and review process
- Prepare Stage 1 curriculum

**Deliverables**:
- 200 validated training examples
- Automated BluePrint syntax checker
- Training pipeline that shows non-zero loss
- Quality evaluation metrics

### 4.2 Phase 2: Initial Training (Weeks 3-4)

**Week 3**:
- Train Stage 1 curriculum (300 examples)
- Monitor training metrics carefully
- Evaluate blueprint generation quality
- Debug and iterate on training approach

**Week 4**:
- Expand training data to 700 examples (Stages 1-2)
- Retrain with larger dataset
- Implement real-time blueprint validation during training
- Test model generalization on held-out examples

**Deliverables**:
- Working model that generates thinking+blueprint tokens
- Validated training methodology
- Performance benchmarks and quality metrics
- 700 high-quality training examples

### 4.3 Phase 3: Refinement and Scaling (Weeks 5-6)

**Week 5**:
- Complete full curriculum (1300 examples)
- Train production-quality model
- Implement comprehensive evaluation suite
- Test on diverse, novel problems

**Week 6**:
- Final model optimization
- Documentation and evaluation
- Performance analysis and lessons learned
- Plan for Phase 2 (computation token)

**Deliverables**:
- Production BluePrint model
- Complete evaluation suite
- Comprehensive documentation
- Roadmap for computation token integration

---

## 5. Technical Architecture

### 5.1 Updated Training Pipeline

```python
# Core training loop modifications
class BluePrintTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.blueprint_validator = BluePrintValidator()
        
    def compute_loss(self, batch):
        # Standard language modeling loss
        base_loss = self.model(**batch).loss
        
        # Additional blueprint syntax validation loss
        blueprint_regions = self.extract_blueprint_regions(batch)
        syntax_penalty = self.blueprint_validator.validate_batch(blueprint_regions)
        
        return base_loss + 0.1 * syntax_penalty
        
    def evaluate_generation_quality(self, examples):
        """Real-time evaluation during training"""
        results = []
        for example in examples:
            generated = self.model.generate(example['prompt'])
            
            # Check for both tokens
            has_thinking = '<thinking>' in generated and '</thinking>' in generated
            has_blueprint = '<blueprint>' in generated and '</blueprint>' in generated
            
            # Validate BluePrint syntax
            if has_blueprint:
                blueprint_content = self.extract_blueprint_content(generated)
                is_valid_syntax = self.blueprint_validator.validate(blueprint_content)
            else:
                is_valid_syntax = False
                
            results.append({
                'has_thinking': has_thinking,
                'has_blueprint': has_blueprint,
                'valid_blueprint': is_valid_syntax,
                'overall_quality': has_thinking and has_blueprint and is_valid_syntax
            })
            
        return results
```

### 5.2 Data Format

**Training Example Structure**:
```json
{
    "id": "example_001",
    "stage": "1_basic",
    "domain": "family_relationships",
    "prompt": "Design a system to track family relationships and find relatives.",
    "response": "<thinking>\nPROBLEM UNDERSTANDING:\n- Need to model family relationships\n- Should support queries about relatives\n- Must handle multiple relationship types\n\nSTRATEGIC APPROACH:\n- Create service to manage family data\n- Define relationships as explicit entities\n- Support querying by relationship type\n- Consider bidirectional relationships\n\nDESIGN PREPARATION:\n- Family member entities\n- Relationship types (parent, child, spouse, etc.)\n- Query interfaces for finding relatives\n- Data persistence requirements\n</thinking>\n\n<blueprint>\nService FamilyTracker {\n  description: \"Tracks family relationships and enables relative queries\",\n  \n  dependencies: {\n    storage: FamilyDatabase\n  },\n  \n  methods: {\n    addPerson(name: string, birthDate: date) -> PersonId,\n    addRelationship(person1: PersonId, person2: PersonId, type: RelationType) -> void,\n    findRelatives(person: PersonId, relationType: RelationType) -> PersonId[],\n    getPersonDetails(person: PersonId) -> PersonInfo\n  }\n}\n\nDatabase FamilyDatabase {\n  Table People {\n    columns: {\n      id: {type: UUID, primaryKey: true},\n      name: {type: String, required: true},\n      birthDate: {type: Date}\n    }\n  },\n  \n  Table Relationships {\n    columns: {\n      person1_id: {type: UUID, foreignKey: \"People.id\"},\n      person2_id: {type: UUID, foreignKey: \"People.id\"},\n      relationship_type: {type: String, required: true}\n    },\n    constraints: [\n      {unique: [\"person1_id\", \"person2_id\", \"relationship_type\"]}\n    ]\n  }\n}\n</blueprint>",
    "validation": {
        "has_thinking": true,
        "has_blueprint": true,
        "blueprint_syntax_valid": true,
        "thinking_quality_score": 0.9,
        "blueprint_completeness_score": 0.95
    }
}
```

### 5.3 Validation Pipeline

**BluePrint Syntax Validator**:
```python
class BluePrintValidator:
    def __init__(self):
        self.syntax_patterns = self.load_blueprint_patterns()
        
    def validate(self, blueprint_content: str) -> ValidationResult:
        """Validate BluePrint syntax and completeness"""
        results = ValidationResult()
        
        # Check basic syntax
        results.syntax_valid = self.check_syntax(blueprint_content)
        
        # Check for required components
        results.has_service_definition = 'Service ' in blueprint_content
        results.has_methods = 'methods:' in blueprint_content
        results.has_dependencies = 'dependencies:' in blueprint_content
        
        # Check BluePrint notation compliance
        results.notation_compliant = self.check_notation_compliance(blueprint_content)
        
        # Overall score
        results.overall_score = self.compute_score(results)
        
        return results
```

---

## 6. Success Metrics

### 6.1 Training Metrics
- **Loss Convergence**: Non-zero loss that decreases meaningfully
- **Token Generation Rate**: >95% of responses include both thinking and blueprint tokens
- **Blueprint Syntax Accuracy**: >90% syntactically correct BluePrint specifications
- **Thinking Quality**: Structured, logical problem understanding

### 6.2 Model Quality Metrics
- **Generalization**: Performance on novel problem types
- **Blueprint Completeness**: Generated specifications cover all necessary components
- **Design Quality**: BluePrint specs are implementable and well-structured
- **Reasoning Chain Coherence**: Thinking logically leads to blueprint design

### 6.3 Evaluation Suite
**Automated Evaluation**:
- BluePrint syntax validation
- Token presence checking
- Structural completeness analysis
- Cross-validation on held-out examples

**Human Evaluation**:
- Design quality assessment
- Implementability review
- Reasoning coherence evaluation
- Novel problem handling

---

## 7. Lessons Learned Integration

### 7.1 Previous Training Challenges
**Issue**: Zero loss in ByteLogic training  
**Solution**: Focus on simpler token patterns first, comprehensive loss monitoring

**Issue**: Inconsistent structured output  
**Solution**: Real-time validation during training, quality-focused dataset

**Issue**: Model not learning target format  
**Solution**: Progressive curriculum, extensive validation

### 7.2 Infrastructure Leverage
**Strengths to Build On**:
- ROCm training pipeline is functional
- LoRA training approach is validated
- Qwen3-0.6B works with your setup
- Monitoring and evaluation infrastructure exists

**Areas to Improve**:
- Data quality over quantity
- Real-time validation during training
- Progressive curriculum design
- Better loss function design

---

## 8. Future Roadmap (Phase 2)

### 8.1 Computation Token Integration
Once BluePrint model is working:
- Add `<computation>` token for executable code
- Integrate with ByteLogic/WASM execution
- Connect BluePrint designs to executable implementations
- Create BluePrint→ByteLogic translation examples

### 8.2 Advanced Capabilities
- Multi-turn design refinement
- Interactive design improvement
- Cross-domain knowledge transfer
- Real-world system design

---

## 9. Implementation Priority

### Immediate (Weeks 1-2)
1. Create 50 high-quality thinking→blueprint examples
2. Fix training pipeline to show meaningful loss
3. Implement BluePrint syntax validation
4. Test small-scale training run

### Short-term (Weeks 3-4) 
1. Scale to 700 training examples
2. Train working BluePrint model
3. Establish quality evaluation metrics
4. Validate generalization capabilities

### Medium-term (Weeks 5-6)
1. Complete 1300 example curriculum
2. Train production-quality model
3. Comprehensive evaluation and documentation
4. Plan Phase 2 (computation integration)

---

This plan consolidates your learnings while providing a clear path toward reliable structured reasoning through the BluePrint methodology. The focus on `<thinking>` → `<blueprint>` as a stepping stone should provide the foundation needed for eventual computational execution capabilities.