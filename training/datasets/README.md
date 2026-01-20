# BluePrint Training Datasets

This directory contains structured training data for teaching LLMs to generate BluePrint specifications following the methodology in `docs/blueprint-prompt.md`.

## Directory Structure

```
datasets/
├── basic_systems/              # 60 examples - Foundation concepts
│   ├── crud_operations.jsonl  # 15 examples - Create, Read, Update, Delete
│   ├── simple_calculations.jsonl # 15 examples - Math and calculations  
│   ├── data_conversions.jsonl # 15 examples - Format/unit conversions
│   └── basic_validation.jsonl # 15 examples - Input validation
├── business_logic/             # 80 examples - Business domains
│   ├── financial_services.jsonl # 20 examples
│   ├── e_commerce.jsonl       # 20 examples  
│   ├── hr_management.jsonl    # 20 examples
│   └── workflow_systems.jsonl # 20 examples
├── technical_systems/          # 80 examples - Infrastructure
│   ├── authentication.jsonl   # 20 examples
│   ├── data_processing.jsonl  # 20 examples
│   ├── infrastructure.jsonl   # 20 examples
│   └── integration_apis.jsonl # 20 examples
├── domain_specific/            # 60 examples - Specialized domains
│   ├── healthcare.jsonl       # 15 examples
│   ├── education.jsonl        # 15 examples
│   ├── logistics.jsonl        # 15 examples
│   └── manufacturing.jsonl    # 15 examples
├── advanced_patterns/          # 40 examples - Complex architectures
│   ├── microservices.jsonl    # 10 examples
│   ├── event_driven.jsonl     # 10 examples
│   ├── distributed_systems.jsonl # 10 examples
│   └── real_time_systems.jsonl # 10 examples
└── validation/                 # 20 examples - Edge cases
    ├── edge_cases.jsonl        # 10 examples
    └── stress_scenarios.jsonl  # 10 examples
```

**Total: ~340 training examples**

## Data Format

Each `.jsonl` file contains one JSON object per line with this structure:

```json
{
  "id": "unique_identifier",
  "category": "main_category",
  "subcategory": "specific_subcategory", 
  "difficulty": "basic|intermediate|advanced",
  "user_query": "Human intent description",
  "response": "<thinking>...</thinking>\\n\\n<blueprint>...</blueprint>"
}
```

### Example Entry

```json
{
  "id": "crud_001",
  "category": "basic_systems",
  "subcategory": "crud_operations",
  "difficulty": "basic",
  "user_query": "Design a book library catalog system",
  "response": "<thinking>\\nPROBLEM UNDERSTANDING:\\n- Need system to manage library book catalog...\\n</thinking>\\n\\n<blueprint>\\nService LibraryCatalog {\\n  description: \\\"Manages library book catalog\\\",\\n  methods: {...}\\n}\\n</blueprint>"
}
```

## BluePrint Methodology Requirements

All examples must follow the BluePrint collaborative specification framework:

### 1. Thinking Structure
```
<thinking>
PROBLEM UNDERSTANDING:
- What is being asked?
- Key requirements and constraints
- Scope and boundaries

STRATEGIC APPROACH:
- High-level solution strategy
- Key design decisions
- Integration considerations

DESIGN PREPARATION:
- Components needed
- Relationships and dependencies
- Data modeling requirements
</thinking>
```

### 2. BluePrint Notation
Must use proper BluePrint syntax:

- **Services**: `Service ServiceName { ... }`
- **Databases**: `Database DbName { Table TableName { ... } }`
- **Types**: `Type TypeName { ... }`
- **Operations**: `Operation OpName { ... }`
- **Scenarios**: `Scenario scenario_name { ... }`
- **Validation**: `Validation RuleName { ... }`

### 3. Required Components

Each BluePrint should include:
- ✅ **Service definition** with description and methods
- ✅ **Method signatures** with preconditions/postconditions/errors
- ✅ **Data models** (Types, Database tables)
- ✅ **Validation rules** where appropriate
- ✅ **Dependencies** clearly specified

### 4. Quality Standards

- **Thinking quality**: Strategic reasoning, not implementation details
- **BluePrint completeness**: All necessary components specified
- **Syntax correctness**: Valid BluePrint notation
- **Real-world applicability**: Implementable specifications
- **Appropriate complexity**: Matches difficulty level

## Generation Guidelines for LLMs

When creating new examples:

### 1. Choose Appropriate Scenarios
- Select realistic, practical systems
- Ensure variety within each subcategory  
- Balance simple and complex requirements
- Consider edge cases and error conditions

### 2. Structure Thinking Properly
- Start with problem understanding, not solutions
- Keep strategic (avoid implementation details)
- Connect problem to BluePrint concepts
- Show clear reasoning progression

### 3. Create Complete BluePrint Specs
- Include all necessary Services, Types, Databases
- Specify method contracts clearly
- Add proper validation and error handling
- Make specifications implementable

### 4. Follow Difficulty Progression
- **Basic**: Single service, simple operations
- **Intermediate**: Multi-service, business logic
- **Advanced**: Complex architectures, integration

### 5. Maintain Consistency
- Use consistent naming conventions
- Follow established patterns from existing examples
- Ensure BluePrint syntax is valid
- Keep similar complexity within subcategories

## Training Usage

This dataset structure supports:

1. **Progressive training**: Start with basic_systems, advance through categories
2. **Selective training**: Focus on specific domains or difficulty levels
3. **Balanced learning**: Mix different categories for comprehensive coverage
4. **Quality validation**: Automatic syntax checking and completeness verification

## Validation Checklist

Before adding new examples, verify:

- [ ] JSON syntax is valid
- [ ] Required fields present (id, category, subcategory, difficulty, user_query, response)
- [ ] Thinking structure follows template
- [ ] BluePrint syntax is correct
- [ ] Specifications are complete and implementable
- [ ] Difficulty matches category expectations
- [ ] Unique ID within subcategory

## Contributing

To expand this dataset:

1. Choose an unfilled subcategory from the catalog
2. Follow the format and quality standards above
3. Generate 10-15 examples per subcategory file
4. Validate syntax and completeness
5. Test with sample scenarios

This systematic approach ensures high-quality training data that teaches LLMs proper BluePrint methodology and collaborative specification skills.