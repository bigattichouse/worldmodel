# BluePrint Training Data Design

**Date:** 2026-01-20  
**Purpose:** Define optimal training data format for `<thinking>` → `<blueprint>` model

---

## Current Approach Analysis

Your existing ByteLogic dataset shows this pattern:
```json
{
  "user_query": "Who are Alice's children?",
  "complete_response": "<thinking>...</thinking>\n\n<pseudocode>...</pseudocode>\n\n<computation>...</computation>"
}
```

### Issues with Current Format

1. **Three tokens instead of two**: `<thinking>` → `<pseudocode>` → `<computation>`
2. **Redundant content**: `<pseudocode>` and `<computation>` are nearly identical
3. **Mixed paradigms**: ByteLogic-specific content in BluePrint phase
4. **Complex learning task**: Model must learn 3 different token types simultaneously

---

## Proposed BluePrint Training Format

### Design Principles

1. **Two-token simplicity**: `<thinking>` → `<blueprint>` only (Phase 1)
2. **Clear separation**: Abstract reasoning vs. formal design
3. **BluePrint-native**: Use proper BluePrint notation, not ByteLogic
4. **Progressive complexity**: Start simple, build toward sophisticated designs

### Ideal Training Example

```json
{
  "id": "blueprint_001",
  "category": "task_management", 
  "difficulty": "intermediate",
  "user_query": "Design a task management system for a small team",
  "response": "<thinking>\nPROBLEM UNDERSTANDING:\n- Need system to track tasks across team members\n- Should handle task assignment, status tracking, deadlines\n- Requires collaboration features and progress visibility\n- Must support different task priorities and categories\n\nSTRATEGIC APPROACH:\n- Create task management service with clear ownership model\n- Design flexible task entity with status workflow\n- Support team collaboration through assignments and comments\n- Enable progress tracking and deadline management\n- Consider notification system for important updates\n\nDESIGN PREPARATION:\n- Task entities with assignees, due dates, priorities\n- Team member management and role definitions\n- Status workflow (todo → in_progress → review → done)\n- Comment/collaboration system on tasks\n- Notification service for updates and deadlines\n</thinking>\n\n<blueprint>\nService TaskManager {\n  description: \"Manages tasks and assignments for small teams\",\n  \n  dependencies: {\n    storage: TaskDatabase,\n    notifications: NotificationService,\n    auth: AuthenticationService\n  },\n  \n  methods: {\n    createTask(title: string, description: string, assigneeId: UserId, dueDate: date, priority: Priority) -> TaskId {\n      preconditions: [title not empty, assignee exists, valid due date],\n      postconditions: [task created, assignee notified],\n      errors: [InvalidInput, UserNotFound, PastDueDate]\n    },\n    \n    updateTaskStatus(taskId: TaskId, newStatus: TaskStatus, userId: UserId) -> void {\n      preconditions: [task exists, user has permission, valid status transition],\n      postconditions: [status updated, stakeholders notified],\n      errors: [TaskNotFound, InvalidPermission, InvalidTransition]\n    },\n    \n    assignTask(taskId: TaskId, newAssigneeId: UserId, assignerId: UserId) -> void {\n      preconditions: [task exists, assigner has permission, assignee exists],\n      postconditions: [assignment updated, both parties notified]\n    },\n    \n    addComment(taskId: TaskId, userId: UserId, comment: string) -> CommentId {\n      preconditions: [task exists, user has access, comment not empty],\n      postconditions: [comment added, task participants notified]\n    },\n    \n    getTasksByAssignee(assigneeId: UserId, status: TaskStatus?) -> Task[] {\n      preconditions: [user exists],\n      postconditions: [returns filtered task list]\n    }\n  }\n}\n\nDatabase TaskDatabase {\n  Table Tasks {\n    columns: {\n      id: {type: UUID, primaryKey: true},\n      title: {type: String, required: true, maxLength: 200},\n      description: {type: Text},\n      assignee_id: {type: UUID, foreignKey: \"Users.id\"},\n      creator_id: {type: UUID, foreignKey: \"Users.id\", required: true},\n      status: {type: String, enum: [\"todo\", \"in_progress\", \"review\", \"done\"]},\n      priority: {type: String, enum: [\"low\", \"medium\", \"high\", \"urgent\"]},\n      due_date: {type: Date},\n      created_at: {type: Timestamp, default: \"now()\"},\n      updated_at: {type: Timestamp, default: \"now()\"}\n    },\n    indexes: [\n      {name: \"idx_assignee_status\", columns: [\"assignee_id\", \"status\"]},\n      {name: \"idx_due_date\", columns: [\"due_date\"]}\n    ]\n  },\n  \n  Table Comments {\n    columns: {\n      id: {type: UUID, primaryKey: true},\n      task_id: {type: UUID, foreignKey: \"Tasks.id\", required: true},\n      user_id: {type: UUID, foreignKey: \"Users.id\", required: true},\n      content: {type: Text, required: true},\n      created_at: {type: Timestamp, default: \"now()\"}\n    }\n  },\n  \n  Table Users {\n    columns: {\n      id: {type: UUID, primaryKey: true},\n      name: {type: String, required: true},\n      email: {type: String, unique: true, required: true},\n      role: {type: String, enum: [\"member\", \"lead\", \"admin\"]}\n    }\n  }\n}\n\nBehavior TaskStatusWorkflow {\n  transitions: {\n    todo -> in_progress: {condition: \"assignee accepts task\"},\n    in_progress -> review: {condition: \"work completed, needs review\"},\n    in_progress -> todo: {condition: \"work paused or blocked\"},\n    review -> done: {condition: \"review approved\"},\n    review -> in_progress: {condition: \"revisions requested\"},\n    any -> todo: {condition: \"task reset by lead/admin\"}\n  },\n  \n  notifications: {\n    status_change: \"notify assignee and creator\",\n    approaching_deadline: \"notify assignee 24h before due date\",\n    overdue: \"notify assignee and team lead\"\n  }\n}\n\nIntegration NotificationFlow {\n  events: [\n    {trigger: \"task assigned\", recipients: [\"assignee\"], message: \"New task assigned\"},\n    {trigger: \"status changed\", recipients: [\"creator\", \"assignee\"], message: \"Task status updated\"},\n    {trigger: \"comment added\", recipients: [\"task participants\"], message: \"New comment on task\"},\n    {trigger: \"deadline approaching\", recipients: [\"assignee\", \"creator\"], message: \"Task due soon\"}\n  ]\n}\n</blueprint>",
  "validation": {\n    "has_thinking": true,\n    "has_blueprint": true,\n    "thinking_structure": [\"problem_understanding\", \"strategic_approach\", \"design_preparation\"],\n    "blueprint_syntax_valid": true,\n    "blueprint_completeness": [\"service_definition\", \"methods_with_contracts\", \"data_model\", \"behaviors\", \"integrations\"]\n  }\n}
```

---

## Training Data Structure Analysis

### JSONL vs JSON Format

**Recommendation: JSONL** for these reasons:

1. **Streaming-friendly**: Can process large datasets incrementally
2. **Memory efficient**: Load one example at a time during training
3. **Append-friendly**: Easy to add new examples without rewriting entire file
4. **Standard**: HuggingFace datasets work natively with JSONL

### Example JSONL Format

```jsonl
{"id": "blueprint_001", "category": "task_management", "difficulty": "intermediate", "user_query": "Design a task management system for a small team", "response": "<thinking>...</thinking>\n\n<blueprint>...</blueprint>"}
{"id": "blueprint_002", "category": "unit_conversion", "difficulty": "basic", "user_query": "Design a temperature conversion service", "response": "<thinking>...</thinking>\n\n<blueprint>...</blueprint>"}
{"id": "blueprint_003", "category": "salary_calculation", "difficulty": "basic", "user_query": "Design a service to calculate yearly income from hourly wages", "response": "<thinking>...</thinking>\n\n<blueprint>...</blueprint>"}
{"id": "blueprint_004", "category": "inventory_management", "difficulty": "advanced", "user_query": "Design an inventory management system for e-commerce", "response": "<thinking>...</thinking>\n\n<blueprint>...</blueprint>"}
```

### Simple Basic Example

```json
{
  "id": "blueprint_002",
  "category": "unit_conversion",
  "difficulty": "basic", 
  "user_query": "Design a temperature conversion service",
  "response": "<thinking>\nPROBLEM UNDERSTANDING:\n- Need to convert temperatures between different scales (Celsius, Fahrenheit, Kelvin)\n- Should handle input validation and error cases\n- Must provide accurate conversions with proper precision\n- Could be extended to other unit types in the future\n\nSTRATEGIC APPROACH:\n- Create a focused conversion service with clear operations\n- Support the most common temperature scales\n- Include input validation and range checking\n- Design for extensibility to other unit types\n- Ensure mathematical accuracy and proper rounding\n\nDESIGN PREPARATION:\n- Conversion methods for each scale combination\n- Input validation for reasonable temperature ranges\n- Error handling for invalid inputs\n- Structured result format with validation status\n</thinking>\n\n<blueprint>\nService TemperatureConverter {\n  description: \"Converts temperatures between Celsius, Fahrenheit, and Kelvin\",\n  \n  methods: {\n    celsiusToFahrenheit(celsius: float) -> ConversionResult {\n      preconditions: [celsius >= -273.15],\n      postconditions: [returns fahrenheit equivalent with 2 decimal precision],\n      errors: [InvalidTemperature]\n    },\n    \n    fahrenheitToCelsius(fahrenheit: float) -> ConversionResult {\n      preconditions: [fahrenheit >= -459.67],\n      postconditions: [returns celsius equivalent with 2 decimal precision],\n      errors: [InvalidTemperature]\n    },\n    \n    celsiusToKelvin(celsius: float) -> ConversionResult {\n      preconditions: [celsius >= -273.15],\n      postconditions: [returns kelvin equivalent with 2 decimal precision],\n      errors: [InvalidTemperature]\n    },\n    \n    kelvinToCelsius(kelvin: float) -> ConversionResult {\n      preconditions: [kelvin >= 0],\n      postconditions: [returns celsius equivalent with 2 decimal precision],\n      errors: [InvalidTemperature]\n    },\n    \n    convertTemperature(value: float, fromScale: TemperatureScale, toScale: TemperatureScale) -> ConversionResult {\n      preconditions: [valid temperature for source scale, different scales],\n      postconditions: [returns converted value with metadata]\n    }\n  }\n}\n\nType ConversionResult {\n  value: float,\n  originalValue: float,\n  fromScale: TemperatureScale,\n  toScale: TemperatureScale,\n  precision: int,\n  isValid: bool\n}\n\nType TemperatureScale {\n  enum: [\"celsius\", \"fahrenheit\", \"kelvin\"]\n}\n\nValidation TemperatureRanges {\n  celsius: {min: -273.15, max: 1000000},\n  fahrenheit: {min: -459.67, max: 1800032},\n  kelvin: {min: 0, max: 1000273.15}\n}\n</blueprint>",
  "validation": {\n    "has_thinking": true,\n    "has_blueprint": true,\n    "thinking_structure": ["problem_understanding", "strategic_approach", "design_preparation"],\n    "blueprint_syntax_valid": true,\n    "blueprint_completeness": ["service_definition", "methods_with_contracts", "types", "validation_rules"]\n  }\n}
```

### Another Basic Example - Salary Calculation

```json
{
  "id": "blueprint_003",
  "category": "salary_calculation",
  "difficulty": "basic",
  "user_query": "Design a service to calculate yearly income from hourly wages",
  "response": "<thinking>\nPROBLEM UNDERSTANDING:\n- Need to calculate annual salary from hourly wage information\n- Should account for different work schedules (part-time, full-time, overtime)\n- Must handle various pay periods and work patterns\n- Should include tax considerations and deductions\n\nSTRATEGIC APPROACH:\n- Create calculation service that handles different employment scenarios\n- Support flexible work schedule inputs (hours per week, weeks per year)\n- Include overtime calculations with premium rates\n- Provide both gross and estimated net income calculations\n- Design for different employment types and tax jurisdictions\n\nDESIGN PREPARATION:\n- Core calculation methods for different scenarios\n- Input validation for realistic wage and hour ranges\n- Overtime rate handling (1.5x, 2x multipliers)\n- Tax estimation capabilities\n- Result formatting with breakdown details\n</thinking>\n\n<blueprint>\nService SalaryCalculator {\n  description: \"Calculates yearly income from hourly wages with various work schedules\",\n  \n  methods: {\n    calculateYearlyIncome(hourlyWage: float, hoursPerWeek: float, weeksPerYear: int) -> IncomeResult {\n      preconditions: [hourlyWage > 0, hoursPerWeek > 0, weeksPerYear between 1 and 52],\n      postconditions: [returns gross annual income with calculation breakdown],\n      errors: [InvalidWage, InvalidHours, InvalidWeeks]\n    },\n    \n    calculateWithOvertime(hourlyWage: float, regularHours: float, overtimeHours: float, weeksPerYear: int, overtimeRate: float) -> IncomeResult {\n      preconditions: [hourlyWage > 0, regularHours > 0, overtimeHours >= 0, overtimeRate >= 1.0],\n      postconditions: [returns income with overtime premium calculations],\n      errors: [InvalidWage, InvalidHours, InvalidOvertimeRate]\n    },\n    \n    estimateNetIncome(grossIncome: float, taxRate: float, deductions: float) -> NetIncomeResult {\n      preconditions: [grossIncome > 0, taxRate between 0 and 1, deductions >= 0],\n      postconditions: [returns estimated net income after taxes and deductions],\n      errors: [InvalidGrossIncome, InvalidTaxRate]\n    },\n    \n    compareEmploymentOptions(scenarios: EmploymentScenario[]) -> ComparisonResult {\n      preconditions: [scenarios not empty, all scenarios valid],\n      postconditions: [returns ranked comparison with pros/cons analysis]\n    }\n  }\n}\n\nType IncomeResult {\n  grossAnnualIncome: float,\n  hourlyWage: float,\n  totalHoursPerYear: float,\n  regularIncome: float,\n  overtimeIncome: float,\n  calculationDate: timestamp,\n  breakdown: IncomeBreakdown\n}\n\nType NetIncomeResult {\n  grossIncome: float,\n  taxAmount: float,\n  deductions: float,\n  netIncome: float,\n  monthlyNetIncome: float,\n  biweeklyNetIncome: float\n}\n\nType EmploymentScenario {\n  name: string,\n  hourlyWage: float,\n  hoursPerWeek: float,\n  weeksPerYear: int,\n  overtimeHours: float,\n  benefits: BenefitsPackage\n}\n\nType IncomeBreakdown {\n  regularPay: {hours: float, rate: float, total: float},\n  overtimePay: {hours: float, rate: float, total: float},\n  weeklyAverage: float,\n  monthlyAverage: float,\n  biweeklyAverage: float\n}\n\nValidation SalaryRanges {\n  minimumWage: {value: 7.25, currency: \"USD\"},\n  maximumReasonableWage: {value: 500.00, currency: \"USD\"},\n  standardWorkWeek: {hours: 40},\n  standardWorkYear: {weeks: 52},\n  overtimeThreshold: {hours: 40}\n}\n</blueprint>",
  "validation": {\n    "has_thinking": true,\n    "has_blueprint": true,\n    "thinking_structure": ["problem_understanding", "strategic_approach", "design_preparation"],\n    "blueprint_syntax_valid": true,\n    "blueprint_completeness": ["service_definition", "methods_with_contracts", "types", "validation_rules"]\n  }\n}
```

---

## Thinking Token Structure

### Required Components

1. **PROBLEM UNDERSTANDING** - What is being asked, key requirements
2. **STRATEGIC APPROACH** - High-level solution strategy, not implementation
3. **DESIGN PREPARATION** - What components/patterns will be needed

### Quality Standards

- **Conceptual focus**: Stays at strategic level, avoids implementation details
- **Complete analysis**: Covers problem scope, constraints, and approach
- **BluePrint-oriented**: Maps problem to BluePrint design concepts
- **Structured format**: Consistent section headings for trainability

---

## BluePrint Token Structure  

### Required Components

1. **Service/Component definitions** with clear responsibilities
2. **Method signatures** with preconditions/postconditions
3. **Data models** with proper relationships and constraints
4. **Type definitions** where needed

### Quality Standards

- **Valid BluePrint syntax**: Follows docs/blueprint-prompt.md exactly
- **Complete specification**: Implementable without guesswork
- **Proper abstraction**: Right level of detail for system design
- **Best practices**: Security, error handling, validation addressed

---

## Training Curriculum Structure

### Dataset Organization

```
training/
├── blueprint_examples.jsonl           # Main training file
├── categories/
│   ├── basic_systems.jsonl           # 50 examples
│   ├── business_logic.jsonl          # 75 examples  
│   ├── technical_systems.jsonl       # 75 examples
│   └── complex_domains.jsonl         # 50 examples
└── validation/
    ├── test_cases.jsonl              # 50 held-out examples
    └── edge_cases.jsonl              # 25 challenging examples
```

### Progressive Complexity

**Stage 1: Basic Systems (50 examples)**
- Simple CRUD operations
- Basic entity relationships
- Single-service designs

**Stage 2: Business Logic (75 examples)**  
- Multi-service architectures
- Business rule enforcement
- State management patterns

**Stage 3: Technical Systems (75 examples)**
- Infrastructure components
- Security and authentication
- Performance and scaling concerns

**Stage 4: Complex Domains (50 examples)**
- Multi-domain integration
- Advanced patterns and relationships
- Real-world complexity

---

## Model Training Integration

### Token Processing

Given your reference to `history/train_bytelogic_worldmodel.py`, the model will need:

1. **Token recognition**: Identify `<thinking>` and `<blueprint>` boundaries
2. **Structured generation**: Generate both tokens in sequence
3. **Validation**: Real-time BluePrint syntax checking during training

### Loss Function Considerations

```python
def compute_blueprint_loss(outputs, labels):
    # Standard language modeling loss
    base_loss = standard_lm_loss(outputs, labels)
    
    # Extract blueprint regions for syntax validation
    blueprint_regions = extract_blueprint_tokens(outputs, labels)
    
    # Add penalty for invalid BluePrint syntax
    syntax_penalty = compute_syntax_penalty(blueprint_regions)
    
    return base_loss + 0.1 * syntax_penalty
```

---

## Implementation Recommendations

### Data Generation Pipeline

1. **Template-based generation**: Create templates for each category
2. **Human review**: Validate quality of generated examples
3. **Iterative refinement**: Start with 50 examples, test training, expand
4. **Automated validation**: Check syntax and completeness automatically

### Training Approach

1. **Start small**: 200 high-quality examples before scaling
2. **Progressive curriculum**: Train stages 1-4 sequentially
3. **Real-time validation**: Monitor BluePrint generation quality during training
4. **Early stopping**: Stop when validation syntax accuracy plateaus

### Success Metrics

- **Token presence**: >95% of outputs contain both tokens
- **BluePrint syntax**: >90% syntactically correct
- **Thinking quality**: Human evaluation of strategic reasoning
- **Generalization**: Performance on novel problem types

---

This approach prioritizes learning the `<thinking>` → `<blueprint>` pattern first, establishing a solid foundation before adding computational execution in Phase 2.

---

## Future Token Extensions

### Phase 2.5: Test Token (`<test>`)

**Concept**: Add a fourth token for generating comprehensive test specifications in Cucumber-style BDD format.

**Token Sequence Evolution**:
```
Phase 1 (Current): <thinking> → <blueprint>
Phase 2 (Planned): <thinking> → <blueprint> → <computation>
Phase 2.5 (Future): <thinking> → <blueprint> → <test> → <computation>
```

**Purpose**: Generate comprehensive test scenarios that validate the BluePrint specification before implementation.

**Format Example**:
```
<test>
Feature: Temperature Conversion Service
  As a user of the temperature conversion service
  I want to convert temperatures between different scales
  So that I can work with temperature data in my preferred format

Scenario: Convert Celsius to Fahrenheit
  Given I have a temperature of 25 degrees Celsius
  When I convert it to Fahrenheit
  Then I should get 77.0 degrees Fahrenheit
  And the conversion should be accurate to 2 decimal places

Scenario: Handle invalid temperature input
  Given I have a temperature below absolute zero (-300 Celsius)
  When I attempt to convert it to Fahrenheit
  Then I should receive an InvalidTemperature error
  And no conversion result should be returned

Scenario: Convert with precision requirements
  Given I have a temperature of 0 degrees Celsius
  When I convert it to Fahrenheit
  Then I should get exactly 32.0 degrees Fahrenheit
  And the result should include conversion metadata
  And the original value should be preserved in the result

Scenario Outline: Multiple scale conversions
  Given I have a temperature of <input> degrees <from_scale>
  When I convert it to <to_scale>
  Then I should get <expected> degrees <to_scale>
  
  Examples:
    | input | from_scale | to_scale   | expected |
    | 0     | celsius    | fahrenheit | 32.0     |
    | 100   | celsius    | fahrenheit | 212.0    |
    | 273.15| celsius    | kelvin     | 546.3    |
    | 32    | fahrenheit | celsius    | 0.0      |
</test>
```

**Training Benefits**:
- **Specification validation**: Tests verify BluePrint completeness
- **Edge case coverage**: Forces consideration of error scenarios
- **Acceptance criteria**: Clear success/failure conditions
- **Implementation guidance**: Tests become development roadmap

**Integration with Phases**:
- **Phase 1**: Focus on `<thinking>` → `<blueprint>` reliability
- **Phase 2**: Add `<computation>` for executable implementation  
- **Phase 2.5**: Insert `<test>` between blueprint and computation
- **Phase 3**: Full TDD cycle with test-driven implementation

**Success Metrics for Test Token**:
- Test coverage of all BluePrint methods and error conditions
- Valid Cucumber/Gherkin syntax (>90% accuracy)
- Comprehensive scenario coverage (happy path + edge cases)
- Alignment between BluePrint specification and test scenarios

This extension would create a complete specification → test → implementation workflow, ensuring robust software design through the full development lifecycle.