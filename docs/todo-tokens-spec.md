# Todo Token Specification

## Overview

The `<todo>` and `<todo_item>` token system provides structured task planning and progress tracking within the WorldModel system. Similar to `<think>` tokens for reasoning and `<wat_model>` tokens for computation, todo tokens enable transparent task management and execution planning.

## Token Syntax

### Basic Structure

```
<todo>
  <todo_item status="pending">Task description</todo_item>
  <todo_item status="in_progress">Another task currently being worked on</todo_item>
  <todo_item status="completed">A finished task</todo_item>
</todo>
```

### Status Values

- `pending` - Task not yet started (default)
- `in_progress` - Task currently being executed
- `completed` - Task finished successfully
- `blocked` - Task cannot proceed (requires dependency or input)
- `skipped` - Task intentionally skipped or deemed unnecessary

### Extended Attributes

```xml
<todo_item 
  status="pending" 
  priority="high|medium|low"
  depends_on="task_id"
  estimated_time="5min|30min|2hrs"
  category="planning|coding|testing|documentation">
  Task description
</todo_item>
```

## Integration with WorldModel

### Positioning in Response Flow

Todo tokens should appear early in model responses for complex multi-step tasks:

```
User: Implement a new feature for user authentication with email verification

Assistant: <todo>
  <todo_item status="pending" category="planning">Design authentication API endpoints</todo_item>
  <todo_item status="pending" category="coding">Implement user registration with email</todo_item>
  <todo_item status="pending" category="coding" depends_on="registration">Create email verification system</todo_item>
  <todo_item status="pending" category="coding">Add login/logout functionality</todo_item>
  <todo_item status="pending" category="testing">Write unit tests for auth system</todo_item>
  <todo_item status="pending" category="documentation">Update API documentation</todo_item>
</todo>

I'll implement a comprehensive user authentication system with email verification. Let me start by designing the API endpoints...

<todo>
  <todo_item status="completed" category="planning">Design authentication API endpoints</todo_item>
  <todo_item status="in_progress" category="coding">Implement user registration with email</todo_item>
  <todo_item status="pending" category="coding" depends_on="registration">Create email verification system</todo_item>
  <todo_item status="pending" category="coding">Add login/logout functionality</todo_item>
  <todo_item status="pending" category="testing">Write unit tests for auth system</todo_item>
  <todo_item status="pending" category="documentation">Update API documentation</todo_item>
</todo>

[Implementation details follow...]
```

## Usage Patterns

### 1. Initial Task Breakdown
When receiving a complex request, immediately create a todo block to outline all required steps:

```xml
<todo>
  <todo_item status="pending" priority="high">Analyze current codebase</todo_item>
  <todo_item status="pending" priority="high">Identify integration points</todo_item>
  <todo_item status="pending" priority="medium">Implement core functionality</todo_item>
  <todo_item status="pending" priority="low">Add error handling</todo_item>
  <todo_item status="pending" priority="low">Write documentation</todo_item>
</todo>
```

### 2. Progress Updates
Update the todo block as work progresses, keeping the same structure but changing statuses:

```xml
<todo>
  <todo_item status="completed" priority="high">Analyze current codebase</todo_item>
  <todo_item status="in_progress" priority="high">Identify integration points</todo_item>
  <todo_item status="pending" priority="medium">Implement core functionality</todo_item>
  <todo_item status="pending" priority="low">Add error handling</todo_item>
  <todo_item status="pending" priority="low">Write documentation</todo_item>
</todo>
```

### 3. Dynamic Task Addition
Tasks can be discovered during implementation and added to the todo block:

```xml
<todo>
  <todo_item status="completed">Set up database schema</todo_item>
  <todo_item status="in_progress">Implement user model</todo_item>
  <todo_item status="pending">Add password hashing</todo_item>
  <todo_item status="pending">Create migration scripts</todo_item> <!-- Newly discovered -->
  <todo_item status="pending">Add input validation</todo_item> <!-- Newly discovered -->
</todo>
```

## Implementation Guidelines

### Token Processing Rules

1. **Atomic Updates**: Todo blocks should be treated as complete units - always show the full current state
2. **Status Consistency**: Only one task should be `in_progress` at a time within a todo block
3. **Completion Tracking**: Completed tasks remain visible to show progress
4. **Dependency Management**: Tasks marked with `depends_on` should not start until dependencies are completed

### Training Data Integration

Todo tokens should be included in training examples to teach models proper task management:

```
User: Fix the authentication bug and add rate limiting
Assistant: <todo>
  <todo_item status="pending" category="debugging">Reproduce authentication bug</todo_item>
  <todo_item status="pending" category="debugging">Identify root cause</todo_item>
  <todo_item status="pending" category="coding">Implement bug fix</todo_item>
  <todo_item status="pending" category="coding">Add rate limiting middleware</todo_item>
  <todo_item status="pending" category="testing">Test both features</todo_item>
</todo>

I'll start by reproducing the authentication bug...
```

### WASM Integration

Todo tokens can coordinate with WASM execution for computational tasks:

```xml
<todo>
  <todo_item status="completed">Parse input data</todo_item>
  <todo_item status="in_progress">Calculate mathematical result</todo_item>
  <todo_item status="pending">Format output</todo_item>
</todo>

<wat_model>
(module
  (func $compute (param f64 f64) (result f64)
    local.get 0
    local.get 1
    f64.mul)
  (export "compute" (func $compute)))
</wat_model>

<computed>42.5</computed>

<todo>
  <todo_item status="completed">Parse input data</todo_item>
  <todo_item status="completed">Calculate mathematical result</todo_item>
  <todo_item status="in_progress">Format output</todo_item>
</todo>
```

## Benefits

1. **Transparency**: Users can see planned work and current progress
2. **Consistency**: Standardized task management across different types of requests
3. **Debugging**: Clear visibility into which step failed or needs attention
4. **Learning**: Models learn structured problem decomposition
5. **Coordination**: Tasks can be coordinated with other token systems (WASM, Think)
6. **Resumability**: Work can be resumed from any point by checking todo status

## Future Considerations

- **Task IDs**: Unique identifiers for complex dependency management
- **Subtasks**: Nested todo structures for hierarchical planning
- **Time Tracking**: Actual vs estimated completion times
- **Collaboration**: Multi-agent task distribution
- **Persistence**: Todo state preservation across sessions

## Example Training Pattern

```
User: Create a comprehensive data validation system

Assistant: <todo>
  <todo_item status="pending" category="planning" estimated_time="15min">Design validation schema structure</todo_item>
  <todo_item status="pending" category="coding" estimated_time="45min">Implement core validation engine</todo_item>
  <todo_item status="pending" category="coding" estimated_time="30min">Add custom validator support</todo_item>
  <todo_item status="pending" category="coding" estimated_time="20min">Create error handling system</todo_item>
  <todo_item status="pending" category="testing" estimated_time="40min">Write comprehensive tests</todo_item>
  <todo_item status="pending" category="documentation" estimated_time="25min">Document usage patterns</todo_item>
</todo>

I'll create a comprehensive data validation system with flexible schema support and robust error handling.

<todo>
  <todo_item status="in_progress" category="planning" estimated_time="15min">Design validation schema structure</todo_item>
  <todo_item status="pending" category="coding" estimated_time="45min">Implement core validation engine</todo_item>
  <todo_item status="pending" category="coding" estimated_time="30min">Add custom validator support</todo_item>
  <todo_item status="pending" category="coding" estimated_time="20min">Create error handling system</todo_item>
  <todo_item status="pending" category="testing" estimated_time="40min">Write comprehensive tests</todo_item>
  <todo_item status="pending" category="documentation" estimated_time="25min">Document usage patterns</todo_item>
</todo>

Let me start by designing the validation schema structure...
[Implementation continues with progressive todo updates]
```