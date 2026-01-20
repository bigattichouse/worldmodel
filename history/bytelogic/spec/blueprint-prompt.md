# ByteLog Blueprint Prompt

**Version 1.0 - LLM Language Understanding Guide**

This document serves as a comprehensive reference for Large Language Models to understand and generate ByteLog code effectively.

---

## What is ByteLog?

ByteLog is a minimal logic programming language designed specifically for LLMs to perform deterministic logical inference. It compiles to WebAssembly and implements Datalog semantics with guaranteed termination.

**Core Philosophy:**
- Simple enough for LLMs to generate reliably
- Expressive enough for practical logic problems (~80% coverage)
- Always terminates (monotonic Datalog semantics)
- Compiles to efficient WebAssembly

---

## Language Overview

### Basic Structure
Every ByteLog program follows this pattern:
```
REL <relation_names>          # Declare relations
FACT <relation> <value> <value>    # Add ground facts
RULE <name>: <operations>, EMIT <relation> <vars>  # Inference rules
SOLVE                         # Execute inference
QUERY <relation> <arg> <arg>  # Extract results
```

### Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Relation** | Set of binary tuples (pairs) | `parent(alice, bob)` |
| **Fact** | Ground truth tuple | `FACT parent alice bob` |
| **Rule** | Inference pattern | `RULE ancestor: SCAN parent, EMIT ancestor $0 $1` |
| **Variable** | Bound values in rules | `$0`, `$1`, `$2` |
| **Fixpoint** | Iterative rule application until no new facts | Automatic |

---

## Complete Syntax Reference

### 1. Relation Declarations
```
REL parent
REL grandparent  
REL ancestor
```
*Declare all relations before use*

### 2. Facts
```
FACT parent alice bob      # String atoms become integers
FACT parent bob charlie
FACT age alice 25          # Direct integer values
```

### 3. Rules
```
RULE target_name: <body>, EMIT <relation> <var> <var>
```

**Rule Operations:**
- `SCAN relation` - Iterate over all tuples, bind `$0` (first) and `$1` (second)
- `SCAN relation MATCH $N` - Only tuples where first column equals variable `$N`
- `JOIN relation $N` - Find tuples where first column equals `$N`, bind second column to next variable

### 4. Variables
- Always start with `$` followed by number: `$0`, `$1`, `$2`...
- `$0` and `$1` are bound by SCAN operations
- `$2`, `$3`... are bound by JOIN operations (sequential)
- Variables must be bound before use

### 5. Queries
```
QUERY relation value ?     # Find all matches for first position
QUERY relation ? value     # Find all matches for second position  
QUERY relation ? ?         # Count all tuples
QUERY relation val1 val2   # Test if tuple exists
```

---

## Common Patterns

### Transitive Closure (Ancestor)
```
REL parent
REL ancestor

FACT parent alice bob
FACT parent bob charlie
FACT parent charlie dave

; Base case: parents are ancestors
RULE ancestor: SCAN parent, EMIT ancestor $0 $1

; Recursive: ancestor of ancestor
RULE ancestor: SCAN parent, JOIN ancestor $1, EMIT ancestor $0 $2

SOLVE
QUERY ancestor alice ?
```

### Symmetric Relations
```
REL friend
REL knows

FACT friend alice bob

; Symmetric: if A friends B, then B friends A
RULE knows: SCAN friend, EMIT knows $0 $1
RULE knows: SCAN friend, EMIT knows $1 $0

SOLVE
QUERY knows bob alice  ; Should return 1 (true)
```

### Graph Reachability
```
REL edge
REL reachable

FACT edge a b
FACT edge b c
FACT edge c d

RULE reachable: SCAN edge, EMIT reachable $0 $1
RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2

SOLVE
QUERY reachable a d  ; Can reach d from a?
```

### Family Relationships
```
REL parent
REL grandparent
REL sibling

FACT parent alice bob
FACT parent alice charlie
FACT parent bob david

; Grandparent: parent of parent
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2

; Sibling: same parent
RULE sibling: SCAN parent MATCH $0, JOIN parent $0, EMIT sibling $1 $2

SOLVE
QUERY grandparent alice ?
QUERY sibling bob ?
```

---

## LLM Generation Guidelines

### DO
✅ Always declare relations with `REL` first
✅ Use simple, meaningful relation names (`parent`, `edge`, `knows`)
✅ Add facts before rules
✅ Include `SOLVE` before `QUERY`
✅ Use sequential variable numbering (`$0`, `$1`, `$2`)
✅ Start with base cases, then add recursive rules
✅ Use comments with `;` to explain logic

### DON'T  
❌ Use undefined relations
❌ Skip variable numbers (`$0`, `$2` without `$1`)
❌ Forget `SOLVE` statement
❌ Use variables before binding them
❌ Create infinite loops (ByteLog prevents this automatically)

### Template for New Problems

```
; Problem: [describe the logical relationship]
; Input: [describe given facts]
; Goal: [describe what to compute]

REL [base_relation]
REL [derived_relation]

; Facts
FACT [base_relation] [value1] [value2]
[... more facts ...]

; Rules
RULE [rule_name]: SCAN [base_relation], EMIT [derived_relation] $0 $1
RULE [rule_name]: SCAN [relation1], JOIN [relation2] $1, EMIT [target] $0 $2

SOLVE
QUERY [target_relation] [query_args]
```

---

## Debugging Tips

### Common Errors
1. **Unbound variables**: Make sure variables are bound by SCAN/JOIN before use
2. **Undefined relations**: All relations must be declared with `REL`
3. **Missing SOLVE**: Rules won't execute without `SOLVE`
4. **Variable ordering**: Use `$0`, `$1`, `$2`... in sequence

### Verification Steps
1. Check all relations are declared
2. Verify variable binding flow in rules  
3. Confirm `SOLVE` appears before `QUERY`
4. Test with simple facts first

---

## Advanced Patterns

### Multiple Rules for Same Relation
```
REL knows
FACT friend a b
FACT colleague a c

; Union: knows includes both friends and colleagues
RULE knows: SCAN friend, EMIT knows $0 $1
RULE knows: SCAN colleague, EMIT knows $0 $1
```

### Filtering with MATCH
```
; Only process tuples where first element matches specific variable
RULE filtered: SCAN relation MATCH $1, EMIT result $0 $1
```

### Chain Joins
```
; Connect through multiple relations
RULE path: SCAN edge1, JOIN edge2 $1, JOIN edge3 $2, EMIT path $0 $3
```

---

## Example Library

### 1. Social Network
```
REL follows
REL friend_of_friend

FACT follows alice bob
FACT follows bob charlie
FACT follows alice david

RULE friend_of_friend: SCAN follows, JOIN follows $1, EMIT friend_of_friend $0 $2

SOLVE
QUERY friend_of_friend alice ?
```

### 2. Hierarchy
```
REL reports_to
REL manages

FACT reports_to emp1 mgr1
FACT reports_to emp2 mgr1
FACT reports_to mgr1 director

RULE manages: SCAN reports_to, EMIT manages $1 $0
RULE manages: SCAN reports_to, JOIN manages $0, EMIT manages $2 $1

SOLVE
QUERY manages director ?
```

### 3. Course Prerequisites
```
REL prerequisite  
REL can_take

FACT prerequisite math101 math201
FACT prerequisite math201 math301
FACT completed alice math101

RULE can_take: SCAN completed MATCH $0, JOIN prerequisite $1, EMIT can_take $0 $2

SOLVE
QUERY can_take alice ?
```

---

## Tool Integration

### Compilation
```bash
# Interpret and run
./build/bytelogic program.bl

# Compile to WebAssembly Text
./build/bytelogic --compile=wat program.bl

# Compile to WASM binary
./build/bytelogic --compile=wasm program.bl
```

### Output Modes
```bash
# Minimal output (results only)
./build/bytelogic program.bl

# Verbose output (detailed execution)
./build/bytelogic --verbose program.bl
```

---

## Semantic Model

ByteLog implements **monotonic Datalog**:
- Facts accumulate (never deleted)
- Rules fire when conditions match
- Execution reaches fixpoint automatically
- Termination guaranteed
- No negation, no aggregation
- Deterministic results

This makes it ideal for LLM generation because:
- Simple semantics → fewer errors
- Guaranteed termination → safe execution  
- Deterministic → predictable results
- Expressive → handles most logic problems

---

## When to Use ByteLog

**Good fit:**
- Family relationships, social networks
- Graph reachability, transitive closure
- Classification hierarchies
- Rule-based systems
- Logical puzzles and constraints

**Not ideal for:**
- Numerical computation (use regular programming)
- String processing (integers only)
- Complex aggregations (count/sum/average)
- Probabilistic reasoning
- Time-based logic

---

*This blueprint enables LLMs to generate correct, efficient ByteLog programs for a wide range of logical inference tasks.*