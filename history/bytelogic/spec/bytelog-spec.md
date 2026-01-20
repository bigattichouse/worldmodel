# ByteLog Language Specification & Implementation Guide

**Version:** 1.0  
**Status:** Design Complete  
**Purpose:** Implementation reference for any target language

---

## 1. Overview

### 1.1 What is ByteLog?

ByteLog is a minimal logic programming notation designed for Large Language Models to generate deterministic logical inference. It compiles to WebAssembly Text format (WAT) for execution.

**Design Philosophy:**
- Simple enough for LLMs to generate reliably
- Expressive enough for practical logic problems (~80% coverage)
- Compiles trivially to efficient WAT
- Guaranteed termination (monotonic Datalog semantics)

### 1.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Relation** | Named set of binary tuples (pairs of integers) |
| **Fact** | Ground tuple explicitly added to a relation |
| **Rule** | Inference pattern that derives new tuples from existing ones |
| **Fixpoint** | Iteratively apply rules until no new facts derived |
| **Query** | Extract results from computed relations |

### 1.3 Semantic Model

ByteLog implements **monotonic Datalog** semantics:
- Facts can only be added, never removed
- Rules fire whenever their conditions match
- Execution terminates when no new facts can be derived (fixpoint)
- No negation, no aggregation, no recursion limits needed

**Termination Guarantee:** Since relations are finite (bounded by memory) and facts only accumulate, the fixpoint is always reached in finite time.

---

## 2. Lexical Specification

### 2.1 Character Set

- Source encoding: UTF-8 (ASCII subset used for tokens)
- Case insensitive keywords (REL, Rel, rel all valid)
- Case sensitive identifiers (parent ≠ Parent)

### 2.2 Token Types
```
KEYWORD     ::= 'REL' | 'FACT' | 'RULE' | 'SCAN' | 'JOIN' 
              | 'EMIT' | 'MATCH' | 'SOLVE' | 'QUERY'

SYMBOL      ::= ':' | ',' | '?'

VARIABLE    ::= '$' DIGIT+

INTEGER     ::= '-'? DIGIT+

IDENTIFIER  ::= ALPHA (ALPHA | DIGIT | '_')*

ALPHA       ::= [a-zA-Z_]
DIGIT       ::= [0-9]
```

### 2.3 Whitespace and Comments
```
WHITESPACE  ::= ' ' | '\t' | '\n' | '\r'

COMMENT     ::= ';' [^\n]* '\n'      // Semicolon to end of line
              | '//' [^\n]* '\n'      // Double slash to end of line
```

### 2.4 Lexer Behavior

1. Skip whitespace between tokens
2. Skip comments (treat as whitespace)
3. Keywords take precedence over identifiers
4. Longest match for integers and identifiers
5. Track line numbers for error reporting

---

## 3. Grammar Specification

### 3.1 EBNF Grammar
```ebnf
program         ::= statement*

statement       ::= rel_decl
                  | fact
                  | rule
                  | solve
                  | query

rel_decl        ::= 'REL' IDENTIFIER

fact            ::= 'FACT' IDENTIFIER INTEGER INTEGER

rule            ::= 'RULE' IDENTIFIER ':' body ',' emit

body            ::= operation (',' operation)*

operation       ::= scan
                  | join

scan            ::= 'SCAN' IDENTIFIER ('MATCH' VARIABLE)?

join            ::= 'JOIN' IDENTIFIER VARIABLE

emit            ::= 'EMIT' IDENTIFIER VARIABLE VARIABLE

solve           ::= 'SOLVE'

query           ::= 'QUERY' IDENTIFIER query_arg query_arg

query_arg       ::= INTEGER
                  | '?'
```

### 3.2 Grammar Notes

1. **Statement order:** Conventionally REL → FACT → RULE → SOLVE → QUERY, but parser accepts any order
2. **Multiple rules:** Same target relation can have multiple rules (union semantics)
3. **Multiple queries:** Only last QUERY is executed (others ignored)
4. **SOLVE placement:** Must appear before QUERY for correct semantics

---

## 4. Abstract Syntax Tree

### 4.1 Node Types
```
┌─────────────────────────────────────────────────────────────┐
│ NodeType        │ Fields                                    │
├─────────────────────────────────────────────────────────────┤
│ Program         │ statements: List<Statement>               │
├─────────────────────────────────────────────────────────────┤
│ RelDecl         │ name: String                              │
├─────────────────────────────────────────────────────────────┤
│ Fact            │ relation: String                          │
│                 │ a: Integer                                │
│                 │ b: Integer                                │
├─────────────────────────────────────────────────────────────┤
│ Rule            │ target: String                            │
│                 │ body: List<Operation>                     │
│                 │ emit: Emit                                │
├─────────────────────────────────────────────────────────────┤
│ Scan            │ relation: String                          │
│                 │ match_var: Optional<Integer>              │
├─────────────────────────────────────────────────────────────┤
│ Join            │ relation: String                          │
│                 │ match_var: Integer                        │
├─────────────────────────────────────────────────────────────┤
│ Emit            │ relation: String                          │
│                 │ var_a: Integer                            │
│                 │ var_b: Integer                            │
├─────────────────────────────────────────────────────────────┤
│ Solve           │ (no fields)                               │
├─────────────────────────────────────────────────────────────┤
│ Query           │ relation: String                          │
│                 │ a: Optional<Integer>  (None = wildcard)   │
│                 │ b: Optional<Integer>  (None = wildcard)   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 AST Construction Pseudocode
```
parse_program():
    statements = []
    while not end_of_input:
        statements.append(parse_statement())
    return Program(statements)

parse_statement():
    token = peek()
    switch token:
        case REL:   return parse_rel_decl()
        case FACT:  return parse_fact()
        case RULE:  return parse_rule()
        case SOLVE: return parse_solve()
        case QUERY: return parse_query()
        default:    error("unexpected token")

parse_rule():
    expect(RULE)
    target = expect(IDENTIFIER)
    expect(COLON)
    body = parse_body()
    expect(COMMA)
    emit = parse_emit()
    return Rule(target, body, emit)

parse_body():
    operations = [parse_operation()]
    while peek() == COMMA and peek(2) not in [EMIT]:
        expect(COMMA)
        operations.append(parse_operation())
    return operations
```

---

## 5. Semantic Analysis

### 5.1 Validation Rules

| Rule | Description | Error |
|------|-------------|-------|
| **REL-EXIST** | Relations must be declared before use | "undefined relation: X" |
| **VAR-BIND** | Variables must be bound before use in JOIN/EMIT | "unbound variable: $N" |
| **VAR-SCOPE** | Variable bindings are scoped to single rule | "variable $N not in scope" |
| **EMIT-TARGET** | EMIT relation should match RULE target | warning only |

### 5.2 Variable Binding Rules

Variables are bound by SCAN operations:
- `SCAN rel` binds `$0` (column A) and `$1` (column B)
- `SCAN rel MATCH $N` requires `$N` already bound, still binds `$0` and `$1`
- `JOIN rel $N` requires `$N` bound, binds next sequential variable to column B

**Binding Analysis Algorithm:**
```
analyze_rule(rule):
    bound = {}
    next_var = 0
    
    for op in rule.body:
        if op is Scan:
            if op.match_var is not None:
                require(op.match_var in bound, "unbound match variable")
            bound[0] = true  // $0 = column A
            bound[1] = true  // $1 = column B
            next_var = 2
            
        if op is Join:
            require(op.match_var in bound, "unbound join variable")
            bound[next_var] = true  // bind column B to next var
            next_var += 1
    
    require(rule.emit.var_a in bound, "unbound emit variable")
    require(rule.emit.var_b in bound, "unbound emit variable")
```

### 5.3 Symbol Table
```
SymbolTable:
    relations: Map<String, RelationInfo>
    
RelationInfo:
    name: String
    base_address: Integer      // assigned during codegen
    declared: Boolean
    referenced: Boolean
```

---

## 6. WAT Code Generation

### 6.1 Memory Layout
```
┌─────────────────────────────────────────────────────────────┐
│ RELATION MEMORY LAYOUT                                      │
├─────────────────────────────────────────────────────────────┤
│ Offset 0:     count (i32) - number of tuples                │
│ Offset 4:     tuple[0].a (i32)                              │
│ Offset 8:     tuple[0].b (i32)                              │
│ Offset 12:    tuple[1].a (i32)                              │
│ Offset 16:    tuple[1].b (i32)                              │
│ ...                                                         │
│ Offset 4+8*i: tuple[i].a (i32)                              │
│ Offset 8+8*i: tuple[i].b (i32)                              │
└─────────────────────────────────────────────────────────────┘

RELATION_SIZE = 4096 bytes (supports 511 tuples per relation)
BYTES_PER_TUPLE = 8 bytes (2 × i32)
```

**Address Calculation:**
```
relation_base(name) = relation_index(name) * RELATION_SIZE

tuple_address(base, index) = base + 4 + (index * 8)

column_a_address(base, index) = base + 4 + (index * 8)
column_b_address(base, index) = base + 8 + (index * 8)
```

### 6.2 Runtime Primitives

The following functions must be generated in every module:
```wat
;; Get count of tuples in relation
(func $rel_count (param $base i32) (result i32)
  (i32.load (local.get $base)))

;; Get column A of tuple at index
(func $rel_get_a (param $base i32) (param $idx i32) (result i32)
  (i32.load 
    (i32.add (local.get $base)
    (i32.add (i32.const 4)
    (i32.mul (local.get $idx) (i32.const 8))))))

;; Get column B of tuple at index
(func $rel_get_b (param $base i32) (param $idx i32) (result i32)
  (i32.load 
    (i32.add (local.get $base)
    (i32.add (i32.const 8)
    (i32.mul (local.get $idx) (i32.const 8))))))

;; Add tuple if not duplicate, return 1 if added, 0 if exists
(func $rel_add (param $base i32) (param $a i32) (param $b i32) (result i32)
  ;; See full implementation in Section 6.5
)

;; Check if tuple exists
(func $rel_has (param $base i32) (param $a i32) (param $b i32) (result i32)
  ;; See full implementation in Section 6.5
)
```

### 6.3 Code Generation Strategy
```
generate_wat(program):
    // Phase 1: Collect metadata
    relations = collect_relations(program)
    assign_base_addresses(relations)
    
    // Phase 2: Generate sections
    output = []
    output.append(generate_header(relations))
    output.append(generate_globals(relations))
    output.append(generate_runtime())
    output.append(generate_init_facts(program.facts))
    output.append(generate_rules(program.rules))
    output.append(generate_solve(program.rules))
    output.append(generate_query(program.query))
    output.append(generate_footer())
    
    return join(output)
```

### 6.4 Rule Compilation

Each rule compiles to a function returning `i32` (1 if any facts added, 0 otherwise).

**Compilation Pattern:**
```
RULE target: SCAN rel1, JOIN rel2 $1, EMIT target $0 $2
```

Becomes:
```wat
(func $rule_N (result i32)
  (local $changed i32)
  (local $i0 i32)         ;; loop index for SCAN
  (local $i1 i32)         ;; loop index for JOIN
  (local $v0 i32)         ;; $0
  (local $v1 i32)         ;; $1
  (local $v2 i32)         ;; $2
  
  ;; SCAN rel1
  (local.set $i0 (i32.const 0))
  (block $done0
    (loop $loop0
      (br_if $done0 (i32.ge_u (local.get $i0) (call $rel_count (global.get $rel1_base))))
      (local.set $v0 (call $rel_get_a (global.get $rel1_base) (local.get $i0)))
      (local.set $v1 (call $rel_get_b (global.get $rel1_base) (local.get $i0)))
      
      ;; JOIN rel2 $1 (where rel2.col_a = $1)
      (local.set $i1 (i32.const 0))
      (block $done1
        (loop $loop1
          (br_if $done1 (i32.ge_u (local.get $i1) (call $rel_count (global.get $rel2_base))))
          (if (i32.eq (call $rel_get_a (global.get $rel2_base) (local.get $i1)) (local.get $v1))
            (then
              (local.set $v2 (call $rel_get_b (global.get $rel2_base) (local.get $i1)))
              
              ;; EMIT target $0 $2
              (local.set $changed (i32.or (local.get $changed)
                (call $rel_add (global.get $target_base) (local.get $v0) (local.get $v2))))
            ))
          (local.set $i1 (i32.add (local.get $i1) (i32.const 1)))
          (br $loop1)))
      
      (local.set $i0 (i32.add (local.get $i0) (i32.const 1)))
      (br $loop0)))
  
  (local.get $changed))
```

### 6.5 Complete Runtime Implementation
```wat
;; ═══════════════════════════════════════════════════════════════════════════
;; RUNTIME PRIMITIVES
;; ═══════════════════════════════════════════════════════════════════════════

(func $rel_count (param $base i32) (result i32)
  (i32.load (local.get $base)))

(func $rel_get_a (param $base i32) (param $idx i32) (result i32)
  (i32.load 
    (i32.add (local.get $base)
    (i32.add (i32.const 4)
    (i32.mul (local.get $idx) (i32.const 8))))))

(func $rel_get_b (param $base i32) (param $idx i32) (result i32)
  (i32.load 
    (i32.add (local.get $base)
    (i32.add (i32.const 8)
    (i32.mul (local.get $idx) (i32.const 8))))))

(func $rel_add (param $base i32) (param $a i32) (param $b i32) (result i32)
  (local $count i32)
  (local $i i32)
  (local $addr i32)
  
  ;; Load current count
  (local.set $count (i32.load (local.get $base)))
  
  ;; Duplicate check loop
  (local.set $i (i32.const 0))
  (block $not_dup
    (loop $check
      ;; Exit if checked all tuples
      (br_if $not_dup (i32.ge_u (local.get $i) (local.get $count)))
      
      ;; Calculate tuple address
      (local.set $addr 
        (i32.add (local.get $base)
        (i32.add (i32.const 4)
        (i32.mul (local.get $i) (i32.const 8)))))
      
      ;; Check if tuple matches
      (if (i32.and
            (i32.eq (i32.load (local.get $addr)) (local.get $a))
            (i32.eq (i32.load (i32.add (local.get $addr) (i32.const 4))) (local.get $b)))
        (then (return (i32.const 0))))  ;; Duplicate found
      
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $check)))
  
  ;; Add new tuple
  (local.set $addr 
    (i32.add (local.get $base)
    (i32.add (i32.const 4)
    (i32.mul (local.get $count) (i32.const 8)))))
  
  (i32.store (local.get $addr) (local.get $a))
  (i32.store (i32.add (local.get $addr) (i32.const 4)) (local.get $b))
  
  ;; Increment count
  (i32.store (local.get $base) (i32.add (local.get $count) (i32.const 1)))
  
  (i32.const 1))  ;; Return 1 = changed

(func $rel_has (param $base i32) (param $a i32) (param $b i32) (result i32)
  (local $count i32)
  (local $i i32)
  (local $addr i32)
  
  (local.set $count (i32.load (local.get $base)))
  (local.set $i (i32.const 0))
  
  (block $not_found
    (loop $check
      (br_if $not_found (i32.ge_u (local.get $i) (local.get $count)))
      
      (local.set $addr 
        (i32.add (local.get $base)
        (i32.add (i32.const 4)
        (i32.mul (local.get $i) (i32.const 8)))))
      
      (if (i32.and
            (i32.eq (i32.load (local.get $addr)) (local.get $a))
            (i32.eq (i32.load (i32.add (local.get $addr) (i32.const 4))) (local.get $b)))
        (then (return (i32.const 1))))  ;; Found
      
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $check)))
  
  (i32.const 0))  ;; Not found
```

### 6.6 Solve Function
```wat
(func $solve (export "solve")
  (local $changed i32)
  
  ;; Initialize facts
  (call $init_facts)
  
  ;; Fixpoint loop
  (local.set $changed (i32.const 1))
  (block $done
    (loop $fixpoint
      (br_if $done (i32.eqz (local.get $changed)))
      (local.set $changed (i32.const 0))
      
      ;; Apply all rules
      (local.set $changed (i32.or (local.get $changed) (call $rule_0)))
      (local.set $changed (i32.or (local.get $changed) (call $rule_1)))
      ;; ... for each rule
      
      (br $fixpoint))))
```

### 6.7 Query Function Variants

**Query with both values specified (membership test):**
```wat
;; QUERY rel A B → returns 1 if exists, 0 otherwise
(func $query (export "query") (result i32)
  (call $solve)
  (call $rel_has (global.get $rel_base) (i32.const A) (i32.const B)))
```

**Query with first value, wildcard second:**
```wat
;; QUERY rel A ? → returns count of matches
(func $query (export "query") (result i32)
  (local $count i32)
  (local $i i32)
  (call $solve)
  (local.set $i (i32.const 0))
  (block $done
    (loop $scan
      (br_if $done (i32.ge_u (local.get $i) (call $rel_count (global.get $rel_base))))
      (if (i32.eq (call $rel_get_a (global.get $rel_base) (local.get $i)) (i32.const A))
        (then (local.set $count (i32.add (local.get $count) (i32.const 1)))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $scan)))
  (local.get $count))
```

**Query with both wildcards:**
```wat
;; QUERY rel ? ? → returns total count
(func $query (export "query") (result i32)
  (call $solve)
  (call $rel_count (global.get $rel_base)))
```

---

## 7. Error Handling

### 7.1 Error Categories

| Category | Phase | Examples |
|----------|-------|----------|
| **Lexical** | Tokenization | Invalid character, malformed number |
| **Syntactic** | Parsing | Missing colon, unexpected token |
| **Semantic** | Analysis | Undefined relation, unbound variable |
| **Runtime** | Execution | Memory overflow (too many tuples) |

### 7.2 Error Message Format
```
[ERROR] <filename>:<line>:<column>: <message>
        <source line>
        <caret indicator>
```

**Example:**
```
[ERROR] test.bl:7:15: undefined relation 'ancestor'
        RULE ancestor: SCAN parent, EMIT ancestor $0 $1
                       ^~~~
```

### 7.3 Recovery Strategy

- **Lexical errors:** Skip to next whitespace, continue tokenizing
- **Syntactic errors:** Skip to next statement keyword (REL, FACT, RULE, etc.)
- **Semantic errors:** Continue analysis to report multiple errors
- **Runtime errors:** Return error code from WASM execution

---

## 8. Implementation Checklist

### 8.1 Core Components
```
[ ] Lexer
    [ ] Token definitions
    [ ] Whitespace/comment handling
    [ ] Line number tracking
    [ ] Error reporting

[ ] Parser  
    [ ] Recursive descent or table-driven
    [ ] AST construction
    [ ] Syntax error recovery

[ ] Semantic Analyzer
    [ ] Symbol table construction
    [ ] Variable binding analysis
    [ ] Type checking (all i32)
    [ ] Undefined reference detection

[ ] Code Generator
    [ ] Memory layout calculation
    [ ] Runtime function emission
    [ ] Fact initialization
    [ ] Rule compilation
    [ ] Solve loop generation
    [ ] Query compilation

[ ] Integration
    [ ] wat2wasm compilation
    [ ] WASM runtime execution
    [ ] Result extraction
```

### 8.2 Testing Requirements

**Lexer Tests:**
```
"REL parent"           → [REL, IDENT("parent")]
"FACT parent 0 1"      → [FACT, IDENT("parent"), INT(0), INT(1)]
"$0 $1 $2"             → [VAR(0), VAR(1), VAR(2)]
"; comment\nREL x"     → [REL, IDENT("x")]
```

**Parser Tests:**
```
"REL parent"                              → RelDecl("parent")
"FACT parent 0 1"                         → Fact("parent", 0, 1)
"RULE a: SCAN b, EMIT a $0 $1"            → Rule("a", [Scan("b")], Emit("a", 0, 1))
"QUERY rel 1 ?"                           → Query("rel", Some(1), None)
```

**Integration Tests:**
```
Test: Transitive Closure
Input:
    REL edge
    REL path
    FACT edge 0 1
    FACT edge 1 2
    FACT edge 2 3
    RULE path: SCAN edge, EMIT path $0 $1
    RULE path: SCAN edge, JOIN path $1, EMIT path $0 $2
    SOLVE
    QUERY path 0 3
Expected: 1 (true)

Test: Symmetric Closure
Input:
    REL friend
    REL knows
    FACT friend 0 1
    FACT friend 2 3
    RULE knows: SCAN friend, EMIT knows $0 $1
    RULE knows: SCAN friend, EMIT knows $1 $0
    SOLVE
    QUERY knows ? ?
Expected: 4 (0↔1, 2↔3)
```

---

## 9. Language Extensions (Future)

### 9.1 Potential Additions

| Extension | Syntax | Complexity | Value |
|-----------|--------|------------|-------|
| Negation | `NOT rel $0 $1` | High | Set difference |
| Arithmetic | `ADD $0 $1 $2` | Medium | Computed values |
| Aggregation | `COUNT`, `MIN`, `MAX` | Medium | Aggregate queries |
| Strings | String interning | High | Named entities |
| N-ary | `REL name 3` | Medium | More expressiveness |

### 9.2 Extension Guidelines

1. **Keep termination guarantees** — no unbounded recursion
2. **Map cleanly to WAT** — no complex runtime needed
3. **Stay LLM-friendly** — simple, regular syntax
4. **Preserve composability** — extensions should combine well

---

## 10. Reference Examples

### 10.1 Ancestor (Canonical)
```bytelog
REL parent
REL ancestor

FACT parent 0 1       ; alice → bob
FACT parent 1 2       ; bob → carol
FACT parent 2 3       ; carol → dave

; Base: direct parents are ancestors
RULE ancestor: SCAN parent, EMIT ancestor $0 $1

; Recursive: ancestor of ancestor
RULE ancestor: SCAN parent, JOIN ancestor $1, EMIT ancestor $0 $2

SOLVE
QUERY ancestor 0 ?    ; who is alice ancestor of? → 3 (bob, carol, dave)
```

### 10.2 Graph Reachability
```bytelog
REL edge
REL reachable

FACT edge 0 1
FACT edge 0 2
FACT edge 1 3
FACT edge 2 3
FACT edge 3 4

RULE reachable: SCAN edge, EMIT reachable $0 $1
RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2

SOLVE
QUERY reachable 0 4   ; can reach 4 from 0? → 1 (yes)
```

### 10.3 Symmetric Relation
```bytelog
REL friend
REL knows

FACT friend 0 1
FACT friend 1 2

; Symmetric: if A friends B, then B friends A
RULE knows: SCAN friend, EMIT knows $0 $1
RULE knows: SCAN friend, EMIT knows $1 $0

; Transitive: friends of friends
RULE knows: SCAN knows, JOIN knows $1, EMIT knows $0 $2

SOLVE
QUERY knows 0 2       ; does 0 know 2? → 1 (yes, through 1)
```

### 10.4 Classification
```bytelog
REL isa
REL has_property

; Taxonomy
FACT isa 0 1          ; tweety isa bird
FACT isa 1 2          ; bird isa animal

; Properties
FACT has_property 1 10   ; bird has_property flies
FACT has_property 2 11   ; animal has_property breathes

; Inheritance
RULE isa: SCAN isa, JOIN isa $1, EMIT isa $0 $2
RULE has_property: SCAN isa, JOIN has_property $1, EMIT has_property $0 $2

SOLVE
QUERY has_property 0 ?   ; what properties does tweety have? → 2 (flies, breathes)
```

---

## 11. Implementation Notes by Language

### 11.1 Python
```python
# Recommended: dataclasses for AST, simple recursive descent parser
# Use: wasmtime package for execution

from dataclasses import dataclass
from typing import List, Optional
import wasmtime

@dataclass
class Fact:
    relation: str
    a: int
    b: int
```

### 11.2 Rust
```rust
// Recommended: nom for parsing, wasmtime crate for execution
// Use: enum for AST nodes

enum Statement {
    RelDecl { name: String },
    Fact { relation: String, a: i32, b: i32 },
    Rule { target: String, body: Vec<Operation>, emit: Emit },
    Solve,
    Query { relation: String, a: Option<i32>, b: Option<i32> },
}
```

### 11.3 TypeScript
```typescript
// Recommended: hand-written parser or pegjs
// Use: @aspect-build/aspect-wasm for execution

interface Fact {
    type: 'fact';
    relation: string;
    a: number;
    b: number;
}
```

### 11.4 C
```c
// Recommended: flex/bison (provided in spec)
// Use: wasmtime-c-api or direct wat2wasm + runtime

typedef struct ASTNode {
    NodeType type;
    char *name;
    int values[4];
    struct ASTNode *child;
    struct ASTNode *next;
} ASTNode;
```

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial specification |

---

## 13. License

This specification is released for implementation in the WASM WorldModel project. Implementations may use any license compatible with the project's goals.

