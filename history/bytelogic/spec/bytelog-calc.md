# ByteLog 3.0 Update Specification

## Overview

Update ByteLog 2.0 (logic + calculation) to ByteLog 3.0 (logic + calculation + loops).
The goal is to add FOR/WHILE loop constructs and string processing capabilities
for general computation while preserving all existing logic and calculation features.

---

## New Tokens to Add

### Keywords
```
CALC      - begin calculation block
INPUT     - declare input parameters  
LET       - bind variable to expression
RESULT    - return value from calculation
IF        - conditional start
THEN      - conditional then-branch
ELSE      - conditional else-branch
END       - end CALC, IF, or FOR block
WHERE     - filter clause in rules
FOR       - begin loop construct
WHILE     - while loop condition
IN        - iterator keyword (FOR var IN ...)
RANGE     - range function for numeric iteration
LENGTH    - string length function
CHAR_AT   - character access function
BREAK     - exit loop early
CONTINUE  - continue to next iteration
MOD       - modulo operator
POW       - power/exponentiation function
ABS       - absolute value function
MIN       - minimum function  
MAX       - maximum function
SQRT      - square root function
SIN       - sine function (radians)
COS       - cosine function (radians)
TAN       - tangent function (radians)
ASIN      - arcsine function
ACOS      - arccosine function
ATAN      - arctangent function
LOG       - natural logarithm
LOG10     - base-10 logarithm
EXP       - exponential function (e^x)
CEIL      - ceiling function
FLOOR     - floor function
```

### Operators
```
PLUS      +
MINUS     -
STAR      *
SLASH     /
MOD       %     (modulo operator)
LPAREN    (
RPAREN    )
ASSIGN    =     (for LET assignments)
GT        >
LT        <
GE        >=
LE        <=
EQ        ==    (alternate equality)
NE        !=
NE        <>    (alternate not-equal)
```

### Literals
```
FLOAT     -?[0-9]+\.[0-9]+    (must match before INTEGER)
STRING    \"[^\"]*\"          (double-quoted string literals)
```

---

## New Grammar Productions

### Calculation Definition
```
calc_def    : CALC IDENT calc_body END

calc_body   : input_decl stmt_list result_or_if
            | stmt_list result_or_if

input_decl  : INPUT var_list

var_list    : VAR
            | var_list VAR

stmt_list   : /* empty */
            | stmt_list let_stmt
            | stmt_list for_stmt
            | stmt_list break_stmt
            | stmt_list continue_stmt

let_stmt    : LET VAR ASSIGN expr

result_or_if : result_stmt
             | if_result

result_stmt : RESULT expr

if_result   : IF condition THEN stmt_list result_stmt END
            | IF condition THEN stmt_list result_stmt ELSE stmt_list result_stmt END
```

### Loop Constructs
```
for_stmt    : for_each_stmt
            | for_range_stmt  
            | for_while_stmt

for_each_stmt : FOR VAR IN LPAREN query RPAREN stmt_list END

for_range_stmt : FOR VAR IN range_call stmt_list END

for_while_stmt : FOR WHILE condition stmt_list END

range_call  : RANGE LPAREN expr COMMA expr RPAREN

string_op   : LENGTH LPAREN expr RPAREN
            | CHAR_AT LPAREN expr COMMA expr RPAREN

break_stmt  : BREAK

continue_stmt : CONTINUE
```

### Calculation Call
```
calc_call   : CALC IDENT LPAREN arg_list RPAREN
            | CALC IDENT LPAREN RPAREN

arg_list    : expr
            | arg_list COMMA expr
```

### Expressions (with precedence)
```
expr        : term
            | expr PLUS term
            | expr MINUS term

term        : factor
            | term STAR factor
            | term SLASH factor  
            | term MOD factor

factor      : INTEGER
            | FLOAT
            | STRING
            | VAR
            | LPAREN expr RPAREN
            | MINUS factor          /* unary negation, highest precedence */
            | string_op             /* LENGTH(...), CHAR_AT(...) */
            | range_call            /* RANGE(...) */
            | calc_call             /* CALC function_name(args) */
            | math_func             /* POW, ABS, MIN, MAX, SQRT */

math_func   : POW LPAREN expr COMMA expr RPAREN
            | ABS LPAREN expr RPAREN
            | MIN LPAREN expr COMMA expr RPAREN  
            | MAX LPAREN expr COMMA expr RPAREN
            | SQRT LPAREN expr RPAREN
```

### Conditions
```
condition   : expr compare_op expr

compare_op  : GT | LT | GE | LE | ASSIGN | EQ | NE
```

### Extended Rule Clauses
Add to existing rule_clause alternatives:
```
rule_clause : scan
            | join
            | let_stmt              /* NEW: LET inside rules */
            | where_clause          /* NEW: WHERE filter */
            | for_stmt              /* NEW: FOR loops inside rules */

where_clause : WHERE condition
```

### Extended EMIT
EMIT arguments should accept simple atoms (to avoid ambiguity with operators):
```
emit        : EMIT IDENT emit_atom emit_atom

emit_atom   : VAR
            | INTEGER
            | FLOAT
            | STRING
            | LPAREN expr RPAREN
```

---

## New AST Node Types

```c
NODE_CALC_DEF       // CALC block definition
NODE_CALC_CALL      // CALC function call
NODE_INPUT          // INPUT declaration
NODE_LET            // LET variable = expr
NODE_IF             // IF/THEN/ELSE
NODE_RESULT         // RESULT expr
NODE_WHERE          // WHERE condition

NODE_FOR_EACH       // FOR var IN (QUERY ...)
NODE_FOR_RANGE      // FOR var IN RANGE(...)
NODE_FOR_WHILE      // FOR WHILE condition
NODE_RANGE          // RANGE function call
NODE_STRING_OP      // LENGTH, CHAR_AT operations
NODE_BREAK          // BREAK statement (future)
NODE_CONTINUE       // CONTINUE statement (future)

NODE_EXPR_VAR       // Variable reference $n
NODE_EXPR_INT       // Integer literal
NODE_EXPR_FLOAT     // Float literal
NODE_EXPR_STRING    // String literal
NODE_EXPR_BINOP     // Binary operation (+, -, *, /)
NODE_EXPR_UNARY     // Unary operation (negation)
NODE_CONDITION      // Comparison (>, <, =, etc.)
```

### Operator Enum
```c
typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG,
    OP_GT, OP_LT, OP_GE, OP_LE, OP_EQ, OP_NE,
    OP_LENGTH, OP_CHAR_AT, OP_RANGE,
    OP_POW, OP_ABS, OP_MIN, OP_MAX, OP_SQRT
} OpType;
```

---

## Precedence Declaration

```yacc
%left PLUS MINUS
%left STAR SLASH
%right UMINUS
```

Use `%prec UMINUS` on the unary minus rule:
```
factor : MINUS factor %prec UMINUS
```

---

## Example Syntax

### Calculation Block
```
CALC f_to_c
  INPUT $0
  LET $1 = ($0 - 32) * 5 / 9
  RESULT $1
END
```

### Conditional Calculation
```
CALC abs
  INPUT $0
  IF $0 < 0 THEN
    RESULT -$0
  ELSE
    RESULT $0
  END
END
```

### Calculation Call
```
CALC f_to_c(212)
CALC max(10, 25)
```

### Rule with LET and WHERE
```
RULE temp_c: SCAN temp_f,
             LET $2 = ($1 - 32) * 5 / 9,
             WHERE $2 > 100,
             EMIT temp_c $0 $2
```

### Loop Examples

#### FOR-EACH: Result Set Iteration
```
CALC process_edges
  FOR edge_result IN (QUERY edge ? ?)
    LET $weight = edge_result.a + edge_result.b
    WHERE $weight > 5
    EMIT weighted_edge edge_result.a edge_result.b $weight
  END
END
```

#### FOR-RANGE: String Character Counting
```
CALC count_chars
  INPUT $word_string
  LET $count = 0
  FOR $i IN RANGE(0, LENGTH($word_string))
    LET $char = CHAR_AT($word_string, $i)
    IF $char == "r" THEN
      LET $count = $count + 1
    END
  END
  RESULT $count
END
```

#### FOR-WHILE: Conditional Iteration
```
CALC fibonacci
  INPUT $n
  LET $a = 0
  LET $b = 1
  LET $i = 0
  FOR WHILE $i < $n
    LET $temp = $a + $b
    LET $a = $b
    LET $b = $temp
    LET $i = $i + 1
  END
  RESULT $a
END
```

#### Nested Loops: Matrix Processing
```
CALC matrix_sum
  INPUT $matrix_rows $matrix_cols
  LET $total = 0
  FOR $i IN RANGE(0, $matrix_rows)
    FOR $j IN RANGE(0, $matrix_cols)
      LET $value = MATRIX_GET($matrix_a, $i, $j)
      LET $total = $total + $value
    END
  END
  RESULT $total
END
```

#### CALC Function Calls in Expressions
```
CALC factorial
  INPUT $n
  IF $n <= 1 THEN
    RESULT 1
  ELSE
    LET $prev = CALC factorial($n - 1)
    RESULT $n * $prev
  END
END

CALC complex_calculation
  INPUT $x
  LET $fact = CALC factorial(5)
  LET $power = POW($x, 3)
  LET $result = $fact + $power
  RESULT $result
END
```

#### Enhanced Mathematical Operations
```
CALC math_examples
  INPUT $a $b
  LET $mod_result = $a MOD $b
  LET $power_result = POW($a, $b)
  LET $abs_result = ABS($a - $b)
  LET $min_result = MIN($a, $b)
  LET $max_result = MAX($a, $b)
  LET $sqrt_result = SQRT($a)
  RESULT $max_result
END
```

#### Loop Control with BREAK and CONTINUE
```
CALC find_first_prime_above
  INPUT $start
  FOR $i IN RANGE($start, 1000)
    IF $i < 2 THEN
      CONTINUE
    END
    LET $is_prime = CALC check_prime($i)
    IF $is_prime == 1 THEN
      RESULT $i
      BREAK
    END
  END
  RESULT -1  ; No prime found
END
```

---

## Statement Type Summary

| Context | Return Single Value | Filter Set |
|---------|---------------------|------------|
| CALC block | RESULT expr | n/a |
| RULE body | LET $n = expr | WHERE condition |

- `RESULT` ends a CALC, returns one value
- `QUERY` extracts from relations, returns set
- `WHERE` filters tuples in rules (declarative)
- `IF/THEN/ELSE` branches in CALC (procedural)

---

## Key Implementation Notes

1. FLOAT regex must appear before INTEGER in lexer
2. STRING regex must appear before IDENT to avoid conflicts
3. Multi-char operators (>=, <=, !=, ==, <>) must appear before single-char
4. EMIT uses emit_atom (not full expr) to avoid ambiguity with `EMIT rel $0 - $1`
5. Add calc_def, calc_call, and for_stmt to statement alternatives
6. Union needs `double fval` for FLOAT token and `char *sval` for STRING token
7. Expression AST nodes need `left`, `right`, `op` fields
8. FOR loops create new variable scopes - loop variables are scoped to loop body
9. String processing requires string interning system in WAT generation
10. RANGE produces iterator that must be managed in loop runtime
11. Query iteration requires temporary result storage in memory
