# ByteLogic Training Curriculum

**Version:** 1.0  
**Date:** 2026-01-19  
**Purpose:** Comprehensive training curriculum for ByteLogic language features in worldmodel LLM

---

## Overview

This curriculum teaches the LLM to generate ByteLogic code for structured reasoning tasks. The curriculum progresses from basic facts and queries to complex multi-domain reasoning, following pedagogical principles for effective learning.

### Learning Objectives

By the end of this curriculum, the model should be able to:
1. Generate syntactically correct ByteLogic programs
2. Model real-world problems using relations, facts, and rules
3. Express complex reasoning patterns (transitive closure, classification, etc.)
4. Combine logical reasoning with mathematical computation
5. Debug and optimize ByteLogic programs

---

## Stage 1: Foundations (Weeks 1-2)

### 1.1 Basic Syntax and Vocabulary

**Core Concepts:**
- Relations as sets of binary tuples
- Facts as ground truth statements
- Variables for binding values
- Simple queries

**Example Training Pairs:**

#### Facts and Queries
```
Input: "Is Alice the parent of Bob?"
Expected Output: "Let me check: <computation>
REL parent
FACT parent alice bob
SOLVE
QUERY parent alice bob
</computation> → Yes, Alice is Bob's parent."
```

#### Simple Relationships
```
Input: "Who are Alice's children?"
Expected Output: "I'll find Alice's children: <computation>
REL parent
FACT parent alice bob
FACT parent alice charlie
SOLVE
QUERY parent alice ?
</computation> → Bob and Charlie are Alice's children."
```

#### Multiple Relations
```
Input: "List all the friendships and who likes ice cream"
Expected Output: "Here are the relationships: <computation>
REL friend
REL likes
FACT friend alice bob
FACT friend bob charlie
FACT likes alice ice_cream
FACT likes charlie ice_cream
SOLVE
QUERY friend ? ?
QUERY likes ? ice_cream
</computation> → Friendships: alice↔bob, bob↔charlie. Ice cream lovers: alice, charlie."
```

### 1.2 Variable Binding Patterns

**Teaching Variables ($0, $1, $2):**

```
Input: "Who are all the people in parent relationships?"
Expected Output: "Let me find all parent-child pairs: <computation>
REL parent
FACT parent alice bob
FACT parent bob charlie
FACT parent charlie david
SOLVE
QUERY parent ? ?
</computation> → Parent relationships: alice→bob, bob→charlie, charlie→david"
```

**SCAN Operation:**
```
Input: "Find everyone who is a parent"
Expected Output: "I'll scan for all parents: <computation>
REL parent
FACT parent alice bob
FACT parent alice charlie
FACT parent bob david
RULE is_parent: SCAN parent, EMIT is_parent $0 1
SOLVE
QUERY is_parent ? 1
</computation> → Parents are: alice, bob"
```

### 1.3 Error Patterns and Debugging

**Common Mistakes to Avoid:**
```
# WRONG - undefined relation
REL friend
FACT likes alice bob  # Error: 'likes' not declared

# WRONG - unbound variable  
RULE test: SCAN parent, EMIT result $2 $0  # Error: $2 not bound

# WRONG - missing SOLVE
REL parent
FACT parent alice bob
QUERY parent alice ?  # Error: no SOLVE before QUERY
```

---

## Stage 2: Inference Rules (Weeks 3-4)

### 2.1 Single-Step Inference

**Symmetric Relations:**
```
Input: "If Alice is friends with Bob, then Bob is friends with Alice. Who are Bob's friends?"
Expected Output: "I'll create symmetric friendship rules: <computation>
REL friend_directed
REL friend
FACT friend_directed alice bob
RULE friend: SCAN friend_directed, EMIT friend $0 $1
RULE friend: SCAN friend_directed, EMIT friend $1 $0
SOLVE
QUERY friend bob ?
</computation> → Bob is friends with: alice"
```

**Classification Rules:**
```
Input: "All birds are animals. Tweety is a bird. What is Tweety?"
Expected Output: "I'll model the classification hierarchy: <computation>
REL is_bird
REL is_animal
FACT is_bird tweety
RULE is_animal: SCAN is_bird, EMIT is_animal $0 $1
SOLVE
QUERY is_animal tweety ?
</computation> → Tweety is an animal."
```

### 2.2 Multi-Step Inference Chains

**Transitive Relationships:**
```
Input: "Alice supervises Bob, Bob supervises Charlie. Who does Alice supervise indirectly?"
Expected Output: "I need to find transitive supervision relationships: <computation>
REL supervises
REL manages
FACT supervises alice bob
FACT supervises bob charlie
RULE manages: SCAN supervises, EMIT manages $0 $1
RULE manages: SCAN supervises, JOIN manages $1, EMIT manages $0 $2
SOLVE
QUERY manages alice ?
</computation> → Alice manages: bob (directly), charlie (indirectly)"
```

**Graph Reachability:**
```
Input: "There are roads from A to B, B to C, C to D. Can you get from A to D?"
Expected Output: "I'll check road connectivity: <computation>
REL road
REL reachable
FACT road a b
FACT road b c  
FACT road c d
RULE reachable: SCAN road, EMIT reachable $0 $1
RULE reachable: SCAN road, JOIN reachable $1, EMIT reachable $0 $2
SOLVE
QUERY reachable a d
</computation> → Yes, you can reach D from A via B→C→D"
```

### 2.3 Complex Pattern Matching

**Family Relationships:**
```
Input: "Model a family tree: Alice's children are Bob and Carol. Bob's child is David. Who are Alice's grandchildren?"
Expected Output: "I'll model grandparent relationships: <computation>
REL parent
REL grandparent
FACT parent alice bob
FACT parent alice carol
FACT parent bob david
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2
SOLVE
QUERY grandparent alice ?
</computation> → Alice's grandchildren: david"
```

**Social Networks:**
```
Input: "In a social network, find friends-of-friends. Alice follows Bob, Bob follows Carol and David. Who should Alice know?"
Expected Output: "I'll find second-degree connections: <computation>
REL follows
REL friend_of_friend
FACT follows alice bob
FACT follows bob carol
FACT follows bob david
RULE friend_of_friend: SCAN follows, JOIN follows $1, EMIT friend_of_friend $0 $2
SOLVE
QUERY friend_of_friend alice ?
</computation> → Alice might know: carol, david (through bob)"
```

---

## Stage 3: Advanced Logic Patterns (Weeks 5-6)

### 3.1 Complex Queries and Filtering

**Conditional Logic:**
```
Input: "Find people who manage others but are not managed by anyone (top-level managers)"
Expected Output: "I need to find managers with no supervisor: <computation>
REL manages  
REL is_manager
REL has_supervisor
FACT manages alice bob
FACT manages alice carol
FACT manages bob david
RULE is_manager: SCAN manages, EMIT is_manager $0 1
RULE has_supervisor: SCAN manages, EMIT has_supervisor $1 1
SOLVE
QUERY is_manager ? 1
</computation>
Then filter out those with supervisors to find: alice (top-level manager)"
```

### 3.2 Multiple Rule Sets

**Organizational Hierarchy:**
```
Input: "Model an org chart with direct reports and skip-level reports"
Expected Output: "I'll model reporting relationships at multiple levels: <computation>
REL reports_to
REL direct_report
REL skip_level_report
FACT reports_to bob alice
FACT reports_to carol alice  
FACT reports_to david bob
FACT reports_to eve bob
RULE direct_report: SCAN reports_to, EMIT direct_report $1 $0
RULE skip_level_report: SCAN reports_to, JOIN reports_to $0, EMIT skip_level_report $2 $1
SOLVE
QUERY direct_report alice ?
QUERY skip_level_report alice ?
</computation> → Alice's direct reports: bob, carol. Skip-level reports: david, eve"
```

### 3.3 Graph Algorithms

**Connected Components:**
```
Input: "Find all people in the same social group as Alice"
Expected Output: "I'll find Alice's connected component: <computation>
REL friend
REL connected
FACT friend alice bob
FACT friend bob carol
FACT friend david eve
RULE connected: SCAN friend, EMIT connected $0 $1
RULE connected: SCAN friend, EMIT connected $1 $0  
RULE connected: SCAN connected, JOIN connected $1, EMIT connected $0 $2
SOLVE
QUERY connected alice ?
</computation> → Alice's social group: bob, carol"
```

**Shortest Path Reasoning:**
```
Input: "What's the shortest connection between Alice and David through mutual contacts?"
Expected Output: "I'll trace connection paths: <computation>
REL knows
REL path_1
REL path_2
REL path_3
FACT knows alice bob
FACT knows bob carol
FACT knows carol david
RULE path_1: SCAN knows, EMIT path_1 $0 $1
RULE path_2: SCAN knows, JOIN path_1 $1, EMIT path_2 $0 $2
RULE path_3: SCAN knows, JOIN path_2 $1, EMIT path_3 $0 $2
SOLVE
QUERY path_3 alice david
</computation> → Alice→Bob→Carol→David (3 steps)"
```

---

## Stage 4: Hybrid Logic + Calculation (Weeks 7-8)

### 4.1 ByteLogic 3.0 Calculation Integration

**Mathematical Reasoning:**
```
Input: "Calculate compound interest and determine if Alice can afford a car costing $25,000"
Expected Output: "I'll calculate Alice's savings growth: <computation>
CALC compound_interest
  INPUT $principal $rate $years
  LET $amount = $principal * POW(1 + $rate, $years)
  RESULT $amount
END

REL can_afford
LET $alice_savings = CALC compound_interest(20000, 0.05, 2)
IF $alice_savings >= 25000 THEN
  FACT can_afford alice car
END
SOLVE
QUERY can_afford alice car
</computation> → Alice will have $22,050 after 2 years, cannot afford the $25,000 car"
```

### 4.2 Data Processing Patterns

**Iterative Calculations:**
```
Input: "Calculate the total network value of Alice's social connections"
Expected Output: "I'll sum connection values: <computation>
REL connection
REL network_value  
FACT connection alice bob
FACT connection alice carol
FACT connection bob david

CALC total_value
  INPUT $person
  LET $total = 0
  FOR conn IN (QUERY connection $person ?)
    LET $value = CALC person_value(conn.b)
    LET $total = $total + $value
  END
  RESULT $total
END

CALC person_value
  INPUT $person
  IF $person == "bob" THEN RESULT 1000
  ELSE IF $person == "carol" THEN RESULT 1500  
  ELSE RESULT 500
  END
END

LET $alice_network = CALC total_value("alice")
RESULT $alice_network
</computation> → Alice's network value: $2,500"
```

### 4.3 String Processing and Logic

**Text Analysis:**
```
Input: "Count how many people have names starting with 'A' in our database"
Expected Output: "I'll analyze names by first letter: <computation>
REL person
REL starts_with_a
FACT person alice
FACT person bob  
FACT person andrew
FACT person carol

CALC check_first_letter
  INPUT $name
  LET $first_char = CHAR_AT($name, 0)
  IF $first_char == "a" THEN RESULT 1
  ELSE RESULT 0
  END
END

CALC count_a_names
  LET $count = 0
  FOR p IN (QUERY person ? ?)
    LET $is_a = CALC check_first_letter(p.a)
    LET $count = $count + $is_a
  END
  RESULT $count
END

RESULT CALC count_a_names()
</computation> → 2 people have names starting with 'A': alice, andrew"
```

---

## Stage 5: Real-World Applications (Weeks 9-10)

### 5.1 Business Logic

**Inventory Management:**
```
Input: "Check which products need restocking (current stock < minimum threshold)"
Expected Output: "I'll analyze inventory levels: <computation>
REL product
REL current_stock  
REL min_threshold
REL needs_restock

FACT product widget_a
FACT product widget_b
FACT product widget_c
FACT current_stock widget_a 5
FACT current_stock widget_b 25  
FACT current_stock widget_c 8
FACT min_threshold widget_a 10
FACT min_threshold widget_b 20
FACT min_threshold widget_c 15

CALC check_restock
  FOR p IN (QUERY product ? ?)
    FOR current IN (QUERY current_stock p.a ?)
      FOR min_val IN (QUERY min_threshold p.a ?)
        IF current.b < min_val.b THEN
          FACT needs_restock p.a 1
        END
      END
    END
  END
END

EXEC check_restock
SOLVE  
QUERY needs_restock ? 1
</computation> → Products needing restock: widget_a (5<10), widget_c (8<15)"
```

### 5.2 Decision Support Systems

**Loan Approval Logic:**
```
Input: "Determine loan eligibility based on credit score, income, and debt-to-income ratio"
Expected Output: "I'll evaluate loan criteria: <computation>
REL applicant
REL credit_score
REL income
REL debt
REL eligible

FACT applicant alice
FACT applicant bob
FACT credit_score alice 750
FACT credit_score bob 620
FACT income alice 75000
FACT income bob 45000  
FACT debt alice 25000
FACT debt bob 35000

CALC evaluate_eligibility
  FOR app IN (QUERY applicant ? ?)
    FOR credit IN (QUERY credit_score app.a ?)
      FOR inc IN (QUERY income app.a ?)
        FOR debt_amt IN (QUERY debt app.a ?)
          LET $debt_ratio = debt_amt.b / inc.b
          IF credit.b >= 650 AND inc.b >= 50000 AND $debt_ratio <= 0.4 THEN
            FACT eligible app.a 1
          END
        END
      END
    END
  END
END

EXEC evaluate_eligibility
SOLVE
QUERY eligible ? 1
</computation> → Eligible applicants: alice (credit: 750, income: $75k, debt ratio: 33%)"
```

### 5.3 Knowledge Base Reasoning

**Medical Diagnosis Support:**
```
Input: "Based on symptoms and medical rules, what conditions should be considered?"
Expected Output: "I'll analyze symptom patterns: <computation>
REL symptom
REL condition
REL indicates
REL patient_symptom
REL possible_condition

FACT symptom fever
FACT symptom cough
FACT symptom fatigue
FACT condition flu
FACT condition cold
FACT condition pneumonia

FACT indicates fever flu
FACT indicates cough flu
FACT indicates fever cold
FACT indicates cough cold
FACT indicates fever pneumonia
FACT indicates cough pneumonia
FACT indicates fatigue pneumonia

FACT patient_symptom patient_1 fever
FACT patient_symptom patient_1 cough
FACT patient_symptom patient_1 fatigue

CALC symptom_match_score
  INPUT $patient $condition
  LET $matches = 0
  LET $total_symptoms = 0
  
  FOR ps IN (QUERY patient_symptom $patient ?)
    LET $total_symptoms = $total_symptoms + 1
    FOR ind IN (QUERY indicates ps.b $condition)
      LET $matches = $matches + 1
    END
  END
  
  IF $total_symptoms > 0 THEN
    RESULT $matches / $total_symptoms
  ELSE
    RESULT 0
  END
END

CALC diagnose
  FOR c IN (QUERY condition ? ?)
    LET $score = CALC symptom_match_score("patient_1", c.a)
    IF $score >= 0.5 THEN
      FACT possible_condition patient_1 c.a
    END
  END
END

EXEC diagnose
SOLVE
QUERY possible_condition patient_1 ?
</computation> → Possible conditions for patient_1: flu (67% match), pneumonia (100% match)"
```

---

## Training Data Generation Strategy

### Data Volume Targets

| Stage | Examples | Difficulty | Focus Areas |
|-------|----------|------------|-------------|
| 1 | 500 | Basic | Syntax, simple queries |
| 2 | 750 | Intermediate | Single-step inference |  
| 3 | 1000 | Advanced | Multi-step reasoning |
| 4 | 750 | Expert | Hybrid logic+calculation |
| 5 | 500 | Real-world | Applied scenarios |
| **Total** | **3500** | | **Comprehensive coverage** |

### Template Structure

Each training example follows this structure:
```json
{
  "stage": "2-inference",
  "difficulty": "intermediate", 
  "topic": "transitive_closure",
  "input": "Human question or scenario",
  "reasoning_steps": ["Step 1", "Step 2", "..."],
  "bytelogic_code": "REL ... FACT ... RULE ... SOLVE QUERY ...",
  "expected_output": "Human-readable result",
  "validation": {
    "syntax_valid": true,
    "compiles": true,
    "expected_result": ["expected", "values"]
  }
}
```

### Quality Assurance

**Automated Validation:**
1. Syntax checking with ByteLogic parser
2. Compilation testing (bl → wat → wasm)
3. Execution testing with expected results
4. Performance benchmarking

**Human Review:**
1. Logical correctness of reasoning
2. Natural language quality
3. Progressive difficulty validation
4. Real-world applicability

### Curriculum Progression Testing

**Skills Assessment:**
- Stage 1: Basic syntax generation (95% accuracy)
- Stage 2: Simple inference rules (90% accuracy)  
- Stage 3: Complex multi-step reasoning (85% accuracy)
- Stage 4: Hybrid reasoning (80% accuracy)
- Stage 5: Real-world applications (75% accuracy)

**Graduation Criteria:**
Each stage requires 85%+ accuracy on held-out test set before progressing to next stage.

---

## Implementation Notes

### Integration with Existing Pipeline

1. **Data Format Compatibility:**
   - Extend existing WASM dataset format
   - Add ByteLogic-specific fields
   - Maintain backward compatibility

2. **Tokenization Updates:**
   - Add ByteLogic keywords to vocabulary
   - Handle `<computation>` tokens
   - Support variable syntax ($0, $1, etc.)

3. **Execution Pipeline:**
   - ByteLogic → WAT compilation
   - WAT → WASM compilation  
   - WASM execution and result parsing

### Performance Considerations

**Compilation Speed:**
- Cache compiled WAT for repeated patterns
- Optimize ByteLogic compiler flags
- Batch compilation for training

**Memory Usage:**
- Limit maximum program size
- Implement garbage collection for WASM instances
- Monitor memory consumption during training

**Training Efficiency:**
- Progressive curriculum reduces training time
- Validation caching speeds up iteration
- Parallel execution for batch processing

---

This comprehensive curriculum provides structured learning progression from basic ByteLogic syntax to sophisticated real-world reasoning applications, ensuring the LLM develops robust logical reasoning capabilities within the WASM sandbox environment.