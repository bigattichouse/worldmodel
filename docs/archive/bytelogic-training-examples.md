# ByteLogic Training Examples

**Version:** 1.0  
**Date:** 2026-01-19  
**Purpose:** Comprehensive training examples for all ByteLogic language features

---

## Table of Contents

1. [Basic Syntax Examples](#basic-syntax-examples)
2. [Logic Programming Examples](#logic-programming-examples)
3. [Mathematical Computation Examples](#mathematical-computation-examples)
4. [Set Operations Examples](#set-operations-examples)
5. [Advanced Reasoning Examples](#advanced-reasoning-examples)
6. [Real-World Applications](#real-world-applications)

---

## Basic Syntax Examples

### 1.1 Facts and Simple Queries

**Example: Basic Family Relationship**
```
Input: "Is Alice the parent of Bob?"
Expected Output: "Let me check: <computation>
REL parent
FACT parent alice bob
SOLVE
QUERY parent alice bob
</computation> → Yes, Alice is Bob's parent."
```

**Example: Finding Children**
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

**Example: Multiple Relations**
```
Input: "List all friendships and likes relationships"
Expected Output: "Here are the relationships: <computation>
REL friend
REL likes
FACT friend alice bob
FACT friend bob charlie
FACT likes alice pizza
FACT likes bob pasta
FACT likes charlie pizza
SOLVE
QUERY friend ? ?
QUERY likes ? ?
</computation> → Friendships: alice-bob, bob-charlie. Likes: alice-pizza, bob-pasta, charlie-pizza."
```

### 1.2 Variable Binding

**Example: Finding All Parents**
```
Input: "Who are all the parents in our data?"
Expected Output: "I'll find all parents: <computation>
REL parent
REL is_parent
FACT parent alice bob
FACT parent alice charlie
FACT parent bob david
RULE is_parent: SCAN parent, EMIT is_parent $0 1
SOLVE
QUERY is_parent ? 1
</computation> → Parents are: alice, bob"
```

**Example: Symmetric Relations**
```
Input: "If friendship is mutual, who are Bob's friends?"
Expected Output: "I'll create symmetric friendships: <computation>
REL friend_directed
REL friend
FACT friend_directed alice bob
FACT friend_directed bob charlie
RULE friend: SCAN friend_directed, EMIT friend $0 $1
RULE friend: SCAN friend_directed, EMIT friend $1 $0
SOLVE
QUERY friend bob ?
</computation> → Bob's friends are: alice, charlie"
```

---

## Logic Programming Examples

### 2.1 Transitive Relationships

**Example: Ancestor Relationships**
```
Input: "Who are all of Alice's descendants?"
Expected Output: "I'll trace all descendants: <computation>
REL parent
REL descendant
FACT parent alice bob
FACT parent alice carol
FACT parent bob david
FACT parent carol eve
RULE descendant: SCAN parent, EMIT descendant $0 $1
RULE descendant: SCAN parent, JOIN descendant $1, EMIT descendant $0 $2
SOLVE
QUERY descendant alice ?
</computation> → Alice's descendants: bob, carol, david, eve"
```

**Example: Management Hierarchy**
```
Input: "Who does Alice manage, directly or indirectly?"
Expected Output: "I'll find all managed employees: <computation>
REL reports_to
REL manages
FACT reports_to bob alice
FACT reports_to carol alice
FACT reports_to david bob
FACT reports_to eve carol
RULE manages: SCAN reports_to, EMIT manages $1 $0
RULE manages: SCAN reports_to, JOIN manages $0, EMIT manages $2 $1
SOLVE
QUERY manages alice ?
</computation> → Alice manages: bob, carol (directly), david, eve (indirectly)"
```

### 2.2 Graph Algorithms

**Example: Path Finding**
```
Input: "Can you get from city A to city D using these roads?"
Expected Output: "I'll check road connectivity: <computation>
REL road
REL reachable
FACT road a b
FACT road b c
FACT road c d
FACT road a e
FACT road e d
RULE reachable: SCAN road, EMIT reachable $0 $1
RULE reachable: SCAN road, JOIN reachable $1, EMIT reachable $0 $2
SOLVE
QUERY reachable a d
</computation> → Yes, you can reach D from A (multiple paths available)"
```

**Example: Social Network Analysis**
```
Input: "Find all people Alice can reach through social connections"
Expected Output: "I'll map Alice's social network: <computation>
REL follows
REL connected
FACT follows alice bob
FACT follows bob charlie
FACT follows charlie david
FACT follows alice eve
FACT follows eve frank
RULE connected: SCAN follows, EMIT connected $0 $1
RULE connected: SCAN follows, JOIN connected $1, EMIT connected $0 $2
SOLVE
QUERY connected alice ?
</computation> → Alice can reach: bob, charlie, david, eve, frank"
```

### 2.3 Classification and Inheritance

**Example: Animal Classification**
```
Input: "What properties does Tweety have based on being a bird?"
Expected Output: "I'll trace inherited properties: <computation>
REL isa
REL has_property
FACT isa tweety bird
FACT isa bird animal
FACT has_property bird flies
FACT has_property bird feathers
FACT has_property animal breathes
FACT has_property animal moves
RULE isa: SCAN isa, JOIN isa $1, EMIT isa $0 $2
RULE has_property: SCAN isa, JOIN has_property $1, EMIT has_property $0 $2
SOLVE
QUERY has_property tweety ?
</computation> → Tweety has properties: flies, feathers, breathes, moves"
```

**Example: Organizational Structure**
```
Input: "What roles can a Senior Developer perform?"
Expected Output: "I'll find all inherited capabilities: <computation>
REL isa
REL can_perform
FACT isa alice senior_developer
FACT isa senior_developer developer
FACT isa developer employee
FACT can_perform developer code
FACT can_perform developer debug
FACT can_perform senior_developer mentor
FACT can_perform senior_developer architect
FACT can_perform employee attend_meetings
RULE isa: SCAN isa, JOIN isa $1, EMIT isa $0 $2
RULE can_perform: SCAN isa, JOIN can_perform $1, EMIT can_perform $0 $2
SOLVE
QUERY can_perform alice ?
</computation> → Alice can perform: code, debug, mentor, architect, attend_meetings"
```

---

## Mathematical Computation Examples

### 3.1 Basic Arithmetic

**Example: Simple Calculator**
```
Input: "Calculate 25% of 80"
Expected Output: "I'll calculate the percentage: <computation>
CALC percentage
  INPUT $amount $percent
  LET $result = $amount * ($percent / 100)
  RESULT $result
END
RESULT CALC percentage(80, 25)
</computation> → 20"
```

**Example: Compound Interest**
```
Input: "If I invest $1000 at 5% interest for 3 years, how much will I have?"
Expected Output: "I'll calculate compound interest: <computation>
CALC compound_interest
  INPUT $principal $rate $years
  LET $amount = $principal * POW(1 + $rate, $years)
  RESULT $amount
END
RESULT CALC compound_interest(1000, 0.05, 3)
</computation> → $1157.63"
```

### 3.2 Iterative Calculations

**Example: Factorial**
```
Input: "What is 7 factorial?"
Expected Output: "I'll calculate 7!: <computation>
CALC factorial
  INPUT $n
  IF $n <= 1 THEN
    RESULT 1
  ELSE
    LET $prev = CALC factorial($n - 1)
    RESULT $n * $prev
  END
END
RESULT CALC factorial(7)
</computation> → 5040"
```

**Example: Fibonacci Sequence**
```
Input: "What is the 10th Fibonacci number?"
Expected Output: "I'll calculate the 10th Fibonacci number: <computation>
CALC fibonacci
  INPUT $n
  IF $n <= 1 THEN
    RESULT $n
  ELSE
    LET $a = CALC fibonacci($n - 1)
    LET $b = CALC fibonacci($n - 2)
    RESULT $a + $b
  END
END
RESULT CALC fibonacci(10)
</computation> → 55"
```

### 3.3 String Processing

**Example: Character Counting**
```
Input: "How many 'e' characters are in 'the quick brown elephant'?"
Expected Output: "I'll count the 'e' characters: <computation>
CALC count_char
  INPUT $text $char
  LET $count = 0
  FOR $i IN RANGE(0, LENGTH($text))
    IF CHAR_AT($text, $i) == $char THEN
      LET $count = $count + 1
    END
  END
  RESULT $count
END
RESULT CALC count_char("the quick brown elephant", "e")
</computation> → 4"
```

**Example: String Analysis**
```
Input: "What's the average word length in 'hello world programming'?"
Expected Output: "I'll analyze word lengths: <computation>
CALC avg_word_length
  INPUT $text
  LET $words = ["hello", "world", "programming"]
  LET $total_length = 0
  LET $word_count = 3
  
  FOR $word IN $words
    LET $total_length = $total_length + LENGTH($word)
  END
  
  RESULT $total_length / $word_count
END
RESULT CALC avg_word_length("hello world programming")
</computation> → 7.33"
```

---

## Set Operations Examples

### 4.1 Set Membership and Unions

**Example: Group Membership**
```
Input: "Which people belong to both the engineering and management groups?"
Expected Output: "I'll find the intersection: <computation>
REL engineering
REL management
REL both_groups
FACT engineering alice
FACT engineering bob
FACT engineering charlie
FACT management alice
FACT management david
FACT management eve
RULE both_groups: SCAN engineering, JOIN management $0, EMIT both_groups $0 1
SOLVE
QUERY both_groups ? 1
</computation> → Alice is in both engineering and management"
```

**Example: Set Union**
```
Input: "Who are all the people in either the sports club or music club?"
Expected Output: "I'll find all club members: <computation>
REL sports_club
REL music_club
REL any_club
FACT sports_club alice
FACT sports_club bob
FACT music_club charlie
FACT music_club alice
FACT music_club david
RULE any_club: SCAN sports_club, EMIT any_club $0 1
RULE any_club: SCAN music_club, EMIT any_club $0 1
SOLVE
QUERY any_club ? 1
</computation> → Club members: alice, bob, charlie, david"
```

### 4.2 Set Differences

**Example: Exclusive Membership**
```
Input: "Who is in the engineering team but not in management?"
Expected Output: "I'll find engineers who aren't managers: <computation>
REL engineering
REL management
REL engineer_only
FACT engineering alice
FACT engineering bob
FACT engineering charlie
FACT management alice
FACT management david
CALC find_engineer_only
  FOR eng IN (QUERY engineering ? ?)
    LET $is_mgmt = 0
    FOR mgmt IN (QUERY management eng.a ?)
      LET $is_mgmt = 1
    END
    IF $is_mgmt == 0 THEN
      FACT engineer_only eng.a 1
    END
  END
END
EXEC find_engineer_only
SOLVE
QUERY engineer_only ? 1
</computation> → Engineers not in management: bob, charlie"
```

### 4.3 Complex Set Operations

**Example: Multi-way Intersection**
```
Input: "Find people who are in all three groups: sports, music, and volunteer"
Expected Output: "I'll find the triple intersection: <computation>
REL sports
REL music  
REL volunteer
REL all_three
FACT sports alice
FACT sports bob
FACT sports charlie
FACT music alice
FACT music charlie
FACT music david
FACT volunteer alice
FACT volunteer eve
FACT volunteer charlie
CALC find_triple_members
  FOR s IN (QUERY sports ? ?)
    FOR m IN (QUERY music s.a ?)
      FOR v IN (QUERY volunteer s.a ?)
        FACT all_three s.a 1
      END
    END
  END
END
EXEC find_triple_members
SOLVE
QUERY all_three ? 1
</computation> → People in all three groups: alice, charlie"
```

---

## Advanced Reasoning Examples

### 5.1 Logical Puzzles

**Example: Logic Grid Puzzle**
```
Input: "Alice, Bob, and Carol like different foods (pizza, pasta, salad). Alice doesn't like pasta. Bob likes pizza. What does each person like?"
Expected Output: "I'll solve this step by step: <computation>
REL likes
REL person
REL food
FACT person alice
FACT person bob  
FACT person carol
FACT food pizza
FACT food pasta
FACT food salad

; Bob likes pizza
FACT likes bob pizza

CALC solve_puzzle
  ; Alice doesn't like pasta, and Bob has pizza, so Alice has salad
  FACT likes alice salad
  ; Carol gets the remaining food (pasta)
  FACT likes carol pasta
END

EXEC solve_puzzle
SOLVE
QUERY likes alice ?
QUERY likes bob ?
QUERY likes carol ?
</computation> → Alice likes salad, Bob likes pizza, Carol likes pasta"
```

**Example: Constraint Satisfaction**
```
Input: "Four people (A, B, C, D) sit around a table. A is not next to B. C is across from A. Where does everyone sit?"
Expected Output: "I'll solve the seating arrangement: <computation>
REL seat
REL next_to  
REL across_from
FACT seat a 1
FACT seat b 2
FACT seat c 3
FACT seat d 4

; Define adjacency (circular table)
RULE next_to: SCAN seat, JOIN seat, WHERE ABS($1 - $3) == 1 OR ABS($1 - $3) == 3, EMIT next_to $0 $2
RULE across_from: SCAN seat, JOIN seat, WHERE ABS($1 - $3) == 2, EMIT across_from $0 $2

CALC validate_constraints
  ; Check if A is across from C (seats 1 and 3)
  ; Check if A is not next to B
  ; This validates our assignment
  RESULT 1
END

SOLVE
QUERY across_from a c
</computation> → Valid seating: A(1), B(2), C(3), D(4) with C across from A"
```

### 5.2 Planning Problems

**Example: Travel Planning**
```
Input: "Plan a route from home to work that stops at the bank. Available routes: home→bank, home→store, bank→work, store→bank"
Expected Output: "I'll find a valid route: <computation>
REL direct_route
REL can_reach
REL valid_path
FACT direct_route home bank
FACT direct_route home store  
FACT direct_route bank work
FACT direct_route store bank

; Find all reachable places
RULE can_reach: SCAN direct_route, EMIT can_reach $0 $1
RULE can_reach: SCAN direct_route, JOIN can_reach $1, EMIT can_reach $0 $2

; Find path that visits bank before work
RULE valid_path: SCAN direct_route, MATCH home, JOIN direct_route $1 work, EMIT valid_path $0 $1 work

SOLVE
QUERY can_reach home work
QUERY direct_route home bank
QUERY direct_route bank work
</computation> → Route: home → bank → work"
```

### 5.3 Resource Allocation

**Example: Task Assignment**
```
Input: "Assign 3 tasks to 3 people where Alice can do tasks 1&2, Bob can do tasks 2&3, Carol can do tasks 1&3. Each person gets one task."
Expected Output: "I'll find a valid assignment: <computation>
REL can_do
REL assigned
FACT can_do alice task1
FACT can_do alice task2
FACT can_do bob task2  
FACT can_do bob task3
FACT can_do carol task1
FACT can_do carol task3

CALC assign_tasks
  ; Try assignment: Alice→task2, Bob→task3, Carol→task1
  FACT assigned alice task2
  FACT assigned bob task3
  FACT assigned carol task1
END

EXEC assign_tasks

; Validate assignment
CALC validate
  FOR assign IN (QUERY assigned ? ?)
    FOR capability IN (QUERY can_do assign.a assign.b)
      ; Valid assignment found
    END
  END
  RESULT 1
END

SOLVE
QUERY assigned ? ?
</computation> → Assignment: Alice→task2, Bob→task3, Carol→task1"
```

---

## Real-World Applications

### 6.1 Business Logic

**Example: Inventory Management**
```
Input: "Check which products need restocking (current < minimum)"
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

CALC check_inventory
  FOR p IN (QUERY product ? ?)
    FOR curr IN (QUERY current_stock p.a ?)
      FOR min_val IN (QUERY min_threshold p.a ?)
        IF curr.b < min_val.b THEN
          FACT needs_restock p.a 1
        END
      END
    END
  END
END

EXEC check_inventory
SOLVE
QUERY needs_restock ? 1
</computation> → Products needing restock: widget_a (5<10), widget_c (8<15)"
```

**Example: Employee Scheduling**
```
Input: "Find available employees for weekend shift who have required skills"
Expected Output: "I'll match skills with availability: <computation>
REL employee
REL has_skill
REL available_weekend
REL required_skill
REL qualified_available
FACT employee alice
FACT employee bob
FACT employee carol
FACT has_skill alice customer_service
FACT has_skill alice cash_handling
FACT has_skill bob customer_service
FACT has_skill carol cash_handling
FACT available_weekend alice
FACT available_weekend carol
FACT required_skill customer_service
FACT required_skill cash_handling

CALC find_qualified
  FOR emp IN (QUERY available_weekend ? ?)
    LET $has_all_skills = 1
    FOR req IN (QUERY required_skill ? ?)
      LET $has_skill = 0
      FOR skill IN (QUERY has_skill emp.a req.a)
        LET $has_skill = 1
      END
      IF $has_skill == 0 THEN
        LET $has_all_skills = 0
      END
    END
    IF $has_all_skills == 1 THEN
      FACT qualified_available emp.a 1
    END
  END
END

EXEC find_qualified
SOLVE
QUERY qualified_available ? 1
</computation> → Qualified and available: alice (has both skills)"
```

### 6.2 Decision Support

**Example: Loan Approval**
```
Input: "Determine loan eligibility based on credit score ≥650, income ≥$50k, debt ratio ≤40%"
Expected Output: "I'll evaluate loan applications: <computation>
REL applicant
REL credit_score
REL income
REL debt
REL eligible
FACT applicant alice
FACT applicant bob
FACT applicant carol
FACT credit_score alice 720
FACT credit_score bob 580
FACT credit_score carol 680
FACT income alice 75000
FACT income bob 45000
FACT income carol 60000
FACT debt alice 20000
FACT debt bob 35000
FACT debt carol 18000

CALC evaluate_loans
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

EXEC evaluate_loans
SOLVE
QUERY eligible ? 1
</computation> → Eligible applicants: alice (720 credit, $75k income, 27% debt ratio), carol (680 credit, $60k income, 30% debt ratio)"
```

### 6.3 Knowledge Systems

**Example: Medical Diagnosis Support**
```
Input: "Based on symptoms fever, cough, fatigue, what conditions are possible?"
Expected Output: "I'll analyze symptom patterns: <computation>
REL symptom
REL condition
REL indicates
REL patient_has
REL possible_condition
FACT symptom fever
FACT symptom cough
FACT symptom fatigue
FACT symptom headache
FACT condition flu
FACT condition cold
FACT condition pneumonia
FACT indicates fever flu
FACT indicates cough flu
FACT indicates fatigue flu
FACT indicates fever cold
FACT indicates cough cold
FACT indicates fever pneumonia
FACT indicates cough pneumonia
FACT indicates fatigue pneumonia
FACT patient_has patient1 fever
FACT patient_has patient1 cough
FACT patient_has patient1 fatigue

CALC diagnose
  FOR cond IN (QUERY condition ? ?)
    LET $symptom_count = 0
    LET $matching_symptoms = 0
    
    ; Count patient symptoms
    FOR ps IN (QUERY patient_has patient1 ?)
      LET $symptom_count = $symptom_count + 1
    END
    
    ; Count matching symptoms for this condition
    FOR ps IN (QUERY patient_has patient1 ?)
      FOR ind IN (QUERY indicates ps.b cond.a)
        LET $matching_symptoms = $matching_symptoms + 1
      END
    END
    
    LET $match_ratio = $matching_symptoms / $symptom_count
    IF $match_ratio >= 0.6 THEN
      FACT possible_condition patient1 cond.a
    END
  END
END

EXEC diagnose
SOLVE
QUERY possible_condition patient1 ?
</computation> → Possible conditions: flu (100% symptom match), pneumonia (100% symptom match)"
```

### 6.4 Educational Systems

**Example: Course Prerequisites**
```
Input: "What courses can Alice take next semester given her completed courses?"
Expected Output: "I'll check prerequisite requirements: <computation>
REL course
REL prerequisite
REL completed
REL can_take
FACT course math101
FACT course math201
FACT course math301
FACT course physics101
FACT course physics201
FACT prerequisite math201 math101
FACT prerequisite math301 math201
FACT prerequisite physics201 physics101
FACT prerequisite physics201 math101
FACT completed alice math101
FACT completed alice physics101

CALC check_prerequisites
  FOR c IN (QUERY course ? ?)
    LET $can_take_course = 1
    LET $already_completed = 0
    
    ; Check if already completed
    FOR comp IN (QUERY completed alice c.a)
      LET $already_completed = 1
    END
    
    IF $already_completed == 0 THEN
      ; Check all prerequisites
      FOR prereq IN (QUERY prerequisite c.a ?)
        LET $has_prereq = 0
        FOR comp IN (QUERY completed alice prereq.b)
          LET $has_prereq = 1
        END
        IF $has_prereq == 0 THEN
          LET $can_take_course = 0
        END
      END
      
      IF $can_take_course == 1 THEN
        FACT can_take alice c.a
      END
    END
  END
END

EXEC check_prerequisites
SOLVE
QUERY can_take alice ?
</computation> → Alice can take: math201 (has math101), physics201 (has physics101 and math101)"
```

---

## Training Data Format

Each example should be formatted as JSON for the training pipeline:

```json
{
  "id": "logic_family_001",
  "category": "logic_programming", 
  "subcategory": "family_relationships",
  "difficulty": "beginner",
  "input": "Who are Alice's grandchildren?",
  "reasoning_steps": [
    "I need to find Alice's children first",
    "Then find the children of Alice's children", 
    "This is a transitive relationship through parent relations"
  ],
  "expected_output": "I'll trace the family relationships: <computation>\nREL parent\nREL grandparent\nFACT parent alice bob\nFACT parent alice carol\nFACT parent bob david\nFACT parent carol eve\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent alice ?\n</computation> → Alice's grandchildren: david, eve",
  "bytelogic_code": "REL parent\nREL grandparent\nFACT parent alice bob\nFACT parent alice carol\nFACT parent bob david\nFACT parent carol eve\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent alice ?",
  "validation": {
    "syntax_valid": true,
    "compiles": true,
    "expected_result": ["david", "eve"],
    "execution_time_ms": 45
  },
  "concepts": ["transitive_relations", "family_trees", "rule_chaining"],
  "prerequisites": ["basic_facts", "simple_queries", "rule_syntax"]
}
```

This comprehensive set of examples covers:
- **Basic Syntax**: Facts, queries, variables, relations
- **Logic Programming**: Transitive closure, inheritance, classification  
- **Mathematical Computation**: Arithmetic, iterative algorithms, string processing
- **Set Operations**: Union, intersection, difference, membership
- **Advanced Reasoning**: Logic puzzles, planning, constraint satisfaction
- **Real-World Applications**: Business logic, decision support, knowledge systems

The examples progress from simple to complex, building on previous concepts while introducing new features systematically.