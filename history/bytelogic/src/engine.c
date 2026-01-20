/* ═══════════════════════════════════════════════════════════════════════════
 * engine.c - ByteLog Execution Engine Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Datalog execution engine with fixpoint computation.
 * Evaluates ByteLog programs and answers queries.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "engine.h"
#include "parser.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* For strdup portability */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Provide strdup for systems that don't have it */
#if !defined(_GNU_SOURCE) && !defined(__GLIBC__)
static char* strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *result = malloc(len);
    if (result) {
        memcpy(result, s, len);
    }
    return result;
}
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Hash Function
 * ───────────────────────────────────────────────────────────────────────── */

static unsigned int hash_fact(const char *relation, int arg_a, int arg_b) {
    unsigned int hash = 5381;
    
    /* Hash relation name */
    for (const char *c = relation; *c; c++) {
        hash = ((hash << 5) + hash) + (unsigned char)*c;
    }
    
    /* Hash arguments */
    hash = ((hash << 5) + hash) + (unsigned int)arg_a;
    hash = ((hash << 5) + hash) + (unsigned int)arg_b;
    
    return hash % FACT_DATABASE_SIZE;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Fact Database Implementation
 * ───────────────────────────────────────────────────────────────────────── */

void factdb_init(FactDatabase *db) {
    memset(db->buckets, 0, sizeof(db->buckets));
    db->count = 0;
    db->capacity = FACT_DATABASE_SIZE;
}

void factdb_cleanup(FactDatabase *db) {
    for (int i = 0; i < FACT_DATABASE_SIZE; i++) {
        Fact *fact = db->buckets[i];
        while (fact) {
            Fact *next = fact->next;
            free(fact->relation);
            free(fact);
            fact = next;
        }
    }
    memset(db->buckets, 0, sizeof(db->buckets));
    db->count = 0;
}

bool factdb_add_fact(FactDatabase *db, const char *relation, int arg_a, int arg_b) {
    if (!relation) return false;
    
    /* Check if fact already exists */
    if (factdb_has_fact(db, relation, arg_a, arg_b)) {
        return true;  /* Already exists, no need to add */
    }
    
    /* Create new fact */
    Fact *fact = malloc(sizeof(Fact));
    if (!fact) return false;
    
    fact->relation = strdup(relation);
    if (!fact->relation) {
        free(fact);
        return false;
    }
    
    fact->arg_a = arg_a;
    fact->arg_b = arg_b;
    
    /* Add to hash table */
    unsigned int bucket = hash_fact(relation, arg_a, arg_b);
    fact->next = db->buckets[bucket];
    db->buckets[bucket] = fact;
    db->count++;
    
    return true;
}

bool factdb_has_fact(FactDatabase *db, const char *relation, int arg_a, int arg_b) {
    if (!relation) return false;
    
    unsigned int bucket = hash_fact(relation, arg_a, arg_b);
    Fact *fact = db->buckets[bucket];
    
    while (fact) {
        if (strcmp(fact->relation, relation) == 0 &&
            fact->arg_a == arg_a && fact->arg_b == arg_b) {
            return true;
        }
        fact = fact->next;
    }
    
    return false;
}

QueryResult* factdb_query(FactDatabase *db, const char *relation, int arg_a, int arg_b) {
    if (!relation) return NULL;
    
    QueryResult *results = NULL;
    QueryResult *tail = NULL;
    
    /* Search all buckets if we have wildcards, otherwise search specific bucket */
    if (arg_a == -1 || arg_b == -1) {
        /* Wildcard query - search all buckets */
        for (int i = 0; i < FACT_DATABASE_SIZE; i++) {
            Fact *fact = db->buckets[i];
            while (fact) {
                if (strcmp(fact->relation, relation) == 0 &&
                    (arg_a == -1 || fact->arg_a == arg_a) &&
                    (arg_b == -1 || fact->arg_b == arg_b)) {
                    
                    QueryResult *result = malloc(sizeof(QueryResult));
                    if (!result) break;
                    
                    result->arg_a = fact->arg_a;
                    result->arg_b = fact->arg_b;
                    result->next = NULL;
                    
                    if (tail) {
                        tail->next = result;
                    } else {
                        results = result;
                    }
                    tail = result;
                }
                fact = fact->next;
            }
        }
    } else {
        /* Exact query */
        if (factdb_has_fact(db, relation, arg_a, arg_b)) {
            QueryResult *result = malloc(sizeof(QueryResult));
            if (result) {
                result->arg_a = arg_a;
                result->arg_b = arg_b;
                result->next = NULL;
                results = result;
            }
        }
    }
    
    return results;
}

QueryResult* factdb_get_all(FactDatabase *db, const char *relation) {
    return factdb_query(db, relation, -1, -1);
}

void factdb_print(const FactDatabase *db, const AtomTable *atoms) {
    printf("Fact Database (%d facts):\n", db->count);
    printf("─────────────────────────\n");
    
    if (db->count == 0) {
        printf("  (empty)\n");
        return;
    }
    
    for (int i = 0; i < FACT_DATABASE_SIZE; i++) {
        Fact *fact = db->buckets[i];
        while (fact) {
            printf("  %s(", fact->relation);
            
            /* Try to get atom name for arg_a */
            const char *name_a = atom_table_name(atoms, fact->arg_a);
            if (name_a) {
                printf("%s", name_a);
            } else {
                printf("%d", fact->arg_a);
            }
            
            printf(", ");
            
            /* Try to get atom name for arg_b */
            const char *name_b = atom_table_name(atoms, fact->arg_b);
            if (name_b) {
                printf("%s", name_b);
            } else {
                printf("%d", fact->arg_b);
            }
            
            printf(")\n");
            fact = fact->next;
        }
    }
}

int factdb_count(const FactDatabase *db) {
    return db->count;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Query Result Implementation
 * ───────────────────────────────────────────────────────────────────────── */

void query_result_free(QueryResult *results) {
    while (results) {
        QueryResult *next = results->next;
        free(results);
        results = next;
    }
}

int query_result_count(QueryResult *results) {
    int count = 0;
    while (results) {
        count++;
        results = results->next;
    }
    return count;
}

void query_result_print(QueryResult *results, const char *relation, const AtomTable *atoms) {
    if (!results) {
        printf("  No results found.\n");
        return;
    }
    
    int count = 0;
    while (results) {
        printf("  %s(", relation);
        
        /* Try to get atom name for arg_a */
        const char *name_a = atom_table_name(atoms, results->arg_a);
        if (name_a) {
            printf("%s", name_a);
        } else {
            printf("%d", results->arg_a);
        }
        
        printf(", ");
        
        /* Try to get atom name for arg_b */
        const char *name_b = atom_table_name(atoms, results->arg_b);
        if (name_b) {
            printf("%s", name_b);
        } else {
            printf("%d", results->arg_b);
        }
        
        printf(")\n");
        results = results->next;
        count++;
    }
    
    if (count == 1) {
        printf("  Found 1 result.\n");
    } else {
        printf("  Found %d results.\n", count);
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Execution Engine Implementation
 * ───────────────────────────────────────────────────────────────────────── */

void engine_init(ExecutionEngine *engine) {
    factdb_init(&engine->facts);
    atom_table_init(&engine->atoms);
    memset(engine->error, 0, sizeof(engine->error));
    engine->error_count = 0;
    engine->debug = false;
}

void engine_cleanup(ExecutionEngine *engine) {
    factdb_cleanup(&engine->facts);
    atom_table_free(&engine->atoms);
    engine->error_count = 0;
}

static void engine_error(ExecutionEngine *engine, const char *message) {
    engine->error_count++;
    snprintf(engine->error, sizeof(engine->error), "%s", message);
}

bool engine_has_errors(ExecutionEngine *engine) {
    return engine->error_count > 0;
}

int engine_get_error_count(ExecutionEngine *engine) {
    return engine->error_count;
}

const char* engine_get_error(ExecutionEngine *engine) {
    return engine->error;
}

void engine_set_debug(ExecutionEngine *engine, bool debug) {
    engine->debug = debug;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Rule Evaluation (Fixpoint Computation)
 * ───────────────────────────────────────────────────────────────────────── */

static bool engine_evaluate_rule(ExecutionEngine *engine, const ASTNode *rule) {
    if (!rule || rule->type != AST_RULE) {
        engine_error(engine, "Invalid rule node");
        return false;
    }
    
    bool new_facts_added = false;
    const char *target = rule->data.rule.target;
    
    if (engine->debug) {
        printf("Evaluating rule for '%s'\n", target);
    }
    
    /* For each combination of facts that matches the rule body */
    ASTNode *body = rule->data.rule.body;
    ASTNode *emit = rule->data.rule.emit;
    
    if (!body || !emit || emit->type != AST_EMIT) {
        engine_error(engine, "Invalid rule structure");
        return false;
    }
    
    /* Simple case: SCAN + optional JOIN + EMIT */
    if (body->type == AST_SCAN) {
        /* Get all facts for the scanned relation */
        QueryResult *scan_results = factdb_get_all(&engine->facts, body->data.scan.relation);
        QueryResult *scan_iter = scan_results;
        
        while (scan_iter) {
            /* Variable bindings: $0 = match_var, $1 = scan_iter->arg_a, $2 = scan_iter->arg_b */
            int var_bindings[3] = {-1, scan_iter->arg_a, scan_iter->arg_b};
            
            /* Set match variable if specified */
            if (body->data.scan.has_match) {
                if (body->data.scan.match_var == 0) {
                    var_bindings[0] = scan_iter->arg_a;
                } else if (body->data.scan.match_var == 1) {
                    var_bindings[0] = scan_iter->arg_b;
                }
            }
            
            /* Check for JOIN operations */
            ASTNode *next_op = body->next;
            bool join_satisfied = true;
            
            while (next_op && join_satisfied) {
                if (next_op->type == AST_JOIN) {
                    /* Find facts that join on the specified variable */
                    int join_var = next_op->data.join.match_var;
                    if (join_var < 0 || join_var > 2 || var_bindings[join_var] == -1) {
                        join_satisfied = false;
                        break;
                    }
                    
                    /* Look for facts where first argument matches the join variable */
                    QueryResult *join_results = factdb_query(&engine->facts, 
                                                            next_op->data.join.relation,
                                                            var_bindings[join_var], -1);
                    
                    if (!join_results) {
                        join_satisfied = false;
                    } else {
                        /* Update variable bindings with join result */
                        /* For simplicity, take first result - more complex rules would iterate */
                        var_bindings[2] = join_results->arg_b;  /* $2 gets second arg of join */
                        query_result_free(join_results);
                    }
                }
                next_op = next_op->next;
            }
            
            /* If all joins satisfied, emit the fact */
            if (join_satisfied) {
                int emit_a = -1, emit_b = -1;
                
                /* Resolve emit variables */
                if (emit->data.emit.var_a >= 0 && emit->data.emit.var_a < 3) {
                    emit_a = var_bindings[emit->data.emit.var_a];
                }
                if (emit->data.emit.var_b >= 0 && emit->data.emit.var_b < 3) {
                    emit_b = var_bindings[emit->data.emit.var_b];
                }
                
                /* Add the derived fact if variables are bound */
                if (emit_a != -1 && emit_b != -1) {
                    if (!factdb_has_fact(&engine->facts, emit->data.emit.relation, emit_a, emit_b)) {
                        factdb_add_fact(&engine->facts, emit->data.emit.relation, emit_a, emit_b);
                        new_facts_added = true;
                        
                        if (engine->debug) {
                            const char *name_a = atom_table_name(&engine->atoms, emit_a);
                            const char *name_b = atom_table_name(&engine->atoms, emit_b);
                            printf("  Derived: %s(%s, %s)\n", emit->data.emit.relation,
                                   name_a ? name_a : "?", name_b ? name_b : "?");
                        }
                    }
                }
            }
            
            scan_iter = scan_iter->next;
        }
        
        query_result_free(scan_results);
    }
    
    return new_facts_added;
}

static bool engine_solve(ExecutionEngine *engine, const ASTNode *program) {
    /* Collect all rules */
    ASTNode *stmt = program->data.program.statements;
    ASTNode **rules = NULL;
    int rule_count = 0;
    
    /* Count rules first */
    while (stmt) {
        if (stmt->type == AST_RULE) {
            rule_count++;
        }
        stmt = stmt->next;
    }
    
    if (rule_count == 0) return true;  /* No rules to evaluate */
    
    /* Allocate rule array */
    rules = malloc(rule_count * sizeof(ASTNode*));
    if (!rules) {
        engine_error(engine, "Out of memory");
        return false;
    }
    
    /* Collect rule pointers */
    stmt = program->data.program.statements;
    int i = 0;
    while (stmt) {
        if (stmt->type == AST_RULE) {
            rules[i++] = stmt;
        }
        stmt = stmt->next;
    }
    
    /* Fixpoint iteration */
    bool changed = true;
    int iteration = 0;
    
    if (engine->debug) {
        printf("Starting fixpoint computation...\n");
    }
    
    while (changed && iteration < 100) {  /* Prevent infinite loops */
        changed = false;
        iteration++;
        
        if (engine->debug) {
            printf("\nIteration %d:\n", iteration);
        }
        
        /* Apply all rules */
        for (i = 0; i < rule_count; i++) {
            if (engine_evaluate_rule(engine, rules[i])) {
                changed = true;
            }
        }
        
        if (engine->debug) {
            printf("Facts after iteration %d: %d\n", iteration, factdb_count(&engine->facts));
        }
    }
    
    if (engine->debug) {
        printf("Fixpoint reached after %d iterations.\n", iteration);
    }
    
    free(rules);
    return true;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Statement Execution
 * ───────────────────────────────────────────────────────────────────────── */

bool engine_execute_statement(ExecutionEngine *engine, const ASTNode *stmt) {
    if (!stmt) return false;
    
    switch (stmt->type) {
        case AST_REL_DECL:
            /* Relation declarations don't need runtime processing */
            return true;
            
        case AST_FACT:
            /* Add fact to database */
            return factdb_add_fact(&engine->facts, 
                                   stmt->data.fact.relation,
                                   stmt->data.fact.a,
                                   stmt->data.fact.b);
            
        case AST_RULE:
            /* Rules are processed during SOLVE */
            return true;
            
        case AST_SOLVE:
            /* Compute fixpoint */
            return engine_solve(engine, stmt->next ? stmt : NULL);
            
        case AST_QUERY:
            /* Queries are handled separately by engine_query */
            return true;
            
        default:
            engine_error(engine, "Unknown statement type");
            return false;
    }
}

bool engine_execute_program(ExecutionEngine *engine, const ASTNode *program) {
    if (!program || program->type != AST_PROGRAM) {
        engine_error(engine, "Invalid program node");
        return false;
    }
    
    /* First pass: Add all facts and copy atom table */
    ASTNode *stmt = program->data.program.statements;
    while (stmt) {
        if (stmt->type == AST_FACT) {
            /* Copy atoms from fact to engine's atom table */
            if (stmt->data.fact.atom_a) {
                atom_table_intern(&engine->atoms, stmt->data.fact.atom_a);
            }
            if (stmt->data.fact.atom_b) {
                atom_table_intern(&engine->atoms, stmt->data.fact.atom_b);
            }
            
            if (!engine_execute_statement(engine, stmt)) {
                return false;
            }
        }
        stmt = stmt->next;
    }
    
    /* Second pass: Process SOLVE (which handles rules) */
    stmt = program->data.program.statements;
    while (stmt) {
        if (stmt->type == AST_SOLVE) {
            if (!engine_solve(engine, program)) {
                return false;
            }
            break;  /* Only process first SOLVE */
        }
        stmt = stmt->next;
    }
    
    return true;
}

QueryResult* engine_query(ExecutionEngine *engine, const ASTNode *query) {
    if (!query || query->type != AST_QUERY) {
        engine_error(engine, "Invalid query node");
        return NULL;
    }
    
    /* Copy atoms from query to engine's atom table */
    int arg_a = query->data.query.arg_a;
    int arg_b = query->data.query.arg_b;
    
    if (query->data.query.atom_a) {
        arg_a = atom_table_intern(&engine->atoms, query->data.query.atom_a);
    }
    if (query->data.query.atom_b) {
        arg_b = atom_table_intern(&engine->atoms, query->data.query.atom_b);
    }
    
    return factdb_query(&engine->facts, query->data.query.relation, arg_a, arg_b);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Convenience Functions
 * ───────────────────────────────────────────────────────────────────────── */

ExecutionEngine* execute_string(const char *source, char *error_buf, size_t error_buf_size) {
    if (!source) return NULL;
    
    /* Parse the source */
    ASTNode *ast = parse_string(source, error_buf, error_buf_size);
    if (!ast) return NULL;
    
    /* Create engine and execute */
    ExecutionEngine *engine = malloc(sizeof(ExecutionEngine));
    if (!engine) {
        ast_free_tree(ast);
        if (error_buf) snprintf(error_buf, error_buf_size, "Out of memory");
        return NULL;
    }
    
    engine_init(engine);
    
    if (!engine_execute_program(engine, ast)) {
        if (error_buf) {
            snprintf(error_buf, error_buf_size, "%s", engine_get_error(engine));
        }
        engine_cleanup(engine);
        free(engine);
        engine = NULL;
    }
    
    ast_free_tree(ast);
    return engine;
}

ExecutionEngine* execute_file(const char *filename, char *error_buf, size_t error_buf_size) {
    if (!filename) return NULL;
    
    /* Parse the file */
    ASTNode *ast = parse_file(filename, error_buf, error_buf_size);
    if (!ast) return NULL;
    
    /* Create engine and execute */
    ExecutionEngine *engine = malloc(sizeof(ExecutionEngine));
    if (!engine) {
        ast_free_tree(ast);
        if (error_buf) snprintf(error_buf, error_buf_size, "Out of memory");
        return NULL;
    }
    
    engine_init(engine);
    
    if (!engine_execute_program(engine, ast)) {
        if (error_buf) {
            snprintf(error_buf, error_buf_size, "%s", engine_get_error(engine));
        }
        engine_cleanup(engine);
        free(engine);
        engine = NULL;
    }
    
    ast_free_tree(ast);
    return engine;
}