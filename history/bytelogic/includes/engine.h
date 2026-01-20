/* ═══════════════════════════════════════════════════════════════════════════
 * engine.h - ByteLog Execution Engine Interface
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Datalog execution engine with fixpoint computation.
 * Evaluates ByteLog programs and answers queries.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_ENGINE_H
#define BYTELOG_ENGINE_H

#include "ast.h"
#include "atoms.h"
#include <stdbool.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Fact Database Structure
 * ───────────────────────────────────────────────────────────────────────── */

#define FACT_DATABASE_SIZE 1024

typedef struct Fact {
    char *relation;             /* Relation name */
    int arg_a;                  /* First argument */
    int arg_b;                  /* Second argument */
    struct Fact *next;          /* Hash collision chain */
} Fact;

typedef struct FactDatabase {
    Fact *buckets[FACT_DATABASE_SIZE];
    int count;                  /* Number of facts */
    int capacity;               /* Total capacity */
} FactDatabase;

/* ─────────────────────────────────────────────────────────────────────────
 * Query Result Structure
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct QueryResult {
    int arg_a;                  /* Bound argument A (-1 if was wildcard) */
    int arg_b;                  /* Bound argument B (-1 if was wildcard) */
    struct QueryResult *next;   /* Next result */
} QueryResult;

/* ─────────────────────────────────────────────────────────────────────────
 * Execution Engine Structure
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct ExecutionEngine {
    FactDatabase facts;         /* Fact database */
    AtomTable atoms;            /* Atom table for name resolution */
    char error[512];           /* Error message buffer */
    int error_count;           /* Number of errors encountered */
    bool debug;                /* Debug output flag */
} ExecutionEngine;

/* ─────────────────────────────────────────────────────────────────────────
 * Engine Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize execution engine */
void engine_init(ExecutionEngine *engine);

/* Free execution engine resources */
void engine_cleanup(ExecutionEngine *engine);

/* Execute a complete ByteLog program */
bool engine_execute_program(ExecutionEngine *engine, const ASTNode *program);

/* Execute a single statement */
bool engine_execute_statement(ExecutionEngine *engine, const ASTNode *stmt);

/* Answer a query and return results */
QueryResult* engine_query(ExecutionEngine *engine, const ASTNode *query);

/* Check if engine encountered errors */
bool engine_has_errors(ExecutionEngine *engine);

/* Get error count */
int engine_get_error_count(ExecutionEngine *engine);

/* Get last error message */
const char* engine_get_error(ExecutionEngine *engine);

/* Set debug mode */
void engine_set_debug(ExecutionEngine *engine, bool debug);

/* ─────────────────────────────────────────────────────────────────────────
 * Fact Database Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize fact database */
void factdb_init(FactDatabase *db);

/* Free fact database */
void factdb_cleanup(FactDatabase *db);

/* Add fact to database */
bool factdb_add_fact(FactDatabase *db, const char *relation, int arg_a, int arg_b);

/* Check if fact exists in database */
bool factdb_has_fact(FactDatabase *db, const char *relation, int arg_a, int arg_b);

/* Query facts matching pattern (wildcards = -1) */
QueryResult* factdb_query(FactDatabase *db, const char *relation, int arg_a, int arg_b);

/* Get all facts for a relation */
QueryResult* factdb_get_all(FactDatabase *db, const char *relation);

/* Print all facts (for debugging) */
void factdb_print(const FactDatabase *db, const AtomTable *atoms);

/* Get fact count */
int factdb_count(const FactDatabase *db);

/* ─────────────────────────────────────────────────────────────────────────
 * Query Result Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Free query result list */
void query_result_free(QueryResult *results);

/* Count results in list */
int query_result_count(QueryResult *results);

/* Print query results */
void query_result_print(QueryResult *results, const char *relation, const AtomTable *atoms);

/* ─────────────────────────────────────────────────────────────────────────
 * Convenience Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Execute ByteLog file and return engine */
ExecutionEngine* execute_file(const char *filename, char *error_buf, size_t error_buf_size);

/* Execute ByteLog source string and return engine */
ExecutionEngine* execute_string(const char *source, char *error_buf, size_t error_buf_size);

#endif /* BYTELOG_ENGINE_H */