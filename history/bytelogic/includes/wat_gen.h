/* ═══════════════════════════════════════════════════════════════════════════
 * wat_gen.h - WebAssembly Text (WAT) Code Generator Interface
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Generates WAT code from ByteLog programs for WASM execution.
 * Compatible with Wasmtime and other WASM runtimes.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_WAT_GEN_H
#define BYTELOG_WAT_GEN_H

#include "ast.h"
#include "atoms.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* ─────────────────────────────────────────────────────────────────────────
 * WAT Generator Structure
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct WATGenerator {
    FILE *output;               /* Output file stream */
    AtomTable atoms;            /* Atom table for name resolution */
    char error[512];           /* Error message buffer */
    int error_count;           /* Number of errors encountered */
    int memory_size;           /* WebAssembly memory size in pages */
    int fact_offset;           /* Offset in memory for fact storage */
    int atom_offset;           /* Offset in memory for atom names */
    int next_func_id;          /* Next function ID */
} WATGenerator;

/* ─────────────────────────────────────────────────────────────────────────
 * WAT Generation Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize WAT generator */
void wat_gen_init(WATGenerator *gen, FILE *output);

/* Free WAT generator resources */
void wat_gen_cleanup(WATGenerator *gen);

/* Generate WAT module from ByteLog program */
bool wat_gen_program(WATGenerator *gen, const ASTNode *program);

/* Generate WAT code for a statement */
bool wat_gen_statement(WATGenerator *gen, const ASTNode *stmt);

/* Check if generator encountered errors */
bool wat_gen_has_errors(WATGenerator *gen);

/* Get error count */
int wat_gen_get_error_count(WATGenerator *gen);

/* Get last error message */
const char* wat_gen_get_error(WATGenerator *gen);

/* ─────────────────────────────────────────────────────────────────────────
 * Memory Layout Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Calculate memory requirements */
void wat_gen_calculate_memory(WATGenerator *gen, const ASTNode *program);

/* Generate memory declarations */
bool wat_gen_memory_section(WATGenerator *gen);

/* Generate data section with atom names */
bool wat_gen_data_section(WATGenerator *gen);

/* ─────────────────────────────────────────────────────────────────────────
 * Function Generation
 * ───────────────────────────────────────────────────────────────────────── */

/* Generate fact database functions */
bool wat_gen_fact_functions(WATGenerator *gen);

/* Generate rule evaluation functions */
bool wat_gen_rule_functions(WATGenerator *gen, const ASTNode *program);

/* Generate query functions */
bool wat_gen_query_functions(WATGenerator *gen, const ASTNode *program);

/* Generate main function */
bool wat_gen_main_function(WATGenerator *gen, const ASTNode *program);

/* ─────────────────────────────────────────────────────────────────────────
 * Export Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Generate exports for JavaScript interface */
bool wat_gen_exports(WATGenerator *gen);

/* ─────────────────────────────────────────────────────────────────────────
 * Utility Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Write string to output with proper escaping */
bool wat_write_string(WATGenerator *gen, const char *str);

/* Write integer to output */
bool wat_write_int(WATGenerator *gen, int value);

/* Write comment to output */
bool wat_write_comment(WATGenerator *gen, const char *comment);

/* ─────────────────────────────────────────────────────────────────────────
 * Convenience Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Generate WAT file from ByteLog source file */
bool generate_wat_file(const char *input_filename, const char *output_filename, 
                      char *error_buf, size_t error_buf_size);

/* Generate WAT from ByteLog source string */
bool generate_wat_string(const char *source, FILE *output, 
                        char *error_buf, size_t error_buf_size);

/* ─────────────────────────────────────────────────────────────────────────
 * WASM Memory Layout Constants
 * ───────────────────────────────────────────────────────────────────────── */

#define WAT_PAGE_SIZE 65536          /* WebAssembly page size */
#define WAT_INITIAL_PAGES 1          /* Initial memory pages */
#define WAT_FACT_SIZE 12             /* Size of fact structure (relation_id + 2 args) */
#define WAT_MAX_FACTS 1000           /* Maximum number of facts */
#define WAT_ATOM_NAME_SIZE 64        /* Maximum atom name length */

#endif /* BYTELOG_WAT_GEN_H */