/* ═══════════════════════════════════════════════════════════════════════════
 * atoms.h - ByteLog Atom Table Interface  
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides string-to-integer mapping for readable atom names.
 * Allows writing "FACT likes alice pizza" instead of "FACT likes 0 10".
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_ATOMS_H
#define BYTELOG_ATOMS_H

#include <stdbool.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Atom Table Structure
 * ───────────────────────────────────────────────────────────────────────── */

#define ATOM_TABLE_SIZE 1024

typedef struct AtomEntry {
    char *name;                 /* Atom name (malloc'd) */
    int id;                     /* Unique integer ID */
    struct AtomEntry *next;     /* Hash collision chain */
} AtomEntry;

typedef struct AtomTable {
    AtomEntry *buckets[ATOM_TABLE_SIZE];
    int next_id;                /* Next ID to assign */
    int count;                  /* Number of atoms */
} AtomTable;

/* ─────────────────────────────────────────────────────────────────────────
 * Atom Table Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize atom table */
void atom_table_init(AtomTable *table);

/* Free atom table and all entries */
void atom_table_free(AtomTable *table);

/* Get or create atom ID for name (assigns new ID if not exists) */
int atom_table_intern(AtomTable *table, const char *name);

/* Get atom ID for name (returns -1 if not found) */
int atom_table_lookup(AtomTable *table, const char *name);

/* Get atom name for ID (returns NULL if not found) */
const char* atom_table_name(const AtomTable *table, int id);

/* Check if string is a valid atom name (identifier) */
bool is_valid_atom_name(const char *name);

/* Check if string represents an integer literal */
bool is_integer_literal(const char *str);

/* Parse integer from string (returns value, sets success flag) */
int parse_integer(const char *str, bool *success);

/* ─────────────────────────────────────────────────────────────────────────
 * Atom Table Utilities
 * ───────────────────────────────────────────────────────────────────────── */

/* Print all atoms in table (for debugging) */
void atom_table_print(const AtomTable *table);

/* Get statistics about atom table */
void atom_table_stats(const AtomTable *table, int *count, int *next_id);

/* Reset atom table (clear all entries) */
void atom_table_reset(AtomTable *table);

#endif /* BYTELOG_ATOMS_H */