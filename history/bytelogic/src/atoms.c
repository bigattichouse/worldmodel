/* ═══════════════════════════════════════════════════════════════════════════
 * atoms.c - ByteLog Atom Table Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Hash table-based string interning for readable atom names.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "atoms.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <limits.h>

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

static unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    int c;
    
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    
    return hash % ATOM_TABLE_SIZE;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Atom Table Implementation
 * ───────────────────────────────────────────────────────────────────────── */

void atom_table_init(AtomTable *table) {
    assert(table);
    
    for (int i = 0; i < ATOM_TABLE_SIZE; i++) {
        table->buckets[i] = NULL;
    }
    
    table->next_id = 0;  /* Start from 0 */
    table->count = 0;
}

void atom_table_free(AtomTable *table) {
    if (!table) return;
    
    for (int i = 0; i < ATOM_TABLE_SIZE; i++) {
        AtomEntry *entry = table->buckets[i];
        while (entry) {
            AtomEntry *next = entry->next;
            free(entry->name);
            free(entry);
            entry = next;
        }
        table->buckets[i] = NULL;
    }
    
    table->next_id = 0;
    table->count = 0;
}

int atom_table_lookup(AtomTable *table, const char *name) {
    assert(table);
    assert(name);
    
    unsigned int bucket = hash_string(name);
    AtomEntry *entry = table->buckets[bucket];
    
    while (entry) {
        if (strcmp(entry->name, name) == 0) {
            return entry->id;
        }
        entry = entry->next;
    }
    
    return -1;  /* Not found */
}

int atom_table_intern(AtomTable *table, const char *name) {
    assert(table);
    assert(name);
    
    /* Check if already exists */
    int existing_id = atom_table_lookup(table, name);
    if (existing_id != -1) {
        return existing_id;
    }
    
    /* Create new entry */
    AtomEntry *entry = malloc(sizeof(AtomEntry));
    if (!entry) return -1;
    
    entry->name = strdup(name);
    if (!entry->name) {
        free(entry);
        return -1;
    }
    
    entry->id = table->next_id++;
    table->count++;
    
    /* Insert into hash table */
    unsigned int bucket = hash_string(name);
    entry->next = table->buckets[bucket];
    table->buckets[bucket] = entry;
    
    return entry->id;
}

const char* atom_table_name(const AtomTable *table, int id) {
    assert(table);
    
    /* Linear search through all buckets (could be optimized with reverse map) */
    for (int i = 0; i < ATOM_TABLE_SIZE; i++) {
        AtomEntry *entry = table->buckets[i];
        while (entry) {
            if (entry->id == id) {
                return entry->name;
            }
            entry = entry->next;
        }
    }
    
    return NULL;  /* Not found */
}

/* ─────────────────────────────────────────────────────────────────────────
 * Validation Functions
 * ───────────────────────────────────────────────────────────────────────── */

bool is_valid_atom_name(const char *name) {
    if (!name || !*name) return false;
    
    /* First character must be letter or underscore */
    if (!isalpha(*name) && *name != '_') {
        return false;
    }
    
    /* Subsequent characters must be alphanumeric or underscore */
    for (const char *p = name + 1; *p; p++) {
        if (!isalnum(*p) && *p != '_') {
            return false;
        }
    }
    
    return true;
}

bool is_integer_literal(const char *str) {
    if (!str || !*str) return false;
    
    const char *p = str;
    
    /* Optional minus sign */
    if (*p == '-') p++;
    
    /* Must have at least one digit */
    if (!*p || !isdigit(*p)) return false;
    
    /* All remaining characters must be digits */
    while (*p) {
        if (!isdigit(*p)) return false;
        p++;
    }
    
    return true;
}

int parse_integer(const char *str, bool *success) {
    assert(success);
    
    *success = false;
    
    if (!is_integer_literal(str)) {
        return 0;
    }
    
    char *endptr;
    long value = strtol(str, &endptr, 10);
    
    /* Check for conversion errors */
    if (*endptr != '\0') {
        return 0;
    }
    
    /* Check for overflow */
    if (value > INT_MAX || value < INT_MIN) {
        return 0;
    }
    
    *success = true;
    return (int)value;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Utility Functions
 * ───────────────────────────────────────────────────────────────────────── */

void atom_table_print(const AtomTable *table) {
    assert(table);
    
    printf("Atom Table (%d atoms):\n", table->count);
    printf("─────────────────────────────────────────\n");
    
    if (table->count == 0) {
        printf("(empty)\n");
        return;
    }
    
    /* Collect all entries and sort by ID for readable output */
    AtomEntry **entries = malloc(table->count * sizeof(AtomEntry*));
    int count = 0;
    
    for (int i = 0; i < ATOM_TABLE_SIZE; i++) {
        AtomEntry *entry = table->buckets[i];
        while (entry) {
            entries[count++] = entry;
            entry = entry->next;
        }
    }
    
    /* Simple selection sort by ID */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (entries[i]->id > entries[j]->id) {
                AtomEntry *temp = entries[i];
                entries[i] = entries[j];
                entries[j] = temp;
            }
        }
    }
    
    /* Print sorted entries */
    for (int i = 0; i < count; i++) {
        printf("%3d: %s\n", entries[i]->id, entries[i]->name);
    }
    
    free(entries);
}

void atom_table_stats(const AtomTable *table, int *count, int *next_id) {
    assert(table);
    
    if (count) *count = table->count;
    if (next_id) *next_id = table->next_id;
}

void atom_table_reset(AtomTable *table) {
    assert(table);
    
    atom_table_free(table);
    atom_table_init(table);
}