/* ═══════════════════════════════════════════════════════════════════════════
 * parser.c - ByteLog Parser Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Recursive descent parser for ByteLog language.
 * Implements the grammar from bytelog-spec.md
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "parser.h"
#include "atoms.h"
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
 * Private Parser Functions
 * ───────────────────────────────────────────────────────────────────────── */

static void advance_token(Parser *parser);
static bool expect_token(Parser *parser, TokenType type);
static void synchronize(Parser *parser);

static ASTNode* parse_statement(Parser *parser);
static ASTNode* parse_rel_decl(Parser *parser);
static ASTNode* parse_fact(Parser *parser);
static ASTNode* parse_rule(Parser *parser);
static ASTNode* parse_solve(Parser *parser);
static ASTNode* parse_query(Parser *parser);

/* Helper function to parse an argument that can be integer, atom, or wildcard */
static bool parse_argument(Parser *parser, int *value, char **atom_name);
static ASTNode* parse_body(Parser *parser);
static ASTNode* parse_operation(Parser *parser);
static ASTNode* parse_scan(Parser *parser);
static ASTNode* parse_join(Parser *parser);
static ASTNode* parse_emit(Parser *parser);

/* ─────────────────────────────────────────────────────────────────────────
 * Token Management
 * ───────────────────────────────────────────────────────────────────────── */

static void advance_token(Parser *parser) {
    /* Free current token if it has a value */
    token_free(&parser->current_token);
    
    if (parser->has_lookahead) {
        parser->current_token = parser->lookahead_token;
        parser->has_lookahead = false;
        /* Don't free lookahead - we moved it to current */
        parser->lookahead_token.value = NULL;
    } else {
        parser->current_token = lexer_next_token(&parser->lexer);
    }
    
    /* Handle lexical errors */
    if (parser->current_token.type == TOK_ERROR) {
        parser_error_at_token(parser, &parser->current_token, 
                             lexer_get_error(&parser->lexer));
    }
}

static Token peek_token(Parser *parser) {
    if (!parser->has_lookahead) {
        parser->lookahead_token = lexer_next_token(&parser->lexer);
        parser->has_lookahead = true;
    }
    return parser->lookahead_token;
}

static bool check_token(Parser *parser, TokenType type) {
    return parser->current_token.type == type;
}

/* match_token removed - not used in current implementation */

static bool expect_token(Parser *parser, TokenType type) {
    if (check_token(parser, type)) {
        advance_token(parser);
        return true;
    }
    
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), 
             "Expected %s but found %s", 
             token_type_name(type), 
             token_type_name(parser->current_token.type));
    parser_error_at_token(parser, &parser->current_token, error_msg);
    
    return false;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Error Handling
 * ───────────────────────────────────────────────────────────────────────── */

void parser_error(Parser *parser, ParseErrorType type, const char *message) {
    (void)type;  /* Unused parameter - reserved for future use */
    parser->error_count++;
    snprintf(parser->error, sizeof(parser->error), "%s", message);
    parser->panic_mode = true;
}

void parser_error_at_token(Parser *parser, Token *token, const char *message) {
    parser->error_count++;
    snprintf(parser->error, sizeof(parser->error), 
             "Line %d:%d: %s", token->line, token->column, message);
    parser->panic_mode = true;
}

static void synchronize(Parser *parser) {
    parser->panic_mode = false;
    
    while (parser->current_token.type != TOK_EOF) {
        /* Sync on statement keywords */
        if (parser->current_token.type == TOK_REL ||
            parser->current_token.type == TOK_FACT ||
            parser->current_token.type == TOK_RULE ||
            parser->current_token.type == TOK_SOLVE ||
            parser->current_token.type == TOK_QUERY) {
            return;
        }
        
        advance_token(parser);
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Parser Initialization
 * ───────────────────────────────────────────────────────────────────────── */

void parser_init(Parser *parser, const char *source) {
    assert(parser);
    assert(source);
    
    lexer_init(&parser->lexer, source);
    parser->has_lookahead = false;
    parser->error[0] = '\0';
    parser->error_count = 0;
    parser->panic_mode = false;
    
    /* Initialize atom table */
    atom_table_init(&parser->atoms);
    
    /* Initialize with first token */
    parser->current_token = lexer_next_token(&parser->lexer);
}

void parser_cleanup(Parser *parser) {
    if (parser) {
        token_free(&parser->current_token);
        if (parser->has_lookahead) {
            token_free(&parser->lookahead_token);
        }
        atom_table_free(&parser->atoms);
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Helper Functions
 * ───────────────────────────────────────────────────────────────────────── */

static bool parse_argument(Parser *parser, int *value, char **atom_name) {
    *value = 0;
    *atom_name = NULL;
    
    if (parser->current_token.type == TOK_INTEGER) {
        *value = parser->current_token.int_value;
        advance_token(parser);
        return true;
    } else if (parser->current_token.type == TOK_WILDCARD) {
        *value = -1;  /* Wildcard */
        advance_token(parser);
        return true;
    } else if (parser->current_token.type == TOK_IDENTIFIER) {
        /* Atom - intern the string to get ID */
        *atom_name = strdup(parser->current_token.value);
        *value = atom_table_intern(&parser->atoms, parser->current_token.value);
        if (*value == -1) {
            parser_error_at_token(parser, &parser->current_token, 
                                 "Failed to intern atom");
            free(*atom_name);
            *atom_name = NULL;
            return false;
        }
        advance_token(parser);
        return true;
    } else {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected integer, atom, or '?' in argument position");
        return false;
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Grammar Implementation
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode* parser_parse_program(Parser *parser) {
    ASTNode *statements = NULL;
    
    while (parser->current_token.type != TOK_EOF && !parser->panic_mode) {
        ASTNode *stmt = parse_statement(parser);
        if (stmt) {
            statements = ast_append(statements, stmt);
        }
        
        /* Skip to next statement on error */
        if (parser->panic_mode) {
            synchronize(parser);
        }
    }
    
    return ast_make_program(statements);
}

static ASTNode* parse_statement(Parser *parser) {
    switch (parser->current_token.type) {
        case TOK_REL:
            return parse_rel_decl(parser);
        case TOK_FACT:
            return parse_fact(parser);
        case TOK_RULE:
            return parse_rule(parser);
        case TOK_SOLVE:
            return parse_solve(parser);
        case TOK_QUERY:
            return parse_query(parser);
        default:
            parser_error_at_token(parser, &parser->current_token, 
                                 "Expected statement (REL, FACT, RULE, SOLVE, or QUERY)");
            return NULL;
    }
}

static ASTNode* parse_rel_decl(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_REL)) return NULL;
    
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after REL");
        return NULL;
    }
    
    char *name = strdup(parser->current_token.value);
    advance_token(parser);
    
    ASTNode *node = ast_make_rel_decl(name, line, column);
    free(name);
    return node;
}

static ASTNode* parse_fact(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_FACT)) return NULL;
    
    /* Relation name */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after FACT");
        return NULL;
    }
    char *relation = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* First argument (integer or atom) */
    int a;
    char *atom_a;
    if (!parse_argument(parser, &a, &atom_a)) {
        free(relation);
        return NULL;
    }
    
    /* Second argument (integer or atom) */
    int b;
    char *atom_b;
    if (!parse_argument(parser, &b, &atom_b)) {
        free(relation);
        free(atom_a);
        return NULL;
    }
    
    ASTNode *node;
    if (atom_a || atom_b) {
        node = ast_make_fact_with_atoms(relation, a, b, atom_a, atom_b, line, column);
    } else {
        node = ast_make_fact(relation, a, b, line, column);
    }
    
    free(relation);
    free(atom_a);
    free(atom_b);
    return node;
}

static ASTNode* parse_rule(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_RULE)) return NULL;
    
    /* Target relation */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected target relation after RULE");
        return NULL;
    }
    char *target = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* Colon */
    if (!expect_token(parser, TOK_COLON)) {
        free(target);
        return NULL;
    }
    
    /* Body operations */
    ASTNode *body = parse_body(parser);
    if (!body) {
        free(target);
        return NULL;
    }
    
    /* Comma before emit */
    if (!expect_token(parser, TOK_COMMA)) {
        free(target);
        ast_free_tree(body);
        return NULL;
    }
    
    /* Emit operation */
    ASTNode *emit = parse_emit(parser);
    if (!emit) {
        free(target);
        ast_free_tree(body);
        return NULL;
    }
    
    ASTNode *node = ast_make_rule(target, body, emit, line, column);
    free(target);
    return node;
}

static ASTNode* parse_body(Parser *parser) {
    ASTNode *operations = NULL;
    
    /* Parse first operation */
    ASTNode *op = parse_operation(parser);
    if (!op) return NULL;
    operations = op;
    
    /* Parse additional operations separated by commas */
    while (parser->current_token.type == TOK_COMMA) {
        /* Look ahead to see if this is the comma before EMIT */
        Token lookahead = peek_token(parser);
        if (lookahead.type == TOK_EMIT) {
            break;  /* This comma is before EMIT, not between operations */
        }
        
        advance_token(parser);  /* Consume comma */
        
        op = parse_operation(parser);
        if (!op) {
            ast_free_tree(operations);
            return NULL;
        }
        operations = ast_append(operations, op);
    }
    
    return operations;
}

static ASTNode* parse_operation(Parser *parser) {
    switch (parser->current_token.type) {
        case TOK_SCAN:
            return parse_scan(parser);
        case TOK_JOIN:
            return parse_join(parser);
        default:
            parser_error_at_token(parser, &parser->current_token, 
                                 "Expected SCAN or JOIN operation");
            return NULL;
    }
}

static ASTNode* parse_scan(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_SCAN)) return NULL;
    
    /* Relation name */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after SCAN");
        return NULL;
    }
    char *relation = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* Optional MATCH clause */
    bool has_match = false;
    int match_var = -1;
    
    if (parser->current_token.type == TOK_MATCH) {
        advance_token(parser);
        has_match = true;
        
        if (parser->current_token.type != TOK_VARIABLE) {
            parser_error_at_token(parser, &parser->current_token, 
                                 "Expected variable after MATCH");
            free(relation);
            return NULL;
        }
        match_var = parser->current_token.int_value;
        advance_token(parser);
    }
    
    ASTNode *node = ast_make_scan(relation, has_match, match_var, line, column);
    free(relation);
    return node;
}

static ASTNode* parse_join(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_JOIN)) return NULL;
    
    /* Relation name */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after JOIN");
        return NULL;
    }
    char *relation = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* Match variable */
    if (parser->current_token.type != TOK_VARIABLE) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected variable after relation name");
        free(relation);
        return NULL;
    }
    int match_var = parser->current_token.int_value;
    advance_token(parser);
    
    ASTNode *node = ast_make_join(relation, match_var, line, column);
    free(relation);
    return node;
}

static ASTNode* parse_emit(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_EMIT)) return NULL;
    
    /* Relation name */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after EMIT");
        return NULL;
    }
    char *relation = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* First variable */
    if (parser->current_token.type != TOK_VARIABLE) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected first variable");
        free(relation);
        return NULL;
    }
    int var_a = parser->current_token.int_value;
    advance_token(parser);
    
    /* Second variable */
    if (parser->current_token.type != TOK_VARIABLE) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected second variable");
        free(relation);
        return NULL;
    }
    int var_b = parser->current_token.int_value;
    advance_token(parser);
    
    ASTNode *node = ast_make_emit(relation, var_a, var_b, line, column);
    free(relation);
    return node;
}

static ASTNode* parse_solve(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_SOLVE)) return NULL;
    
    return ast_make_solve(line, column);
}

static ASTNode* parse_query(Parser *parser) {
    int line = parser->current_token.line;
    int column = parser->current_token.column;
    
    if (!expect_token(parser, TOK_QUERY)) return NULL;
    
    /* Relation name */
    if (parser->current_token.type != TOK_IDENTIFIER) {
        parser_error_at_token(parser, &parser->current_token, 
                             "Expected relation name after QUERY");
        return NULL;
    }
    char *relation = strdup(parser->current_token.value);
    advance_token(parser);
    
    /* First argument (integer, atom, or wildcard) */
    int arg_a;
    char *atom_a;
    if (!parse_argument(parser, &arg_a, &atom_a)) {
        free(relation);
        return NULL;
    }
    
    /* Second argument (integer, atom, or wildcard) */
    int arg_b;
    char *atom_b;
    if (!parse_argument(parser, &arg_b, &atom_b)) {
        free(relation);
        free(atom_a);
        return NULL;
    }
    
    ASTNode *node;
    if (atom_a || atom_b) {
        node = ast_make_query_with_atoms(relation, arg_a, arg_b, atom_a, atom_b, line, column);
    } else {
        node = ast_make_query(relation, arg_a, arg_b, line, column);
    }
    
    free(relation);
    free(atom_a);
    free(atom_b);
    return node;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Parser Status Functions
 * ───────────────────────────────────────────────────────────────────────── */

bool parser_has_errors(Parser *parser) {
    return parser->error_count > 0;
}

int parser_get_error_count(Parser *parser) {
    return parser->error_count;
}

const char* parser_get_error(Parser *parser) {
    return parser->error[0] ? parser->error : NULL;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Convenience Functions
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode* parse_string(const char *source, char *error_buf, size_t error_buf_size) {
    Parser parser;
    parser_init(&parser, source);
    
    ASTNode *ast = parser_parse_program(&parser);
    
    if (parser_has_errors(&parser)) {
        if (error_buf && error_buf_size > 0) {
            strncpy(error_buf, parser_get_error(&parser), error_buf_size - 1);
            error_buf[error_buf_size - 1] = '\0';
        }
        ast_free_tree(ast);
        ast = NULL;
    }
    
    parser_cleanup(&parser);
    return ast;
}

ASTNode* parse_file(const char *filename, char *error_buf, size_t error_buf_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        if (error_buf && error_buf_size > 0) {
            snprintf(error_buf, error_buf_size, "Cannot open file '%s'", filename);
        }
        return NULL;
    }
    
    /* Read entire file into memory */
    /* Handle stdin specially since ftell() doesn't work reliably on pipes */
    long file_size = -1;
    if (strcmp(filename, "/dev/stdin") != 0) {
        fseek(file, 0, SEEK_END);
        file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
    }
    
    char *source = NULL;
    size_t bytes_read = 0;
    
    if (file_size >= 0) {
        /* Regular file - allocate exact size */
        source = malloc(file_size + 1);
        if (!source) {
            fclose(file);
            if (error_buf && error_buf_size > 0) {
                snprintf(error_buf, error_buf_size, "Out of memory");
            }
            return NULL;
        }
        bytes_read = fread(source, 1, file_size, file);
    } else {
        /* stdin or pipe - read in chunks */
        size_t capacity = 1024;
        source = malloc(capacity);
        if (!source) {
            fclose(file);
            if (error_buf && error_buf_size > 0) {
                snprintf(error_buf, error_buf_size, "Out of memory");
            }
            return NULL;
        }
        
        int ch;
        while ((ch = fgetc(file)) != EOF) {
            if (bytes_read >= capacity - 1) {
                capacity *= 2;
                char *new_source = realloc(source, capacity);
                if (!new_source) {
                    free(source);
                    fclose(file);
                    if (error_buf && error_buf_size > 0) {
                        snprintf(error_buf, error_buf_size, "Out of memory");
                    }
                    return NULL;
                }
                source = new_source;
            }
            source[bytes_read++] = ch;
        }
    }
    
    source[bytes_read] = '\0';
    fclose(file);
    
    ASTNode *ast = parse_string(source, error_buf, error_buf_size);
    
    free(source);
    return ast;
}