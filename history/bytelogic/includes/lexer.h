/* ═══════════════════════════════════════════════════════════════════════════
 * lexer.h - ByteLog Lexer Interface
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Portable C implementation of the ByteLog lexer.
 * Converts source text into a stream of tokens for parsing.
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_LEXER_H
#define BYTELOG_LEXER_H

#include <stdio.h>
#include <stdbool.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Token Types
 * ───────────────────────────────────────────────────────────────────────── */

typedef enum {
    /* Keywords */
    TOK_REL,
    TOK_FACT,
    TOK_RULE,
    TOK_SCAN,
    TOK_JOIN,
    TOK_EMIT,
    TOK_MATCH,
    TOK_SOLVE,
    TOK_QUERY,
    
    /* Symbols */
    TOK_COLON,      /* : */
    TOK_COMMA,      /* , */
    TOK_WILDCARD,   /* ? */
    
    /* Literals */
    TOK_VARIABLE,   /* $0, $1, $2, ... */
    TOK_INTEGER,    /* 123, -456 */
    TOK_IDENTIFIER, /* relation names */
    
    /* Special */
    TOK_EOF,
    TOK_ERROR
} TokenType;

/* ─────────────────────────────────────────────────────────────────────────
 * Token Structure
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    TokenType type;
    char *value;        /* Token text (malloc'd, must be freed) */
    int int_value;      /* For TOK_INTEGER and TOK_VARIABLE */
    int line;           /* Line number (1-based) */
    int column;         /* Column number (1-based) */
} Token;

/* ─────────────────────────────────────────────────────────────────────────
 * Lexer State
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    const char *source; /* Input source text */
    size_t length;      /* Length of source */
    size_t pos;         /* Current position */
    int line;           /* Current line (1-based) */
    int column;         /* Current column (1-based) */
    char error[256];    /* Error message buffer */
} Lexer;

/* ─────────────────────────────────────────────────────────────────────────
 * Function Declarations
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize lexer with source text */
void lexer_init(Lexer *lex, const char *source);

/* Get next token (caller must free token->value) */
Token lexer_next_token(Lexer *lex);

/* Peek at next token without consuming it */
Token lexer_peek_token(Lexer *lex);

/* Check if at end of input */
bool lexer_at_eof(Lexer *lex);

/* Get current line number */
int lexer_get_line(Lexer *lex);

/* Get current column number */
int lexer_get_column(Lexer *lex);

/* Get last error message */
const char* lexer_get_error(Lexer *lex);

/* Free token memory */
void token_free(Token *token);

/* Create token with string value (copies string) */
Token token_make_string(TokenType type, const char *value, int line, int column);

/* Create token with integer value */
Token token_make_int(TokenType type, int value, int line, int column);

/* Create token with no value */
Token token_make_simple(TokenType type, int line, int column);

/* Print token for debugging */
void token_print(const Token *token);

/* Convert token type to string */
const char* token_type_name(TokenType type);

#endif /* BYTELOG_LEXER_H */