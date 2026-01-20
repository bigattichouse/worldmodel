/* ═══════════════════════════════════════════════════════════════════════════
 * parser.h - ByteLog Parser Interface
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Recursive descent parser for ByteLog language.
 * Converts token stream into Abstract Syntax Tree.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_PARSER_H
#define BYTELOG_PARSER_H

#include "lexer.h"
#include "ast.h"
#include "atoms.h"
#include <stdbool.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Parser State
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    Lexer lexer;                /* Lexer state */
    Token current_token;        /* Current token */
    Token lookahead_token;      /* Next token (for LL(2) parsing) */
    bool has_lookahead;         /* Whether lookahead is valid */
    char error[512];           /* Error message buffer */
    int error_count;           /* Number of errors encountered */
    bool panic_mode;           /* Error recovery mode */
    AtomTable atoms;           /* Atom table for string-to-ID mapping */
} Parser;

/* ─────────────────────────────────────────────────────────────────────────
 * Parser Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Initialize parser with source text */
void parser_init(Parser *parser, const char *source);

/* Parse complete program and return AST */
ASTNode* parser_parse_program(Parser *parser);

/* Check if parser encountered errors */
bool parser_has_errors(Parser *parser);

/* Get error count */
int parser_get_error_count(Parser *parser);

/* Get last error message */
const char* parser_get_error(Parser *parser);

/* Free parser resources */
void parser_cleanup(Parser *parser);

/* ─────────────────────────────────────────────────────────────────────────
 * Error Handling
 * ───────────────────────────────────────────────────────────────────────── */

/* Parser error types */
typedef enum {
    PARSE_ERROR_UNEXPECTED_TOKEN,
    PARSE_ERROR_MISSING_TOKEN,
    PARSE_ERROR_INVALID_SYNTAX,
    PARSE_ERROR_LEXICAL_ERROR,
    PARSE_ERROR_OUT_OF_MEMORY
} ParseErrorType;

/* Report parser error */
void parser_error(Parser *parser, ParseErrorType type, const char *message);

/* Report error with token context */
void parser_error_at_token(Parser *parser, Token *token, const char *message);

/* ─────────────────────────────────────────────────────────────────────────
 * Convenience Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Parse ByteLog source text directly */
ASTNode* parse_string(const char *source, char *error_buf, size_t error_buf_size);

/* Parse ByteLog file */
ASTNode* parse_file(const char *filename, char *error_buf, size_t error_buf_size);

#endif /* BYTELOG_PARSER_H */