/* ═══════════════════════════════════════════════════════════════════════════
 * lexer.c - ByteLog Lexer Implementation  
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Portable C implementation of the ByteLog lexer.
 * Based on the flex specification in bytelog.l
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "lexer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

/* For strcasecmp and strdup portability */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Provide strcasecmp for systems that don't have it */
#if defined(_MSC_VER) || defined(__MINGW32__)
#define strcasecmp _stricmp
#endif

/* Provide strdup for systems that don't have it */
#if !defined(_GNU_SOURCE) && !defined(__GLIBC__)
static char* strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *result = malloc(len);
    if (result) {
        memcpy(result, s, len);
    }
    return result;
}
#endif

/* Provide strcasecmp for C99 compatibility */
#if !defined(_GNU_SOURCE) && !defined(__GLIBC__) && !defined(_MSC_VER)
static int strcasecmp(const char *s1, const char *s2) {
    while (*s1 && *s2) {
        int c1 = tolower(*s1);
        int c2 = tolower(*s2);
        if (c1 != c2) return c1 - c2;
        s1++;
        s2++;
    }
    return tolower(*s1) - tolower(*s2);
}
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Keyword Table
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    const char *text;
    TokenType type;
} Keyword;

static const Keyword KEYWORDS[] = {
    {"REL", TOK_REL},
    {"FACT", TOK_FACT}, 
    {"RULE", TOK_RULE},
    {"SCAN", TOK_SCAN},
    {"JOIN", TOK_JOIN},
    {"EMIT", TOK_EMIT},
    {"MATCH", TOK_MATCH},
    {"SOLVE", TOK_SOLVE},
    {"QUERY", TOK_QUERY},
    {NULL, TOK_ERROR}  /* Sentinel */
};

/* ─────────────────────────────────────────────────────────────────────────
 * Utility Functions
 * ───────────────────────────────────────────────────────────────────────── */

static char current_char(Lexer *lex) {
    if (lex->pos >= lex->length) return '\0';
    return lex->source[lex->pos];
}

static char peek_char(Lexer *lex, int offset) {
    size_t pos = lex->pos + offset;
    if (pos >= lex->length) return '\0';
    return lex->source[pos];
}

static void advance_char(Lexer *lex) {
    if (lex->pos < lex->length) {
        if (lex->source[lex->pos] == '\n') {
            lex->line++;
            lex->column = 1;
        } else {
            lex->column++;
        }
        lex->pos++;
    }
}

static void skip_whitespace(Lexer *lex) {
    while (current_char(lex) && strchr(" \t\r", current_char(lex))) {
        advance_char(lex);
    }
}

static void skip_comment(Lexer *lex) {
    char ch = current_char(lex);
    
    /* Semicolon comment */
    if (ch == ';') {
        while (current_char(lex) && current_char(lex) != '\n') {
            advance_char(lex);
        }
    }
    /* C++ style comment */
    else if (ch == '/' && peek_char(lex, 1) == '/') {
        advance_char(lex); /* Skip first / */
        advance_char(lex); /* Skip second / */
        while (current_char(lex) && current_char(lex) != '\n') {
            advance_char(lex);
        }
    }
}

static bool is_alpha_or_underscore(char ch) {
    return isalpha(ch) || ch == '_';
}

static bool is_identifier_char(char ch) {
    return isalnum(ch) || ch == '_';
}

static TokenType lookup_keyword(const char *text) {
    for (int i = 0; KEYWORDS[i].text; i++) {
        if (strcasecmp(text, KEYWORDS[i].text) == 0) {
            return KEYWORDS[i].type;
        }
    }
    return TOK_IDENTIFIER;
}

static char* copy_string(const char *start, size_t length) {
    char *result = malloc(length + 1);
    if (!result) return NULL;
    
    strncpy(result, start, length);
    result[length] = '\0';
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Token Functions
 * ───────────────────────────────────────────────────────────────────────── */

Token token_make_string(TokenType type, const char *value, int line, int column) {
    Token token;
    token.type = type;
    token.value = value ? strdup(value) : NULL;
    token.int_value = 0;
    token.line = line;
    token.column = column;
    return token;
}

Token token_make_int(TokenType type, int value, int line, int column) {
    Token token;
    token.type = type;
    token.value = NULL;
    token.int_value = value;
    token.line = line;
    token.column = column;
    return token;
}

Token token_make_simple(TokenType type, int line, int column) {
    Token token;
    token.type = type;
    token.value = NULL;
    token.int_value = 0;
    token.line = line;
    token.column = column;
    return token;
}

void token_free(Token *token) {
    if (token && token->value) {
        free(token->value);
        token->value = NULL;
    }
}

const char* token_type_name(TokenType type) {
    switch (type) {
        case TOK_REL: return "REL";
        case TOK_FACT: return "FACT";
        case TOK_RULE: return "RULE";
        case TOK_SCAN: return "SCAN";
        case TOK_JOIN: return "JOIN";
        case TOK_EMIT: return "EMIT";
        case TOK_MATCH: return "MATCH";
        case TOK_SOLVE: return "SOLVE";
        case TOK_QUERY: return "QUERY";
        case TOK_COLON: return "COLON";
        case TOK_COMMA: return "COMMA";
        case TOK_WILDCARD: return "WILDCARD";
        case TOK_VARIABLE: return "VARIABLE";
        case TOK_INTEGER: return "INTEGER";
        case TOK_IDENTIFIER: return "IDENTIFIER";
        case TOK_EOF: return "EOF";
        case TOK_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void token_print(const Token *token) {
    printf("%-12s", token_type_name(token->type));
    if (token->value) {
        printf(" '%s'", token->value);
    } else if (token->type == TOK_INTEGER || token->type == TOK_VARIABLE) {
        printf(" %d", token->int_value);
    }
    printf(" @%d:%d\n", token->line, token->column);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Lexer Functions
 * ───────────────────────────────────────────────────────────────────────── */

void lexer_init(Lexer *lex, const char *source) {
    assert(lex);
    assert(source);
    
    lex->source = source;
    lex->length = strlen(source);
    lex->pos = 0;
    lex->line = 1;
    lex->column = 1;
    lex->error[0] = '\0';
}

static Token scan_variable(Lexer *lex) {
    int start_line = lex->line;
    int start_column = lex->column;
    
    assert(current_char(lex) == '$');
    advance_char(lex); /* Skip $ */
    
    const char *start = lex->source + lex->pos;
    size_t length = 0;
    
    /* Scan digits */
    while (current_char(lex) && isdigit(current_char(lex))) {
        advance_char(lex);
        length++;
    }
    
    if (length == 0) {
        snprintf(lex->error, sizeof(lex->error), 
                "Variable must have digit after $");
        return token_make_simple(TOK_ERROR, start_line, start_column);
    }
    
    /* Convert to integer */
    char *var_text = copy_string(start, length);
    int var_num = atoi(var_text);
    free(var_text);
    
    return token_make_int(TOK_VARIABLE, var_num, start_line, start_column);
}

static Token scan_number(Lexer *lex) {
    int start_line = lex->line;
    int start_column = lex->column;
    
    const char *start = lex->source + lex->pos;
    size_t length = 0;
    
    /* Handle optional minus sign */
    if (current_char(lex) == '-') {
        advance_char(lex);
        length++;
    }
    
    /* Scan digits */
    while (current_char(lex) && isdigit(current_char(lex))) {
        advance_char(lex);
        length++;
    }
    
    /* Convert to integer */
    char *num_text = copy_string(start, length);
    int number = atoi(num_text);
    free(num_text);
    
    return token_make_int(TOK_INTEGER, number, start_line, start_column);
}

static Token scan_identifier(Lexer *lex) {
    int start_line = lex->line;
    int start_column = lex->column;
    
    const char *start = lex->source + lex->pos;
    size_t length = 0;
    
    /* First character must be alpha or underscore */
    assert(is_alpha_or_underscore(current_char(lex)));
    advance_char(lex);
    length++;
    
    /* Subsequent characters can be alphanumeric or underscore */
    while (current_char(lex) && is_identifier_char(current_char(lex))) {
        advance_char(lex);
        length++;
    }
    
    char *text = copy_string(start, length);
    TokenType type = lookup_keyword(text);
    
    if (type == TOK_IDENTIFIER) {
        return token_make_string(TOK_IDENTIFIER, text, start_line, start_column);
    } else {
        Token token = token_make_simple(type, start_line, start_column);
        free(text);
        return token;
    }
}

Token lexer_next_token(Lexer *lex) {
    assert(lex);
    
    while (true) {
        /* Skip whitespace */
        skip_whitespace(lex);
        
        char ch = current_char(lex);
        int line = lex->line;
        int column = lex->column;
        
        /* End of input */
        if (ch == '\0') {
            return token_make_simple(TOK_EOF, line, column);
        }
        
        /* Skip newlines */
        if (ch == '\n') {
            advance_char(lex);
            continue;
        }
        
        /* Comments */
        if (ch == ';' || (ch == '/' && peek_char(lex, 1) == '/')) {
            skip_comment(lex);
            continue;
        }
        
        /* Single character tokens */
        if (ch == ':') {
            advance_char(lex);
            return token_make_simple(TOK_COLON, line, column);
        }
        if (ch == ',') {
            advance_char(lex);
            return token_make_simple(TOK_COMMA, line, column);
        }
        if (ch == '?') {
            advance_char(lex);
            return token_make_simple(TOK_WILDCARD, line, column);
        }
        
        /* Variables: $0, $1, $2, ... */
        if (ch == '$') {
            return scan_variable(lex);
        }
        
        /* Numbers: -123, 456 */
        if (isdigit(ch) || (ch == '-' && isdigit(peek_char(lex, 1)))) {
            return scan_number(lex);
        }
        
        /* Identifiers and keywords */
        if (is_alpha_or_underscore(ch)) {
            return scan_identifier(lex);
        }
        
        /* Unknown character */
        snprintf(lex->error, sizeof(lex->error), 
                "Unexpected character '%c'", ch);
        advance_char(lex);
        return token_make_simple(TOK_ERROR, line, column);
    }
}

Token lexer_peek_token(Lexer *lex) {
    /* Save lexer state */
    size_t saved_pos = lex->pos;
    int saved_line = lex->line;
    int saved_column = lex->column;
    
    /* Get next token */
    Token token = lexer_next_token(lex);
    
    /* Restore lexer state */
    lex->pos = saved_pos;
    lex->line = saved_line;
    lex->column = saved_column;
    
    return token;
}

bool lexer_at_eof(Lexer *lex) {
    return lex->pos >= lex->length;
}

int lexer_get_line(Lexer *lex) {
    return lex->line;
}

int lexer_get_column(Lexer *lex) {
    return lex->column;
}

const char* lexer_get_error(Lexer *lex) {
    return lex->error[0] ? lex->error : NULL;
}