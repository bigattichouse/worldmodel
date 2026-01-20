/* ═══════════════════════════════════════════════════════════════════════════
 * bytelog.l - Lexer specification for ByteLog
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * ByteLog: A minimal logic notation for LLMs
 * Compiles binary relation logic to WebAssembly
 *
 * Usage: flex bytelog.l && bison -d bytelog.y && gcc lex.yy.c bytelog.tab.c
 * ═══════════════════════════════════════════════════════════════════════════
 */

%{
#include "bytelog.tab.h"
#include <stdlib.h>
#include <string.h>

int line_num = 1;
%}

%option noyywrap
%option case-insensitive

%%

 /* ─────────────────────────────────────────────────────────────────────────
  * Whitespace and Comments
  * ───────────────────────────────────────────────────────────────────────── */

[ \t]+                  { /* skip whitespace */ }
\n                      { line_num++; }
;[^\n]*                 { /* skip ; comments */ }
\/\/[^\n]*              { /* skip // comments */ }

 /* ─────────────────────────────────────────────────────────────────────────
  * Keywords
  * ───────────────────────────────────────────────────────────────────────── */

"REL"                   { return REL; }
"FACT"                  { return FACT; }
"RULE"                  { return RULE; }
"SCAN"                  { return SCAN; }
"JOIN"                  { return JOIN; }
"EMIT"                  { return EMIT; }
"MATCH"                 { return MATCH; }
"SOLVE"                 { return SOLVE; }
"QUERY"                 { return QUERY; }
"CALC"                  { return CALC; }
"INPUT"                 { return INPUT; }
"LET"                   { return LET; }
"RESULT"                { return RESULT; }
"IF"                    { return IF; }
"THEN"                  { return THEN; }
"ELSE"                  { return ELSE; }
"END"                   { return END; }
"WHERE"                 { return WHERE; }
"FOR"                   { return FOR; }
"WHILE"                 { return WHILE; }
"IN"                    { return IN; }
"RANGE"                 { return RANGE; }
"LENGTH"                { return LENGTH; }
"CHAR_AT"               { return CHAR_AT; }
"BREAK"                 { return BREAK; }
"CONTINUE"              { return CONTINUE; }
"MOD"                   { return MOD; }
"POW"                   { return POW; }
"ABS"                   { return ABS; }
"MIN"                   { return MIN; }
"MAX"                   { return MAX; }
"SQRT"                  { return SQRT; }
"SIN"                   { return SIN; }
"COS"                   { return COS; }
"TAN"                   { return TAN; }
"ASIN"                  { return ASIN; }
"ACOS"                  { return ACOS; }
"ATAN"                  { return ATAN; }
"LOG"                   { return LOG; }
"LOG10"                 { return LOG10; }
"EXP"                   { return EXP; }
"CEIL"                  { return CEIL; }
"FLOOR"                 { return FLOOR; }

 /* ─────────────────────────────────────────────────────────────────────────
  * Symbols
  * ───────────────────────────────────────────────────────────────────────── */

"<="                    { return LE; }
">="                    { return GE; }
"=="                    { return EQ; }
"!="                    { return NE; }
"<>"                    { return NE; }
":"                     { return COLON; }
","                     { return COMMA; }
"?"                     { return WILDCARD; }
"+"                     { return PLUS; }
"-"                     { return MINUS; }
"*"                     { return STAR; }
"/"                     { return SLASH; }
"%"                     { return MOD; }
"("                     { return LPAREN; }
")"                     { return RPAREN; }
"="                     { return ASSIGN; }
">"                     { return GT; }
"<"                     { return LT; }

 /* ─────────────────────────────────────────────────────────────────────────
  * Variables: $0, $1, $2, ...
  * ───────────────────────────────────────────────────────────────────────── */

\$[0-9]+                { 
                          yylval.ival = atoi(yytext + 1);  /* skip $ */
                          return VAR; 
                        }

 /* ─────────────────────────────────────────────────────────────────────────
  * Literals (order matters: FLOAT before INTEGER, STRING before IDENT)
  * ───────────────────────────────────────────────────────────────────────── */

\"[^\"]*\"              { 
                          yylval.sval = strdup(yytext);
                          return STRING; 
                        }

-?[0-9]+\.[0-9]+        { 
                          yylval.fval = atof(yytext); 
                          return FLOAT; 
                        }

-?[0-9]+                { 
                          yylval.ival = atoi(yytext); 
                          return INTEGER; 
                        }

 /* ─────────────────────────────────────────────────────────────────────────
  * Identifiers (relation names)
  * ───────────────────────────────────────────────────────────────────────── */

[a-zA-Z_][a-zA-Z0-9_]*  { 
                          yylval.sval = strdup(yytext); 
                          return IDENT; 
                        }

 /* ─────────────────────────────────────────────────────────────────────────
  * Error handling
  * ───────────────────────────────────────────────────────────────────────── */

.                       { 
                          fprintf(stderr, "Line %d: unexpected character '%s'\n", 
                                  line_num, yytext);
                          return yytext[0]; 
                        }

%%
