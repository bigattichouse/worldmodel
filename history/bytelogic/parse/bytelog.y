/* ═══════════════════════════════════════════════════════════════════════════
 * bytelog.y - Parser specification for ByteLog
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ByteLog Grammar (EBNF):
 *
 *   program     ::= statement*
 *   statement   ::= rel_decl | fact | rule | solve | query
 *   
 *   rel_decl    ::= 'REL' IDENT
 *   fact        ::= 'FACT' IDENT INTEGER INTEGER
 *   rule        ::= 'RULE' IDENT ':' body ',' emit
 *   solve       ::= 'SOLVE'
 *   query       ::= 'QUERY' IDENT query_arg query_arg
 *   
 *   body        ::= operation (',' operation)*
 *   operation   ::= scan | join
 *   scan        ::= 'SCAN' IDENT ('MATCH' VAR)?
 *   join        ::= 'JOIN' IDENT VAR
 *   emit        ::= 'EMIT' IDENT VAR VAR
 *   
 *   query_arg   ::= INTEGER | '?'
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG,
    OP_GT, OP_LT, OP_GE, OP_LE, OP_EQ, OP_NE,
    OP_LENGTH, OP_CHAR_AT, OP_RANGE,
    OP_POW, OP_ABS, OP_MIN, OP_MAX, OP_SQRT,
    OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN,
    OP_LOG, OP_LOG10, OP_EXP, OP_CEIL, OP_FLOOR
} OpType;

/* ─────────────────────────────────────────────────────────────────────────
 * AST Node Types
 * ───────────────────────────────────────────────────────────────────────── */

typedef enum {
    NODE_PROGRAM,
    NODE_REL_DECL,
    NODE_FACT,
    NODE_RULE,
    NODE_SCAN,
    NODE_JOIN,
    NODE_EMIT,
    NODE_SOLVE,
    NODE_QUERY,
    
    NODE_CALC_DEF,
    NODE_CALC_CALL,
    NODE_INPUT,
    NODE_LET,
    NODE_IF,
    NODE_RESULT,
    NODE_WHERE,
    
    NODE_FOR_EACH,
    NODE_FOR_RANGE,
    NODE_FOR_WHILE,
    NODE_BREAK,
    NODE_CONTINUE,
    NODE_RANGE,
    NODE_STRING_OP,
    NODE_MATH_FUNC,
    
    NODE_EXPR_VAR,
    NODE_EXPR_INT,
    NODE_EXPR_FLOAT,
    NODE_EXPR_STRING,
    NODE_EXPR_BINOP,
    NODE_EXPR_UNARY,
    NODE_CONDITION
} NodeType;

typedef struct ASTNode {
    NodeType type;
    char *name;              /* relation name */
    char *str_val;           /* string literal value */
    int values[4];           /* integers/vars (overloaded by node type) */
    double fval;             /* float value */
    OpType op;               /* operator type */
    int has_match;           /* for SCAN: whether MATCH clause present */
    struct ASTNode *child;   /* first child (body ops for RULE) */
    struct ASTNode *left;    /* left operand for expressions */
    struct ASTNode *right;   /* right operand for expressions */
    struct ASTNode *next;    /* sibling (linked list) */
} ASTNode;

/* ─────────────────────────────────────────────────────────────────────────
 * AST Construction Functions
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode* make_node(NodeType type) {
    ASTNode *node = (ASTNode*)calloc(1, sizeof(ASTNode));
    node->type = type;
    node->values[0] = -1;  /* -1 = wildcard for queries */
    node->values[1] = -1;
    node->values[2] = -1;
    node->values[3] = -1;
    node->fval = 0.0;
    node->op = OP_ADD;  /* default op */
    node->name = NULL;
    node->str_val = NULL;
    node->child = NULL;
    node->left = NULL;
    node->right = NULL;
    node->next = NULL;
    node->has_match = 0;
    return node;
}

ASTNode* make_rel_decl(char *name) {
    ASTNode *node = make_node(NODE_REL_DECL);
    node->name = name;
    return node;
}

ASTNode* make_fact(char *rel, int a, int b) {
    ASTNode *node = make_node(NODE_FACT);
    node->name = rel;
    node->values[0] = a;
    node->values[1] = b;
    return node;
}

ASTNode* make_scan(char *rel, int has_match, int match_var) {
    ASTNode *node = make_node(NODE_SCAN);
    node->name = rel;
    node->has_match = has_match;
    node->values[0] = match_var;
    return node;
}

ASTNode* make_join(char *rel, int match_var) {
    ASTNode *node = make_node(NODE_JOIN);
    node->name = rel;
    node->values[0] = match_var;
    return node;
}

ASTNode* make_emit(char *rel, int var_a, int var_b) {
    ASTNode *node = make_node(NODE_EMIT);
    node->name = rel;
    node->values[0] = var_a;
    node->values[1] = var_b;
    return node;
}

ASTNode* make_rule(char *target, ASTNode *body, ASTNode *emit) {
    ASTNode *node = make_node(NODE_RULE);
    node->name = target;
    node->child = body;
    /* Append emit to end of body chain */
    ASTNode *last = body;
    while (last->next) last = last->next;
    last->next = emit;
    return node;
}

ASTNode* make_solve(void) {
    return make_node(NODE_SOLVE);
}

ASTNode* make_query(char *rel, int a, int b) {
    ASTNode *node = make_node(NODE_QUERY);
    node->name = rel;
    node->values[0] = a;
    node->values[1] = b;
    return node;
}

ASTNode* append_node(ASTNode *list, ASTNode *node) {
    if (!list) return node;
    ASTNode *last = list;
    while (last->next) last = last->next;
    last->next = node;
    return list;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Global AST Root
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode *ast_root = NULL;

/* ─────────────────────────────────────────────────────────────────────────
 * Error Handling
 * ───────────────────────────────────────────────────────────────────────── */

extern int line_num;
extern char *yytext;

void yyerror(const char *s) {
    fprintf(stderr, "Line %d: %s near '%s'\n", line_num, s, yytext);
}

int yylex(void);

%}

/* ─────────────────────────────────────────────────────────────────────────
 * Token Declarations
 * ───────────────────────────────────────────────────────────────────────── */

%union {
    int ival;
    double fval;
    char *sval;
    struct ASTNode *node;
    int op;
}

%token REL FACT RULE SCAN JOIN EMIT MATCH SOLVE QUERY
%token CALC INPUT LET RESULT IF THEN ELSE END WHERE
%token FOR WHILE IN RANGE LENGTH CHAR_AT
%token BREAK CONTINUE MOD POW ABS MIN MAX SQRT
%token SIN COS TAN ASIN ACOS ATAN LOG LOG10 EXP CEIL FLOOR
%token COLON COMMA WILDCARD
%token PLUS MINUS STAR SLASH LPAREN RPAREN ASSIGN
%token GT LT GE LE EQ NE

%token <ival> INTEGER VAR
%token <fval> FLOAT
%token <sval> IDENT STRING

%type <node> program statement_list statement
%type <node> rel_decl fact rule solve query
%type <node> body operation_list operation scan join emit
%type <node> calc_def calc_call let_stmt if_stmt for_stmt break_stmt continue_stmt
%type <node> expr term factor condition string_op range_call math_func
%type <node> stmt_list emit_atom
%type <ival> query_arg
%type <ival> compare_op

/* Precedence declarations */
%left PLUS MINUS
%left STAR SLASH MOD
%right UMINUS

/* ─────────────────────────────────────────────────────────────────────────
 * Grammar Rules
 * ───────────────────────────────────────────────────────────────────────── */

%%

program
    : statement_list
        { ast_root = $1; }
    ;

statement_list
    : /* empty */
        { $$ = NULL; }
    | statement_list statement
        { $$ = append_node($1, $2); }
    ;

statement
    : rel_decl      { $$ = $1; }
    | fact          { $$ = $1; }
    | rule          { $$ = $1; }
    | solve         { $$ = $1; }
    | query         { $$ = $1; }
    | calc_def      { $$ = $1; }
    | calc_call     { $$ = $1; }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * REL <name>
 * ───────────────────────────────────────────────────────────────────────── */

rel_decl
    : REL IDENT
        { $$ = make_rel_decl($2); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * FACT <rel> <int> <int>
 * ───────────────────────────────────────────────────────────────────────── */

fact
    : FACT IDENT INTEGER INTEGER
        { $$ = make_fact($2, $3, $4); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * RULE <target>: <body>, EMIT <rel> <var> <var>
 * ───────────────────────────────────────────────────────────────────────── */

rule
    : RULE IDENT COLON body COMMA emit
        { $$ = make_rule($2, $4, $6); }
    ;

body
    : operation_list
        { $$ = $1; }
    ;

operation_list
    : operation
        { $$ = $1; }
    | operation_list COMMA operation
        { $$ = append_node($1, $3); }
    ;

operation
    : scan          { $$ = $1; }
    | join          { $$ = $1; }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * SCAN <rel> [MATCH <var>]
 * ───────────────────────────────────────────────────────────────────────── */

scan
    : SCAN IDENT
        { $$ = make_scan($2, 0, -1); }
    | SCAN IDENT MATCH VAR
        { $$ = make_scan($2, 1, $4); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * JOIN <rel> <var>
 * ───────────────────────────────────────────────────────────────────────── */

join
    : JOIN IDENT VAR
        { $$ = make_join($2, $3); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * EMIT <rel> <var> <var>
 * ───────────────────────────────────────────────────────────────────────── */

emit
    : EMIT IDENT VAR VAR
        { $$ = make_emit($2, $3, $4); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * SOLVE
 * ───────────────────────────────────────────────────────────────────────── */

solve
    : SOLVE
        { $$ = make_solve(); }
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * QUERY <rel> <arg> <arg>
 * ───────────────────────────────────────────────────────────────────────── */

query
    : QUERY IDENT query_arg query_arg
        { $$ = make_query($2, $3, $4); }
    ;

query_arg
    : INTEGER       { $$ = $1; }
    | WILDCARD      { $$ = -1; }  /* -1 represents wildcard */
    ;

/* ─────────────────────────────────────────────────────────────────────────
 * CALC blocks and expressions (stubbed for now)
 * ───────────────────────────────────────────────────────────────────────── */

calc_def
    : CALC IDENT stmt_list END
        { $$ = make_node(NODE_CALC_DEF); $$->name = $2; $$->child = $3; }
    ;

calc_call
    : CALC IDENT LPAREN RPAREN
        { $$ = make_node(NODE_CALC_CALL); $$->name = $2; }
    ;

stmt_list
    : /* empty */
        { $$ = NULL; }
    | stmt_list let_stmt
        { $$ = append_node($1, $2); }
    | stmt_list for_stmt
        { $$ = append_node($1, $2); }
    | stmt_list if_stmt
        { $$ = append_node($1, $2); }
    | stmt_list break_stmt
        { $$ = append_node($1, $2); }
    | stmt_list continue_stmt
        { $$ = append_node($1, $2); }
    ;

let_stmt
    : LET VAR ASSIGN expr
        { $$ = make_node(NODE_LET); $$->values[0] = $2; $$->child = $4; }
    ;

if_stmt
    : IF condition THEN stmt_list END
        { $$ = make_node(NODE_IF); $$->child = $2; $$->child->next = $4; }
    ;

for_stmt
    : FOR VAR IN RANGE LPAREN expr COMMA expr RPAREN stmt_list END
        { $$ = make_node(NODE_FOR_RANGE); $$->values[0] = $2; $$->left = $6; $$->right = $8; $$->child = $10; }
    | FOR WHILE condition stmt_list END
        { $$ = make_node(NODE_FOR_WHILE); $$->child = $3; $$->child->next = $4; }
    ;

expr
    : term                     { $$ = $1; }
    | expr PLUS term          { $$ = make_node(NODE_EXPR_BINOP); $$->op = OP_ADD; $$->left = $1; $$->right = $3; }
    | expr MINUS term         { $$ = make_node(NODE_EXPR_BINOP); $$->op = OP_SUB; $$->left = $1; $$->right = $3; }
    ;

term
    : factor                  { $$ = $1; }
    | term STAR factor        { $$ = make_node(NODE_EXPR_BINOP); $$->op = OP_MUL; $$->left = $1; $$->right = $3; }
    | term SLASH factor       { $$ = make_node(NODE_EXPR_BINOP); $$->op = OP_DIV; $$->left = $1; $$->right = $3; }
    | term MOD factor         { $$ = make_node(NODE_EXPR_BINOP); $$->op = OP_MOD; $$->left = $1; $$->right = $3; }
    ;

factor
    : INTEGER                 { $$ = make_node(NODE_EXPR_INT); $$->values[0] = $1; }
    | FLOAT                   { $$ = make_node(NODE_EXPR_FLOAT); $$->fval = $1; }
    | STRING                  { $$ = make_node(NODE_EXPR_STRING); $$->str_val = $1; }
    | VAR                     { $$ = make_node(NODE_EXPR_VAR); $$->values[0] = $1; }
    | LPAREN expr RPAREN      { $$ = $2; }
    | MINUS factor %prec UMINUS { $$ = make_node(NODE_EXPR_UNARY); $$->op = OP_NEG; $$->left = $2; }
    | string_op               { $$ = $1; }
    | range_call              { $$ = $1; }
    | calc_call               { $$ = $1; }
    | math_func               { $$ = $1; }
    ;

string_op
    : LENGTH LPAREN expr RPAREN        { $$ = make_node(NODE_STRING_OP); $$->op = OP_LENGTH; $$->left = $3; }
    | CHAR_AT LPAREN expr COMMA expr RPAREN { $$ = make_node(NODE_STRING_OP); $$->op = OP_CHAR_AT; $$->left = $3; $$->right = $5; }
    ;

range_call
    : RANGE LPAREN expr COMMA expr RPAREN
        { $$ = make_node(NODE_RANGE); $$->left = $3; $$->right = $5; }
    ;

condition
    : expr compare_op expr
        { $$ = make_node(NODE_CONDITION); $$->op = $2; $$->left = $1; $$->right = $3; }
    ;

compare_op
    : GT    { $$ = OP_GT; }
    | LT    { $$ = OP_LT; }
    | GE    { $$ = OP_GE; }
    | LE    { $$ = OP_LE; }
    | ASSIGN { $$ = OP_EQ; }
    | EQ    { $$ = OP_EQ; }
    | NE    { $$ = OP_NE; }
    ;

emit_atom
    : VAR                     { $$ = make_node(NODE_EXPR_VAR); $$->values[0] = $1; }
    | INTEGER                 { $$ = make_node(NODE_EXPR_INT); $$->values[0] = $1; }
    | FLOAT                   { $$ = make_node(NODE_EXPR_FLOAT); $$->fval = $1; }
    | STRING                  { $$ = make_node(NODE_EXPR_STRING); $$->str_val = $1; }
    | LPAREN expr RPAREN      { $$ = $2; }
    ;

break_stmt
    : BREAK
        { $$ = make_node(NODE_BREAK); }
    ;

continue_stmt
    : CONTINUE  
        { $$ = make_node(NODE_CONTINUE); }
    ;

math_func
    : POW LPAREN expr COMMA expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_POW; $$->left = $3; $$->right = $5; }
    | ABS LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_ABS; $$->left = $3; }
    | MIN LPAREN expr COMMA expr RPAREN  
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_MIN; $$->left = $3; $$->right = $5; }
    | MAX LPAREN expr COMMA expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_MAX; $$->left = $3; $$->right = $5; }
    | SQRT LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_SQRT; $$->left = $3; }
    | SIN LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_SIN; $$->left = $3; }
    | COS LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_COS; $$->left = $3; }
    | TAN LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_TAN; $$->left = $3; }
    | ASIN LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_ASIN; $$->left = $3; }
    | ACOS LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_ACOS; $$->left = $3; }
    | ATAN LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_ATAN; $$->left = $3; }
    | LOG LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_LOG; $$->left = $3; }
    | LOG10 LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_LOG10; $$->left = $3; }
    | EXP LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_EXP; $$->left = $3; }
    | CEIL LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_CEIL; $$->left = $3; }
    | FLOOR LPAREN expr RPAREN
        { $$ = make_node(NODE_MATH_FUNC); $$->op = OP_FLOOR; $$->left = $3; }
    ;

%%

/* ═══════════════════════════════════════════════════════════════════════════
 * AST Printer (for debugging)
 * ═══════════════════════════════════════════════════════════════════════════
 */

void print_ast(ASTNode *node, int indent) {
    while (node) {
        for (int i = 0; i < indent; i++) printf("  ");
        
        switch (node->type) {
            case NODE_REL_DECL:
                printf("REL %s\n", node->name);
                break;
            case NODE_FACT:
                printf("FACT %s %d %d\n", node->name, node->values[0], node->values[1]);
                break;
            case NODE_RULE:
                printf("RULE %s:\n", node->name);
                print_ast(node->child, indent + 1);
                break;
            case NODE_SCAN:
                if (node->has_match)
                    printf("SCAN %s MATCH $%d\n", node->name, node->values[0]);
                else
                    printf("SCAN %s\n", node->name);
                break;
            case NODE_JOIN:
                printf("JOIN %s $%d\n", node->name, node->values[0]);
                break;
            case NODE_EMIT:
                printf("EMIT %s $%d $%d\n", node->name, node->values[0], node->values[1]);
                break;
            case NODE_SOLVE:
                printf("SOLVE\n");
                break;
            case NODE_QUERY:
                printf("QUERY %s ", node->name);
                if (node->values[0] == -1) printf("? "); else printf("%d ", node->values[0]);
                if (node->values[1] == -1) printf("?\n"); else printf("%d\n", node->values[1]);
                break;
            default:
                printf("UNKNOWN\n");
        }
        
        node = node->next;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main Entry Point
 * ═══════════════════════════════════════════════════════════════════════════
 */

int main(int argc, char **argv) {
    printf("ByteLog Parser v1.0\n");
    printf("═══════════════════════════════════════\n\n");
    
    if (yyparse() == 0) {
        printf("Parse successful!\n\n");
        printf("AST:\n");
        printf("───────────────────────────────────────\n");
        print_ast(ast_root, 0);
    } else {
        printf("Parse failed.\n");
        return 1;
    }
    
    return 0;
}
