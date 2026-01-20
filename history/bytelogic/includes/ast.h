/* ═══════════════════════════════════════════════════════════════════════════
 * ast.h - ByteLog Abstract Syntax Tree
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Defines AST node structures and manipulation functions.
 * Based on the grammar specification in bytelog-spec.md
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#ifndef BYTELOG_AST_H
#define BYTELOG_AST_H

#include <stdbool.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────
 * AST Node Types
 * ───────────────────────────────────────────────────────────────────────── */

typedef enum {
    AST_PROGRAM,
    AST_REL_DECL,
    AST_FACT,
    AST_RULE,
    AST_SCAN,
    AST_JOIN,
    AST_EMIT,
    AST_SOLVE,
    AST_QUERY,
    
    AST_CALC_DEF,
    AST_CALC_CALL,
    AST_INPUT,
    AST_LET,
    AST_IF,
    AST_RESULT,
    AST_WHERE,
    
    AST_FOR_EACH,
    AST_FOR_RANGE,
    AST_FOR_WHILE,
    AST_BREAK,
    AST_CONTINUE,
    AST_RANGE,
    AST_STRING_OP,
    AST_MATH_FUNC,
    
    AST_EXPR_VAR,
    AST_EXPR_INT,
    AST_EXPR_FLOAT,
    AST_EXPR_STRING,
    AST_EXPR_BINOP,
    AST_EXPR_UNARY,
    AST_CONDITION
} ASTNodeType;

/* Operator types for expressions and conditions */
typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG,
    OP_GT, OP_LT, OP_GE, OP_LE, OP_EQ, OP_NE,
    OP_LENGTH, OP_CHAR_AT, OP_RANGE,
    OP_POW, OP_ABS, OP_MIN, OP_MAX, OP_SQRT
} OpType;

/* ─────────────────────────────────────────────────────────────────────────
 * AST Node Structure
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct ASTNode {
    ASTNodeType type;
    
    /* Node-specific data */
    union {
        /* Program: list of statements */
        struct {
            struct ASTNode *statements;
        } program;
        
        /* REL declaration: REL name */
        struct {
            char *name;
        } rel_decl;
        
        /* Fact: FACT relation a b */
        struct {
            char *relation;
            int a;                  /* Resolved integer value */
            int b;                  /* Resolved integer value */
            char *atom_a;           /* Original atom name (NULL if was integer) */
            char *atom_b;           /* Original atom name (NULL if was integer) */
        } fact;
        
        /* Rule: RULE target: body, emit */
        struct {
            char *target;
            struct ASTNode *body;       /* List of operations */
            struct ASTNode *emit;       /* Single emit operation */
        } rule;
        
        /* Scan: SCAN relation [MATCH var] */
        struct {
            char *relation;
            bool has_match;
            int match_var;              /* Only valid if has_match */
        } scan;
        
        /* Join: JOIN relation var */
        struct {
            char *relation;
            int match_var;
        } join;
        
        /* Emit: EMIT relation var_a var_b */
        struct {
            char *relation;
            int var_a;
            int var_b;
        } emit;
        
        /* Solve: SOLVE (no data) */
        struct {
            int dummy;  /* Empty struct not allowed in C */
        } solve;
        
        /* Query: QUERY relation arg_a arg_b */
        struct {
            char *relation;
            int arg_a;                  /* -1 = wildcard, resolved integer value */
            int arg_b;                  /* -1 = wildcard, resolved integer value */
            char *atom_a;               /* Original atom name (NULL if was integer/wildcard) */
            char *atom_b;               /* Original atom name (NULL if was integer/wildcard) */
        } query;
        
        /* CALC definition: CALC name { body } */
        struct {
            char *name;
            struct ASTNode *body;       /* List of statements */
            struct ASTNode *input;      /* INPUT declaration (optional) */
        } calc_def;
        
        /* CALC call: CALC name(args) */
        struct {
            char *name;
            struct ASTNode *args;       /* List of expressions */
        } calc_call;
        
        /* INPUT declaration: INPUT var_list */
        struct {
            struct ASTNode *vars;       /* List of variables */
        } input;
        
        /* LET assignment: LET var = expr */
        struct {
            int var;                    /* Variable number */
            struct ASTNode *expr;       /* Expression */
        } let;
        
        /* IF statement: IF condition THEN body [ELSE body] */
        struct {
            struct ASTNode *condition;
            struct ASTNode *then_body;
            struct ASTNode *else_body;  /* NULL if no else */
        } if_stmt;
        
        /* RESULT statement: RESULT expr */
        struct {
            struct ASTNode *expr;
        } result;
        
        /* WHERE clause: WHERE condition */
        struct {
            struct ASTNode *condition;
        } where;
        
        /* FOR-EACH: FOR var IN (QUERY ...) */
        struct {
            int var;                    /* Loop variable */
            struct ASTNode *query;      /* Query expression */
            struct ASTNode *body;       /* Loop body */
        } for_each;
        
        /* FOR-RANGE: FOR var IN RANGE(start, end) */
        struct {
            int var;                    /* Loop variable */
            struct ASTNode *start;      /* Start expression */
            struct ASTNode *end;        /* End expression */
            struct ASTNode *body;       /* Loop body */
        } for_range;
        
        /* FOR-WHILE: FOR WHILE condition */
        struct {
            struct ASTNode *condition;
            struct ASTNode *body;       /* Loop body */
        } for_while;
        
        /* RANGE function: RANGE(start, end) */
        struct {
            struct ASTNode *start;
            struct ASTNode *end;
        } range;
        
        /* String operations: LENGTH, CHAR_AT */
        struct {
            OpType op;                  /* OP_LENGTH, OP_CHAR_AT */
            struct ASTNode *arg1;       /* First argument */
            struct ASTNode *arg2;       /* Second argument (for CHAR_AT) */
        } string_op;
        
        /* Mathematical functions: POW, ABS, MIN, MAX, SQRT */
        struct {
            OpType op;                  /* OP_POW, OP_ABS, OP_MIN, OP_MAX, OP_SQRT */
            struct ASTNode *arg1;       /* First argument */
            struct ASTNode *arg2;       /* Second argument (for binary functions) */
        } math_func;
        
        /* BREAK statement */
        struct {
            int dummy;                  /* No data needed */
        } break_stmt;
        
        /* CONTINUE statement */
        struct {
            int dummy;                  /* No data needed */
        } continue_stmt;
        
        /* Expression nodes */
        struct {
            int var_num;                /* Variable number for EXPR_VAR */
            int int_val;                /* Integer value for EXPR_INT */
            double float_val;           /* Float value for EXPR_FLOAT */
            char *string_val;           /* String value for EXPR_STRING */
            OpType op;                  /* Operator for EXPR_BINOP/EXPR_UNARY */
            struct ASTNode *left;       /* Left operand */
            struct ASTNode *right;      /* Right operand */
        } expr;
        
        /* Condition: expr op expr */
        struct {
            OpType op;                  /* Comparison operator */
            struct ASTNode *left;
            struct ASTNode *right;
        } condition;
    } data;
    
    /* Source location */
    int line;
    int column;
    
    /* Linked list for sibling nodes */
    struct ASTNode *next;
} ASTNode;

/* ─────────────────────────────────────────────────────────────────────────
 * AST Constructor Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Create program node */
ASTNode* ast_make_program(ASTNode *statements);

/* Create relation declaration */
ASTNode* ast_make_rel_decl(const char *name, int line, int column);

/* Create fact */
ASTNode* ast_make_fact(const char *relation, int a, int b, int line, int column);

/* Create fact with atom names */
ASTNode* ast_make_fact_with_atoms(const char *relation, int a, int b, 
                                  const char *atom_a, const char *atom_b, 
                                  int line, int column);

/* Create rule */
ASTNode* ast_make_rule(const char *target, ASTNode *body, ASTNode *emit, int line, int column);

/* Create scan operation */
ASTNode* ast_make_scan(const char *relation, bool has_match, int match_var, int line, int column);

/* Create join operation */
ASTNode* ast_make_join(const char *relation, int match_var, int line, int column);

/* Create emit operation */
ASTNode* ast_make_emit(const char *relation, int var_a, int var_b, int line, int column);

/* Create solve statement */
ASTNode* ast_make_solve(int line, int column);

/* Create query statement */
ASTNode* ast_make_query(const char *relation, int arg_a, int arg_b, int line, int column);

/* Create query statement with atom names */
ASTNode* ast_make_query_with_atoms(const char *relation, int arg_a, int arg_b,
                                   const char *atom_a, const char *atom_b,
                                   int line, int column);

/* Create CALC definition */
ASTNode* ast_make_calc_def(const char *name, ASTNode *input, ASTNode *body, int line, int column);

/* Create CALC call */
ASTNode* ast_make_calc_call(const char *name, ASTNode *args, int line, int column);

/* Create INPUT declaration */
ASTNode* ast_make_input(ASTNode *vars, int line, int column);

/* Create LET assignment */
ASTNode* ast_make_let(int var, ASTNode *expr, int line, int column);

/* Create IF statement */
ASTNode* ast_make_if(ASTNode *condition, ASTNode *then_body, ASTNode *else_body, int line, int column);

/* Create RESULT statement */
ASTNode* ast_make_result(ASTNode *expr, int line, int column);

/* Create WHERE clause */
ASTNode* ast_make_where(ASTNode *condition, int line, int column);

/* Create FOR-EACH loop */
ASTNode* ast_make_for_each(int var, ASTNode *query, ASTNode *body, int line, int column);

/* Create FOR-RANGE loop */
ASTNode* ast_make_for_range(int var, ASTNode *start, ASTNode *end, ASTNode *body, int line, int column);

/* Create FOR-WHILE loop */
ASTNode* ast_make_for_while(ASTNode *condition, ASTNode *body, int line, int column);

/* Create RANGE function */
ASTNode* ast_make_range(ASTNode *start, ASTNode *end, int line, int column);

/* Create string operation */
ASTNode* ast_make_string_op(OpType op, ASTNode *arg1, ASTNode *arg2, int line, int column);

/* Create variable expression */
ASTNode* ast_make_expr_var(int var_num, int line, int column);

/* Create integer expression */
ASTNode* ast_make_expr_int(int value, int line, int column);

/* Create float expression */
ASTNode* ast_make_expr_float(double value, int line, int column);

/* Create string expression */
ASTNode* ast_make_expr_string(const char *value, int line, int column);

/* Create binary operation expression */
ASTNode* ast_make_expr_binop(OpType op, ASTNode *left, ASTNode *right, int line, int column);

/* Create unary operation expression */
ASTNode* ast_make_expr_unary(OpType op, ASTNode *operand, int line, int column);

/* Create condition */
ASTNode* ast_make_condition(OpType op, ASTNode *left, ASTNode *right, int line, int column);

/* Create mathematical function */
ASTNode* ast_make_math_func(OpType op, ASTNode *arg1, ASTNode *arg2, int line, int column);

/* Create BREAK statement */
ASTNode* ast_make_break(int line, int column);

/* Create CONTINUE statement */
ASTNode* ast_make_continue(int line, int column);

/* ─────────────────────────────────────────────────────────────────────────
 * AST Manipulation Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Append node to end of linked list */
ASTNode* ast_append(ASTNode *list, ASTNode *node);

/* Count nodes in linked list */
int ast_count_nodes(ASTNode *list);

/* Get nth node from linked list (0-based) */
ASTNode* ast_get_nth(ASTNode *list, int index);

/* Free AST node and all its children */
void ast_free(ASTNode *node);

/* Free entire AST tree recursively */
void ast_free_tree(ASTNode *root);

/* ─────────────────────────────────────────────────────────────────────────
 * AST Utility Functions
 * ───────────────────────────────────────────────────────────────────────── */

/* Print AST node (for debugging) */
void ast_print_node(const ASTNode *node, int indent);

/* Print entire AST tree */
void ast_print_tree(const ASTNode *root);

/* Get string name for AST node type */
const char* ast_node_type_name(ASTNodeType type);

/* Clone AST node (deep copy) */
ASTNode* ast_clone(const ASTNode *node);

/* ─────────────────────────────────────────────────────────────────────────
 * AST Visitor Pattern
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct ASTVisitor {
    void (*visit_program)(const ASTNode *node, void *context);
    void (*visit_rel_decl)(const ASTNode *node, void *context);
    void (*visit_fact)(const ASTNode *node, void *context);
    void (*visit_rule)(const ASTNode *node, void *context);
    void (*visit_scan)(const ASTNode *node, void *context);
    void (*visit_join)(const ASTNode *node, void *context);
    void (*visit_emit)(const ASTNode *node, void *context);
    void (*visit_solve)(const ASTNode *node, void *context);
    void (*visit_query)(const ASTNode *node, void *context);
    void (*visit_calc_def)(const ASTNode *node, void *context);
    void (*visit_calc_call)(const ASTNode *node, void *context);
    void (*visit_input)(const ASTNode *node, void *context);
    void (*visit_let)(const ASTNode *node, void *context);
    void (*visit_if)(const ASTNode *node, void *context);
    void (*visit_result)(const ASTNode *node, void *context);
    void (*visit_where)(const ASTNode *node, void *context);
    void (*visit_for_each)(const ASTNode *node, void *context);
    void (*visit_for_range)(const ASTNode *node, void *context);
    void (*visit_for_while)(const ASTNode *node, void *context);
    void (*visit_range)(const ASTNode *node, void *context);
    void (*visit_string_op)(const ASTNode *node, void *context);
    void (*visit_expr_var)(const ASTNode *node, void *context);
    void (*visit_expr_int)(const ASTNode *node, void *context);
    void (*visit_expr_float)(const ASTNode *node, void *context);
    void (*visit_expr_string)(const ASTNode *node, void *context);
    void (*visit_expr_binop)(const ASTNode *node, void *context);
    void (*visit_expr_unary)(const ASTNode *node, void *context);
    void (*visit_condition)(const ASTNode *node, void *context);
} ASTVisitor;

/* Walk AST tree with visitor pattern */
void ast_walk(const ASTNode *node, const ASTVisitor *visitor, void *context);

/* ─────────────────────────────────────────────────────────────────────────
 * AST Validation
 * ───────────────────────────────────────────────────────────────────────── */

/* Check if AST is well-formed */
bool ast_validate(const ASTNode *root, char *error_buf, size_t error_buf_size);

#endif /* BYTELOG_AST_H */