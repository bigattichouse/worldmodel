/* ═══════════════════════════════════════════════════════════════════════════
 * ast.c - ByteLog AST Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implementation of AST node creation and manipulation functions.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "ast.h"
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
 * Private Helper Functions
 * ───────────────────────────────────────────────────────────────────────── */

static ASTNode* ast_alloc_node(ASTNodeType type, int line, int column) {
    ASTNode *node = calloc(1, sizeof(ASTNode));
    if (!node) return NULL;
    
    node->type = type;
    node->line = line;
    node->column = column;
    node->next = NULL;
    
    return node;
}

static char* ast_copy_string(const char *str) {
    return str ? strdup(str) : NULL;
}

/* ─────────────────────────────────────────────────────────────────────────
 * AST Constructor Functions
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode* ast_make_program(ASTNode *statements) {
    ASTNode *node = ast_alloc_node(AST_PROGRAM, 1, 1);
    if (!node) return NULL;
    
    node->data.program.statements = statements;
    return node;
}

ASTNode* ast_make_rel_decl(const char *name, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_REL_DECL, line, column);
    if (!node) return NULL;
    
    node->data.rel_decl.name = ast_copy_string(name);
    return node;
}

ASTNode* ast_make_fact(const char *relation, int a, int b, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_FACT, line, column);
    if (!node) return NULL;
    
    node->data.fact.relation = ast_copy_string(relation);
    node->data.fact.a = a;
    node->data.fact.b = b;
    node->data.fact.atom_a = NULL;
    node->data.fact.atom_b = NULL;
    return node;
}

ASTNode* ast_make_fact_with_atoms(const char *relation, int a, int b, 
                                  const char *atom_a, const char *atom_b, 
                                  int line, int column) {
    ASTNode *node = ast_alloc_node(AST_FACT, line, column);
    if (!node) return NULL;
    
    node->data.fact.relation = ast_copy_string(relation);
    node->data.fact.a = a;
    node->data.fact.b = b;
    node->data.fact.atom_a = ast_copy_string(atom_a);
    node->data.fact.atom_b = ast_copy_string(atom_b);
    return node;
}

ASTNode* ast_make_rule(const char *target, ASTNode *body, ASTNode *emit, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_RULE, line, column);
    if (!node) return NULL;
    
    node->data.rule.target = ast_copy_string(target);
    node->data.rule.body = body;
    node->data.rule.emit = emit;
    return node;
}

ASTNode* ast_make_scan(const char *relation, bool has_match, int match_var, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_SCAN, line, column);
    if (!node) return NULL;
    
    node->data.scan.relation = ast_copy_string(relation);
    node->data.scan.has_match = has_match;
    node->data.scan.match_var = match_var;
    return node;
}

ASTNode* ast_make_join(const char *relation, int match_var, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_JOIN, line, column);
    if (!node) return NULL;
    
    node->data.join.relation = ast_copy_string(relation);
    node->data.join.match_var = match_var;
    return node;
}

ASTNode* ast_make_emit(const char *relation, int var_a, int var_b, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EMIT, line, column);
    if (!node) return NULL;
    
    node->data.emit.relation = ast_copy_string(relation);
    node->data.emit.var_a = var_a;
    node->data.emit.var_b = var_b;
    return node;
}

ASTNode* ast_make_solve(int line, int column) {
    ASTNode *node = ast_alloc_node(AST_SOLVE, line, column);
    if (!node) return NULL;
    
    node->data.solve.dummy = 0;
    return node;
}

ASTNode* ast_make_query(const char *relation, int arg_a, int arg_b, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_QUERY, line, column);
    if (!node) return NULL;
    
    node->data.query.relation = ast_copy_string(relation);
    node->data.query.arg_a = arg_a;
    node->data.query.arg_b = arg_b;
    node->data.query.atom_a = NULL;
    node->data.query.atom_b = NULL;
    return node;
}

ASTNode* ast_make_query_with_atoms(const char *relation, int arg_a, int arg_b,
                                   const char *atom_a, const char *atom_b,
                                   int line, int column) {
    ASTNode *node = ast_alloc_node(AST_QUERY, line, column);
    if (!node) return NULL;
    
    node->data.query.relation = ast_copy_string(relation);
    node->data.query.arg_a = arg_a;
    node->data.query.arg_b = arg_b;
    node->data.query.atom_a = ast_copy_string(atom_a);
    node->data.query.atom_b = ast_copy_string(atom_b);
    return node;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ByteLog 3.0 Extensions - Loop and Expression Constructors
 * ═══════════════════════════════════════════════════════════════════════════
 */

ASTNode* ast_make_calc_def(const char *name, ASTNode *input, ASTNode *body, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_CALC_DEF, line, column);
    if (!node) return NULL;
    
    node->data.calc_def.name = ast_copy_string(name);
    node->data.calc_def.input = input;
    node->data.calc_def.body = body;
    return node;
}

ASTNode* ast_make_calc_call(const char *name, ASTNode *args, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_CALC_CALL, line, column);
    if (!node) return NULL;
    
    node->data.calc_call.name = ast_copy_string(name);
    node->data.calc_call.args = args;
    return node;
}

ASTNode* ast_make_input(ASTNode *vars, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_INPUT, line, column);
    if (!node) return NULL;
    
    node->data.input.vars = vars;
    return node;
}

ASTNode* ast_make_let(int var, ASTNode *expr, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_LET, line, column);
    if (!node) return NULL;
    
    node->data.let.var = var;
    node->data.let.expr = expr;
    return node;
}

ASTNode* ast_make_if(ASTNode *condition, ASTNode *then_body, ASTNode *else_body, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_IF, line, column);
    if (!node) return NULL;
    
    node->data.if_stmt.condition = condition;
    node->data.if_stmt.then_body = then_body;
    node->data.if_stmt.else_body = else_body;
    return node;
}

ASTNode* ast_make_result(ASTNode *expr, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_RESULT, line, column);
    if (!node) return NULL;
    
    node->data.result.expr = expr;
    return node;
}

ASTNode* ast_make_where(ASTNode *condition, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_WHERE, line, column);
    if (!node) return NULL;
    
    node->data.where.condition = condition;
    return node;
}

ASTNode* ast_make_for_each(int var, ASTNode *query, ASTNode *body, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_FOR_EACH, line, column);
    if (!node) return NULL;
    
    node->data.for_each.var = var;
    node->data.for_each.query = query;
    node->data.for_each.body = body;
    return node;
}

ASTNode* ast_make_for_range(int var, ASTNode *start, ASTNode *end, ASTNode *body, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_FOR_RANGE, line, column);
    if (!node) return NULL;
    
    node->data.for_range.var = var;
    node->data.for_range.start = start;
    node->data.for_range.end = end;
    node->data.for_range.body = body;
    return node;
}

ASTNode* ast_make_for_while(ASTNode *condition, ASTNode *body, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_FOR_WHILE, line, column);
    if (!node) return NULL;
    
    node->data.for_while.condition = condition;
    node->data.for_while.body = body;
    return node;
}

ASTNode* ast_make_range(ASTNode *start, ASTNode *end, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_RANGE, line, column);
    if (!node) return NULL;
    
    node->data.range.start = start;
    node->data.range.end = end;
    return node;
}

ASTNode* ast_make_string_op(OpType op, ASTNode *arg1, ASTNode *arg2, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_STRING_OP, line, column);
    if (!node) return NULL;
    
    node->data.string_op.op = op;
    node->data.string_op.arg1 = arg1;
    node->data.string_op.arg2 = arg2;
    return node;
}

ASTNode* ast_make_expr_var(int var_num, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_VAR, line, column);
    if (!node) return NULL;
    
    node->data.expr.var_num = var_num;
    return node;
}

ASTNode* ast_make_expr_int(int value, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_INT, line, column);
    if (!node) return NULL;
    
    node->data.expr.int_val = value;
    return node;
}

ASTNode* ast_make_expr_float(double value, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_FLOAT, line, column);
    if (!node) return NULL;
    
    node->data.expr.float_val = value;
    return node;
}

ASTNode* ast_make_expr_string(const char *value, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_STRING, line, column);
    if (!node) return NULL;
    
    node->data.expr.string_val = ast_copy_string(value);
    return node;
}

ASTNode* ast_make_expr_binop(OpType op, ASTNode *left, ASTNode *right, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_BINOP, line, column);
    if (!node) return NULL;
    
    node->data.expr.op = op;
    node->data.expr.left = left;
    node->data.expr.right = right;
    return node;
}

ASTNode* ast_make_expr_unary(OpType op, ASTNode *operand, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_EXPR_UNARY, line, column);
    if (!node) return NULL;
    
    node->data.expr.op = op;
    node->data.expr.left = operand;
    node->data.expr.right = NULL;
    return node;
}

ASTNode* ast_make_condition(OpType op, ASTNode *left, ASTNode *right, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_CONDITION, line, column);
    if (!node) return NULL;
    
    node->data.condition.op = op;
    node->data.condition.left = left;
    node->data.condition.right = right;
    return node;
}

ASTNode* ast_make_math_func(OpType op, ASTNode *arg1, ASTNode *arg2, int line, int column) {
    ASTNode *node = ast_alloc_node(AST_MATH_FUNC, line, column);
    if (!node) return NULL;
    
    node->data.math_func.op = op;
    node->data.math_func.arg1 = arg1;
    node->data.math_func.arg2 = arg2;
    return node;
}

ASTNode* ast_make_break(int line, int column) {
    ASTNode *node = ast_alloc_node(AST_BREAK, line, column);
    if (!node) return NULL;
    
    node->data.break_stmt.dummy = 0;
    return node;
}

ASTNode* ast_make_continue(int line, int column) {
    ASTNode *node = ast_alloc_node(AST_CONTINUE, line, column);
    if (!node) return NULL;
    
    node->data.continue_stmt.dummy = 0;
    return node;
}

/* ─────────────────────────────────────────────────────────────────────────
 * AST Manipulation Functions
 * ───────────────────────────────────────────────────────────────────────── */

ASTNode* ast_append(ASTNode *list, ASTNode *node) {
    if (!node) return list;
    if (!list) return node;
    
    ASTNode *current = list;
    while (current->next) {
        current = current->next;
    }
    current->next = node;
    
    return list;
}

int ast_count_nodes(ASTNode *list) {
    int count = 0;
    ASTNode *current = list;
    
    while (current) {
        count++;
        current = current->next;
    }
    
    return count;
}

ASTNode* ast_get_nth(ASTNode *list, int index) {
    ASTNode *current = list;
    int i = 0;
    
    while (current && i < index) {
        current = current->next;
        i++;
    }
    
    return current;
}

void ast_free(ASTNode *node) {
    if (!node) return;
    
    /* Free strings */
    switch (node->type) {
        case AST_REL_DECL:
            free(node->data.rel_decl.name);
            break;
        case AST_FACT:
            free(node->data.fact.relation);
            free(node->data.fact.atom_a);
            free(node->data.fact.atom_b);
            break;
        case AST_RULE:
            free(node->data.rule.target);
            break;
        case AST_SCAN:
            free(node->data.scan.relation);
            break;
        case AST_JOIN:
            free(node->data.join.relation);
            break;
        case AST_EMIT:
            free(node->data.emit.relation);
            break;
        case AST_QUERY:
            free(node->data.query.relation);
            free(node->data.query.atom_a);
            free(node->data.query.atom_b);
            break;
        case AST_PROGRAM:
        case AST_SOLVE:
            break;
    }
    
    free(node);
}

void ast_free_tree(ASTNode *root) {
    if (!root) return;
    
    /* Free children first */
    switch (root->type) {
        case AST_PROGRAM:
            ast_free_tree(root->data.program.statements);
            break;
        case AST_RULE:
            ast_free_tree(root->data.rule.body);
            ast_free_tree(root->data.rule.emit);
            break;
        default:
            break;
    }
    
    /* Free siblings */
    ast_free_tree(root->next);
    
    /* Free this node */
    ast_free(root);
}

/* ─────────────────────────────────────────────────────────────────────────
 * AST Utility Functions
 * ───────────────────────────────────────────────────────────────────────── */

const char* ast_node_type_name(ASTNodeType type) {
    switch (type) {
        case AST_PROGRAM: return "PROGRAM";
        case AST_REL_DECL: return "REL_DECL";
        case AST_FACT: return "FACT";
        case AST_RULE: return "RULE";
        case AST_SCAN: return "SCAN";
        case AST_JOIN: return "JOIN";
        case AST_EMIT: return "EMIT";
        case AST_SOLVE: return "SOLVE";
        case AST_QUERY: return "QUERY";
        default: return "UNKNOWN";
    }
}

static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("  ");
    }
}

void ast_print_node(const ASTNode *node, int indent) {
    if (!node) return;
    
    print_indent(indent);
    printf("%s @%d:%d", ast_node_type_name(node->type), node->line, node->column);
    
    switch (node->type) {
        case AST_PROGRAM:
            printf("\n");
            ast_print_tree(node->data.program.statements);
            break;
            
        case AST_REL_DECL:
            printf(" name='%s'\n", node->data.rel_decl.name);
            break;
            
        case AST_FACT:
            printf(" relation='%s'", node->data.fact.relation);
            if (node->data.fact.atom_a) {
                printf(" a=%s", node->data.fact.atom_a);
            } else {
                printf(" a=%d", node->data.fact.a);
            }
            if (node->data.fact.atom_b) {
                printf(" b=%s", node->data.fact.atom_b);
            } else {
                printf(" b=%d", node->data.fact.b);
            }
            printf("\n");
            break;
            
        case AST_RULE:
            printf(" target='%s'\n", node->data.rule.target);
            if (node->data.rule.body) {
                print_indent(indent + 1);
                printf("body:\n");
                ast_print_node(node->data.rule.body, indent + 2);
            }
            if (node->data.rule.emit) {
                print_indent(indent + 1);
                printf("emit:\n");
                ast_print_node(node->data.rule.emit, indent + 2);
            }
            break;
            
        case AST_SCAN:
            printf(" relation='%s'", node->data.scan.relation);
            if (node->data.scan.has_match) {
                printf(" match=$%d", node->data.scan.match_var);
            }
            printf("\n");
            break;
            
        case AST_JOIN:
            printf(" relation='%s' match=$%d\n", 
                   node->data.join.relation, 
                   node->data.join.match_var);
            break;
            
        case AST_EMIT:
            printf(" relation='%s' var_a=$%d var_b=$%d\n", 
                   node->data.emit.relation, 
                   node->data.emit.var_a, 
                   node->data.emit.var_b);
            break;
            
        case AST_SOLVE:
            printf("\n");
            break;
            
        case AST_QUERY:
            printf(" relation='%s'", node->data.query.relation);
            
            printf(" arg_a=");
            if (node->data.query.arg_a == -1) {
                printf("?");
            } else if (node->data.query.atom_a) {
                printf("%s", node->data.query.atom_a);
            } else {
                printf("%d", node->data.query.arg_a);
            }
            
            printf(" arg_b=");
            if (node->data.query.arg_b == -1) {
                printf("?");
            } else if (node->data.query.atom_b) {
                printf("%s", node->data.query.atom_b);
            } else {
                printf("%d", node->data.query.arg_b);
            }
            printf("\n");
            break;
    }
}

void ast_print_tree(const ASTNode *root) {
    const ASTNode *current = root;
    
    while (current) {
        ast_print_node(current, 0);
        current = current->next;
    }
}

ASTNode* ast_clone(const ASTNode *node) {
    if (!node) return NULL;
    
    ASTNode *clone = NULL;
    
    switch (node->type) {
        case AST_PROGRAM:
            clone = ast_make_program(ast_clone(node->data.program.statements));
            break;
            
        case AST_REL_DECL:
            clone = ast_make_rel_decl(node->data.rel_decl.name, node->line, node->column);
            break;
            
        case AST_FACT:
            clone = ast_make_fact(node->data.fact.relation, 
                                 node->data.fact.a, 
                                 node->data.fact.b,
                                 node->line, node->column);
            break;
            
        case AST_RULE:
            clone = ast_make_rule(node->data.rule.target,
                                 ast_clone(node->data.rule.body),
                                 ast_clone(node->data.rule.emit),
                                 node->line, node->column);
            break;
            
        case AST_SCAN:
            clone = ast_make_scan(node->data.scan.relation,
                                 node->data.scan.has_match,
                                 node->data.scan.match_var,
                                 node->line, node->column);
            break;
            
        case AST_JOIN:
            clone = ast_make_join(node->data.join.relation,
                                 node->data.join.match_var,
                                 node->line, node->column);
            break;
            
        case AST_EMIT:
            clone = ast_make_emit(node->data.emit.relation,
                                 node->data.emit.var_a,
                                 node->data.emit.var_b,
                                 node->line, node->column);
            break;
            
        case AST_SOLVE:
            clone = ast_make_solve(node->line, node->column);
            break;
            
        case AST_QUERY:
            clone = ast_make_query(node->data.query.relation,
                                  node->data.query.arg_a,
                                  node->data.query.arg_b,
                                  node->line, node->column);
            break;
    }
    
    if (clone) {
        clone->next = ast_clone(node->next);
    }
    
    return clone;
}

/* ─────────────────────────────────────────────────────────────────────────
 * AST Visitor Pattern
 * ───────────────────────────────────────────────────────────────────────── */

void ast_walk(const ASTNode *node, const ASTVisitor *visitor, void *context) {
    if (!node || !visitor) return;
    
    /* Visit this node */
    switch (node->type) {
        case AST_PROGRAM:
            if (visitor->visit_program) visitor->visit_program(node, context);
            ast_walk(node->data.program.statements, visitor, context);
            break;
            
        case AST_REL_DECL:
            if (visitor->visit_rel_decl) visitor->visit_rel_decl(node, context);
            break;
            
        case AST_FACT:
            if (visitor->visit_fact) visitor->visit_fact(node, context);
            break;
            
        case AST_RULE:
            if (visitor->visit_rule) visitor->visit_rule(node, context);
            ast_walk(node->data.rule.body, visitor, context);
            ast_walk(node->data.rule.emit, visitor, context);
            break;
            
        case AST_SCAN:
            if (visitor->visit_scan) visitor->visit_scan(node, context);
            break;
            
        case AST_JOIN:
            if (visitor->visit_join) visitor->visit_join(node, context);
            break;
            
        case AST_EMIT:
            if (visitor->visit_emit) visitor->visit_emit(node, context);
            break;
            
        case AST_SOLVE:
            if (visitor->visit_solve) visitor->visit_solve(node, context);
            break;
            
        case AST_QUERY:
            if (visitor->visit_query) visitor->visit_query(node, context);
            break;
    }
    
    /* Visit siblings */
    ast_walk(node->next, visitor, context);
}

/* ─────────────────────────────────────────────────────────────────────────
 * AST Validation
 * ───────────────────────────────────────────────────────────────────────── */

bool ast_validate(const ASTNode *root, char *error_buf, size_t error_buf_size) {
    if (!root) {
        snprintf(error_buf, error_buf_size, "Empty AST");
        return false;
    }
    
    if (root->type != AST_PROGRAM) {
        snprintf(error_buf, error_buf_size, "Root must be PROGRAM node");
        return false;
    }
    
    /* TODO: Add more validation rules */
    return true;
}