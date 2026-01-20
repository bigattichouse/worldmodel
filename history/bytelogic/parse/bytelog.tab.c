/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 27 "bytelog.y"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG,
    OP_GT, OP_LT, OP_GE, OP_LE, OP_EQ, OP_NE,
    OP_LENGTH, OP_CHAR_AT, OP_RANGE,
    OP_POW, OP_ABS, OP_MIN, OP_MAX, OP_SQRT
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


#line 251 "bytelog.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "bytelog.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_REL = 3,                        /* REL  */
  YYSYMBOL_FACT = 4,                       /* FACT  */
  YYSYMBOL_RULE = 5,                       /* RULE  */
  YYSYMBOL_SCAN = 6,                       /* SCAN  */
  YYSYMBOL_JOIN = 7,                       /* JOIN  */
  YYSYMBOL_EMIT = 8,                       /* EMIT  */
  YYSYMBOL_MATCH = 9,                      /* MATCH  */
  YYSYMBOL_SOLVE = 10,                     /* SOLVE  */
  YYSYMBOL_QUERY = 11,                     /* QUERY  */
  YYSYMBOL_CALC = 12,                      /* CALC  */
  YYSYMBOL_INPUT = 13,                     /* INPUT  */
  YYSYMBOL_LET = 14,                       /* LET  */
  YYSYMBOL_RESULT = 15,                    /* RESULT  */
  YYSYMBOL_IF = 16,                        /* IF  */
  YYSYMBOL_THEN = 17,                      /* THEN  */
  YYSYMBOL_ELSE = 18,                      /* ELSE  */
  YYSYMBOL_END = 19,                       /* END  */
  YYSYMBOL_WHERE = 20,                     /* WHERE  */
  YYSYMBOL_FOR = 21,                       /* FOR  */
  YYSYMBOL_WHILE = 22,                     /* WHILE  */
  YYSYMBOL_IN = 23,                        /* IN  */
  YYSYMBOL_RANGE = 24,                     /* RANGE  */
  YYSYMBOL_LENGTH = 25,                    /* LENGTH  */
  YYSYMBOL_CHAR_AT = 26,                   /* CHAR_AT  */
  YYSYMBOL_BREAK = 27,                     /* BREAK  */
  YYSYMBOL_CONTINUE = 28,                  /* CONTINUE  */
  YYSYMBOL_MOD = 29,                       /* MOD  */
  YYSYMBOL_POW = 30,                       /* POW  */
  YYSYMBOL_ABS = 31,                       /* ABS  */
  YYSYMBOL_MIN = 32,                       /* MIN  */
  YYSYMBOL_MAX = 33,                       /* MAX  */
  YYSYMBOL_SQRT = 34,                      /* SQRT  */
  YYSYMBOL_COLON = 35,                     /* COLON  */
  YYSYMBOL_COMMA = 36,                     /* COMMA  */
  YYSYMBOL_WILDCARD = 37,                  /* WILDCARD  */
  YYSYMBOL_PLUS = 38,                      /* PLUS  */
  YYSYMBOL_MINUS = 39,                     /* MINUS  */
  YYSYMBOL_STAR = 40,                      /* STAR  */
  YYSYMBOL_SLASH = 41,                     /* SLASH  */
  YYSYMBOL_LPAREN = 42,                    /* LPAREN  */
  YYSYMBOL_RPAREN = 43,                    /* RPAREN  */
  YYSYMBOL_ASSIGN = 44,                    /* ASSIGN  */
  YYSYMBOL_GT = 45,                        /* GT  */
  YYSYMBOL_LT = 46,                        /* LT  */
  YYSYMBOL_GE = 47,                        /* GE  */
  YYSYMBOL_LE = 48,                        /* LE  */
  YYSYMBOL_EQ = 49,                        /* EQ  */
  YYSYMBOL_NE = 50,                        /* NE  */
  YYSYMBOL_INTEGER = 51,                   /* INTEGER  */
  YYSYMBOL_VAR = 52,                       /* VAR  */
  YYSYMBOL_FLOAT = 53,                     /* FLOAT  */
  YYSYMBOL_IDENT = 54,                     /* IDENT  */
  YYSYMBOL_STRING = 55,                    /* STRING  */
  YYSYMBOL_UMINUS = 56,                    /* UMINUS  */
  YYSYMBOL_YYACCEPT = 57,                  /* $accept  */
  YYSYMBOL_program = 58,                   /* program  */
  YYSYMBOL_statement_list = 59,            /* statement_list  */
  YYSYMBOL_statement = 60,                 /* statement  */
  YYSYMBOL_rel_decl = 61,                  /* rel_decl  */
  YYSYMBOL_fact = 62,                      /* fact  */
  YYSYMBOL_rule = 63,                      /* rule  */
  YYSYMBOL_body = 64,                      /* body  */
  YYSYMBOL_operation_list = 65,            /* operation_list  */
  YYSYMBOL_operation = 66,                 /* operation  */
  YYSYMBOL_scan = 67,                      /* scan  */
  YYSYMBOL_join = 68,                      /* join  */
  YYSYMBOL_emit = 69,                      /* emit  */
  YYSYMBOL_solve = 70,                     /* solve  */
  YYSYMBOL_query = 71,                     /* query  */
  YYSYMBOL_query_arg = 72,                 /* query_arg  */
  YYSYMBOL_calc_def = 73,                  /* calc_def  */
  YYSYMBOL_calc_call = 74,                 /* calc_call  */
  YYSYMBOL_stmt_list = 75,                 /* stmt_list  */
  YYSYMBOL_let_stmt = 76,                  /* let_stmt  */
  YYSYMBOL_if_stmt = 77,                   /* if_stmt  */
  YYSYMBOL_for_stmt = 78,                  /* for_stmt  */
  YYSYMBOL_expr = 79,                      /* expr  */
  YYSYMBOL_term = 80,                      /* term  */
  YYSYMBOL_factor = 81,                    /* factor  */
  YYSYMBOL_string_op = 82,                 /* string_op  */
  YYSYMBOL_range_call = 83,                /* range_call  */
  YYSYMBOL_condition = 84,                 /* condition  */
  YYSYMBOL_compare_op = 85,                /* compare_op  */
  YYSYMBOL_break_stmt = 86,                /* break_stmt  */
  YYSYMBOL_continue_stmt = 87,             /* continue_stmt  */
  YYSYMBOL_math_func = 88                  /* math_func  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   183

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  57
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  74
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  164

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   311


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   252,   252,   258,   259,   264,   265,   266,   267,   268,
     269,   270,   278,   287,   296,   301,   306,   308,   313,   314,
     322,   324,   333,   342,   351,   360,   365,   366,   374,   379,
     385,   386,   388,   390,   392,   394,   399,   404,   409,   411,
     416,   417,   418,   422,   423,   424,   425,   429,   430,   431,
     432,   433,   434,   435,   436,   437,   438,   442,   443,   447,
     452,   457,   458,   459,   460,   461,   462,   463,   475,   480,
     485,   487,   489,   491,   493
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "REL", "FACT", "RULE",
  "SCAN", "JOIN", "EMIT", "MATCH", "SOLVE", "QUERY", "CALC", "INPUT",
  "LET", "RESULT", "IF", "THEN", "ELSE", "END", "WHERE", "FOR", "WHILE",
  "IN", "RANGE", "LENGTH", "CHAR_AT", "BREAK", "CONTINUE", "MOD", "POW",
  "ABS", "MIN", "MAX", "SQRT", "COLON", "COMMA", "WILDCARD", "PLUS",
  "MINUS", "STAR", "SLASH", "LPAREN", "RPAREN", "ASSIGN", "GT", "LT", "GE",
  "LE", "EQ", "NE", "INTEGER", "VAR", "FLOAT", "IDENT", "STRING", "UMINUS",
  "$accept", "program", "statement_list", "statement", "rel_decl", "fact",
  "rule", "body", "operation_list", "operation", "scan", "join", "emit",
  "solve", "query", "query_arg", "calc_def", "calc_call", "stmt_list",
  "let_stmt", "if_stmt", "for_stmt", "expr", "term", "factor", "string_op",
  "range_call", "condition", "compare_op", "break_stmt", "continue_stmt",
  "math_func", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-111)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -111,     6,    49,  -111,   -43,   -37,   -23,  -111,   -15,   -11,
    -111,  -111,  -111,  -111,  -111,  -111,  -111,  -111,  -111,    -6,
      20,   -21,    27,    28,     8,  -111,  -111,   -21,    38,    -9,
    -111,    34,    36,    50,    56,  -111,  -111,  -111,  -111,  -111,
      48,    32,  -111,   -18,  -111,  -111,  -111,  -111,  -111,  -111,
    -111,   101,    63,   109,     8,    76,    67,    91,   107,   113,
     117,   121,   124,   125,   127,    32,    32,  -111,  -111,  -111,
    -111,  -111,    80,    -8,  -111,  -111,  -111,   151,  -111,    32,
     147,   119,  -111,   120,  -111,  -111,    32,    27,    32,    32,
      32,    32,    32,    32,    32,    32,  -111,   -30,    32,    32,
    -111,  -111,  -111,  -111,  -111,  -111,  -111,    32,    32,    32,
      32,  -111,  -111,   148,  -111,   123,    11,    59,    -3,    69,
     114,    -1,   118,   122,    39,  -111,    -8,    -8,    11,  -111,
    -111,  -111,    75,    85,   131,   128,    32,  -111,    32,    32,
    -111,    32,    32,  -111,  -111,  -111,    32,  -111,    93,    96,
      99,   102,   105,   126,  -111,  -111,  -111,  -111,  -111,    32,
     108,  -111,    95,  -111
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       3,     0,     2,     1,     0,     0,     0,    24,     0,     0,
       4,     5,     6,     7,     8,     9,    10,    11,    12,     0,
       0,     0,    30,     0,     0,    27,    26,     0,     0,     0,
      13,     0,     0,     0,     0,    16,    18,    19,    25,    29,
       0,     0,    28,     0,    68,    69,    31,    33,    32,    34,
      35,    20,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    50,    48,
      49,    55,     0,    40,    43,    53,    54,     0,    56,     0,
       0,     0,    22,     0,    14,    17,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,     0,     0,     0,
      65,    61,    62,    63,    64,    66,    67,     0,     0,     0,
       0,    30,    30,     0,    21,     0,    36,     0,     0,     0,
       0,     0,     0,     0,     0,    51,    41,    42,    60,    46,
      44,    45,     0,     0,     0,     0,     0,    57,     0,     0,
      71,     0,     0,    74,    37,    39,     0,    23,     0,     0,
       0,     0,     0,     0,    59,    58,    70,    72,    73,     0,
       0,    30,     0,    38
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -111,  -111,  -111,  -111,  -111,  -111,  -111,  -111,  -111,   129,
    -111,  -111,  -111,  -111,  -111,   149,  -111,   175,  -110,  -111,
    -111,  -111,   -66,   -31,   -62,  -111,  -111,   100,  -111,  -111,
    -111,  -111
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
       0,     1,     2,    10,    11,    12,    13,    33,    34,    35,
      36,    37,    84,    14,    15,    27,    16,    71,    29,    46,
      47,    48,    72,    73,    74,    75,    76,    77,   107,    49,
      50,    78
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      97,   132,   133,    96,    79,    40,     3,    41,    98,    99,
      42,    18,    43,   125,    31,    32,    25,    19,    44,    45,
     116,   108,   117,   118,   119,   120,   121,   122,   123,   124,
      26,    20,   109,   110,    80,    98,    99,    98,    99,    21,
     137,   128,   140,    22,    56,    23,   129,   130,   131,    98,
      99,   162,     4,     5,     6,    24,    57,    58,    59,     7,
       8,     9,    60,    61,    62,    63,    64,   126,   127,    28,
     148,    65,   149,   150,    66,   151,   152,    98,    99,    30,
     153,    39,   143,    67,    68,    69,    53,    70,    51,    40,
      52,    41,    54,   160,   144,   136,    43,    98,    99,    40,
      55,    41,    44,    45,   145,   138,    43,    98,    99,    40,
      81,    41,    44,    45,   163,    82,    43,    83,    98,    99,
      86,    87,    44,    45,   100,   101,   102,   103,   104,   105,
     106,    98,    99,    88,    98,    99,   154,    98,    99,   155,
      98,    99,   156,    98,    99,   157,    98,    99,   158,    89,
     139,   161,    98,    99,   141,    90,    98,    99,   142,    91,
      98,    99,   159,    92,    98,    99,    93,    94,   111,    95,
     113,   114,   134,   146,   115,   135,    38,    17,     0,   112,
     147,     0,     0,    85
};

static const yytype_int16 yycheck[] =
{
      66,   111,   112,    65,    22,    14,     0,    16,    38,    39,
      19,    54,    21,    43,     6,     7,    37,    54,    27,    28,
      86,    29,    88,    89,    90,    91,    92,    93,    94,    95,
      51,    54,    40,    41,    52,    38,    39,    38,    39,    54,
      43,   107,    43,    54,    12,    51,   108,   109,   110,    38,
      39,   161,     3,     4,     5,    35,    24,    25,    26,    10,
      11,    12,    30,    31,    32,    33,    34,    98,    99,    42,
     136,    39,   138,   139,    42,   141,   142,    38,    39,    51,
     146,    43,    43,    51,    52,    53,    36,    55,    54,    14,
      54,    16,    36,   159,    19,    36,    21,    38,    39,    14,
      52,    16,    27,    28,    19,    36,    21,    38,    39,    14,
       9,    16,    27,    28,    19,    52,    21,     8,    38,    39,
      44,    54,    27,    28,    44,    45,    46,    47,    48,    49,
      50,    38,    39,    42,    38,    39,    43,    38,    39,    43,
      38,    39,    43,    38,    39,    43,    38,    39,    43,    42,
      36,    43,    38,    39,    36,    42,    38,    39,    36,    42,
      38,    39,    36,    42,    38,    39,    42,    42,    17,    42,
      23,    52,    24,    42,    54,    52,    27,     2,    -1,    79,
      52,    -1,    -1,    54
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    58,    59,     0,     3,     4,     5,    10,    11,    12,
      60,    61,    62,    63,    70,    71,    73,    74,    54,    54,
      54,    54,    54,    51,    35,    37,    51,    72,    42,    75,
      51,     6,     7,    64,    65,    66,    67,    68,    72,    43,
      14,    16,    19,    21,    27,    28,    76,    77,    78,    86,
      87,    54,    54,    36,    36,    52,    12,    24,    25,    26,
      30,    31,    32,    33,    34,    39,    42,    51,    52,    53,
      55,    74,    79,    80,    81,    82,    83,    84,    88,    22,
      52,     9,    52,     8,    69,    66,    44,    54,    42,    42,
      42,    42,    42,    42,    42,    42,    81,    79,    38,    39,
      44,    45,    46,    47,    48,    49,    50,    85,    29,    40,
      41,    17,    84,    23,    52,    54,    79,    79,    79,    79,
      79,    79,    79,    79,    79,    43,    80,    80,    79,    81,
      81,    81,    75,    75,    24,    52,    36,    43,    36,    36,
      43,    36,    36,    43,    19,    19,    42,    52,    79,    79,
      79,    79,    79,    79,    43,    43,    43,    43,    43,    36,
      79,    43,    75,    19
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    57,    58,    59,    59,    60,    60,    60,    60,    60,
      60,    60,    61,    62,    63,    64,    65,    65,    66,    66,
      67,    67,    68,    69,    70,    71,    72,    72,    73,    74,
      75,    75,    75,    75,    75,    75,    76,    77,    78,    78,
      79,    79,    79,    80,    80,    80,    80,    81,    81,    81,
      81,    81,    81,    81,    81,    81,    81,    82,    82,    83,
      84,    85,    85,    85,    85,    85,    85,    85,    86,    87,
      88,    88,    88,    88,    88
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     1,     1,     1,     1,
       1,     1,     2,     4,     6,     1,     1,     3,     1,     1,
       2,     4,     3,     4,     1,     4,     1,     1,     4,     4,
       0,     2,     2,     2,     2,     2,     4,     5,    11,     5,
       1,     3,     3,     1,     3,     3,     3,     1,     1,     1,
       1,     3,     2,     1,     1,     1,     1,     4,     6,     6,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       6,     4,     6,     6,     4
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* program: statement_list  */
#line 253 "bytelog.y"
        { ast_root = (yyvsp[0].node); }
#line 1440 "bytelog.tab.c"
    break;

  case 3: /* statement_list: %empty  */
#line 258 "bytelog.y"
        { (yyval.node) = NULL; }
#line 1446 "bytelog.tab.c"
    break;

  case 4: /* statement_list: statement_list statement  */
#line 260 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1452 "bytelog.tab.c"
    break;

  case 5: /* statement: rel_decl  */
#line 264 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1458 "bytelog.tab.c"
    break;

  case 6: /* statement: fact  */
#line 265 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1464 "bytelog.tab.c"
    break;

  case 7: /* statement: rule  */
#line 266 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1470 "bytelog.tab.c"
    break;

  case 8: /* statement: solve  */
#line 267 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1476 "bytelog.tab.c"
    break;

  case 9: /* statement: query  */
#line 268 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1482 "bytelog.tab.c"
    break;

  case 10: /* statement: calc_def  */
#line 269 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1488 "bytelog.tab.c"
    break;

  case 11: /* statement: calc_call  */
#line 270 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1494 "bytelog.tab.c"
    break;

  case 12: /* rel_decl: REL IDENT  */
#line 279 "bytelog.y"
        { (yyval.node) = make_rel_decl((yyvsp[0].sval)); }
#line 1500 "bytelog.tab.c"
    break;

  case 13: /* fact: FACT IDENT INTEGER INTEGER  */
#line 288 "bytelog.y"
        { (yyval.node) = make_fact((yyvsp[-2].sval), (yyvsp[-1].ival), (yyvsp[0].ival)); }
#line 1506 "bytelog.tab.c"
    break;

  case 14: /* rule: RULE IDENT COLON body COMMA emit  */
#line 297 "bytelog.y"
        { (yyval.node) = make_rule((yyvsp[-4].sval), (yyvsp[-2].node), (yyvsp[0].node)); }
#line 1512 "bytelog.tab.c"
    break;

  case 15: /* body: operation_list  */
#line 302 "bytelog.y"
        { (yyval.node) = (yyvsp[0].node); }
#line 1518 "bytelog.tab.c"
    break;

  case 16: /* operation_list: operation  */
#line 307 "bytelog.y"
        { (yyval.node) = (yyvsp[0].node); }
#line 1524 "bytelog.tab.c"
    break;

  case 17: /* operation_list: operation_list COMMA operation  */
#line 309 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-2].node), (yyvsp[0].node)); }
#line 1530 "bytelog.tab.c"
    break;

  case 18: /* operation: scan  */
#line 313 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1536 "bytelog.tab.c"
    break;

  case 19: /* operation: join  */
#line 314 "bytelog.y"
                    { (yyval.node) = (yyvsp[0].node); }
#line 1542 "bytelog.tab.c"
    break;

  case 20: /* scan: SCAN IDENT  */
#line 323 "bytelog.y"
        { (yyval.node) = make_scan((yyvsp[0].sval), 0, -1); }
#line 1548 "bytelog.tab.c"
    break;

  case 21: /* scan: SCAN IDENT MATCH VAR  */
#line 325 "bytelog.y"
        { (yyval.node) = make_scan((yyvsp[-2].sval), 1, (yyvsp[0].ival)); }
#line 1554 "bytelog.tab.c"
    break;

  case 22: /* join: JOIN IDENT VAR  */
#line 334 "bytelog.y"
        { (yyval.node) = make_join((yyvsp[-1].sval), (yyvsp[0].ival)); }
#line 1560 "bytelog.tab.c"
    break;

  case 23: /* emit: EMIT IDENT VAR VAR  */
#line 343 "bytelog.y"
        { (yyval.node) = make_emit((yyvsp[-2].sval), (yyvsp[-1].ival), (yyvsp[0].ival)); }
#line 1566 "bytelog.tab.c"
    break;

  case 24: /* solve: SOLVE  */
#line 352 "bytelog.y"
        { (yyval.node) = make_solve(); }
#line 1572 "bytelog.tab.c"
    break;

  case 25: /* query: QUERY IDENT query_arg query_arg  */
#line 361 "bytelog.y"
        { (yyval.node) = make_query((yyvsp[-2].sval), (yyvsp[-1].ival), (yyvsp[0].ival)); }
#line 1578 "bytelog.tab.c"
    break;

  case 26: /* query_arg: INTEGER  */
#line 365 "bytelog.y"
                    { (yyval.ival) = (yyvsp[0].ival); }
#line 1584 "bytelog.tab.c"
    break;

  case 27: /* query_arg: WILDCARD  */
#line 366 "bytelog.y"
                    { (yyval.ival) = -1; }
#line 1590 "bytelog.tab.c"
    break;

  case 28: /* calc_def: CALC IDENT stmt_list END  */
#line 375 "bytelog.y"
        { (yyval.node) = make_node(NODE_CALC_DEF); (yyval.node)->name = (yyvsp[-2].sval); (yyval.node)->child = (yyvsp[-1].node); }
#line 1596 "bytelog.tab.c"
    break;

  case 29: /* calc_call: CALC IDENT LPAREN RPAREN  */
#line 380 "bytelog.y"
        { (yyval.node) = make_node(NODE_CALC_CALL); (yyval.node)->name = (yyvsp[-2].sval); }
#line 1602 "bytelog.tab.c"
    break;

  case 30: /* stmt_list: %empty  */
#line 385 "bytelog.y"
        { (yyval.node) = NULL; }
#line 1608 "bytelog.tab.c"
    break;

  case 31: /* stmt_list: stmt_list let_stmt  */
#line 387 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1614 "bytelog.tab.c"
    break;

  case 32: /* stmt_list: stmt_list for_stmt  */
#line 389 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1620 "bytelog.tab.c"
    break;

  case 33: /* stmt_list: stmt_list if_stmt  */
#line 391 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1626 "bytelog.tab.c"
    break;

  case 34: /* stmt_list: stmt_list break_stmt  */
#line 393 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1632 "bytelog.tab.c"
    break;

  case 35: /* stmt_list: stmt_list continue_stmt  */
#line 395 "bytelog.y"
        { (yyval.node) = append_node((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1638 "bytelog.tab.c"
    break;

  case 36: /* let_stmt: LET VAR ASSIGN expr  */
#line 400 "bytelog.y"
        { (yyval.node) = make_node(NODE_LET); (yyval.node)->values[0] = (yyvsp[-2].ival); (yyval.node)->child = (yyvsp[0].node); }
#line 1644 "bytelog.tab.c"
    break;

  case 37: /* if_stmt: IF condition THEN stmt_list END  */
#line 405 "bytelog.y"
        { (yyval.node) = make_node(NODE_IF); (yyval.node)->child = (yyvsp[-3].node); (yyval.node)->child->next = (yyvsp[-1].node); }
#line 1650 "bytelog.tab.c"
    break;

  case 38: /* for_stmt: FOR VAR IN RANGE LPAREN expr COMMA expr RPAREN stmt_list END  */
#line 410 "bytelog.y"
        { (yyval.node) = make_node(NODE_FOR_RANGE); (yyval.node)->values[0] = (yyvsp[-9].ival); (yyval.node)->left = (yyvsp[-5].node); (yyval.node)->right = (yyvsp[-3].node); (yyval.node)->child = (yyvsp[-1].node); }
#line 1656 "bytelog.tab.c"
    break;

  case 39: /* for_stmt: FOR WHILE condition stmt_list END  */
#line 412 "bytelog.y"
        { (yyval.node) = make_node(NODE_FOR_WHILE); (yyval.node)->child = (yyvsp[-2].node); (yyval.node)->child->next = (yyvsp[-1].node); }
#line 1662 "bytelog.tab.c"
    break;

  case 40: /* expr: term  */
#line 416 "bytelog.y"
                               { (yyval.node) = (yyvsp[0].node); }
#line 1668 "bytelog.tab.c"
    break;

  case 41: /* expr: expr PLUS term  */
#line 417 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_BINOP); (yyval.node)->op = OP_ADD; (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1674 "bytelog.tab.c"
    break;

  case 42: /* expr: expr MINUS term  */
#line 418 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_BINOP); (yyval.node)->op = OP_SUB; (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1680 "bytelog.tab.c"
    break;

  case 43: /* term: factor  */
#line 422 "bytelog.y"
                              { (yyval.node) = (yyvsp[0].node); }
#line 1686 "bytelog.tab.c"
    break;

  case 44: /* term: term STAR factor  */
#line 423 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_BINOP); (yyval.node)->op = OP_MUL; (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1692 "bytelog.tab.c"
    break;

  case 45: /* term: term SLASH factor  */
#line 424 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_BINOP); (yyval.node)->op = OP_DIV; (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1698 "bytelog.tab.c"
    break;

  case 46: /* term: term MOD factor  */
#line 425 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_BINOP); (yyval.node)->op = OP_MOD; (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1704 "bytelog.tab.c"
    break;

  case 47: /* factor: INTEGER  */
#line 429 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_INT); (yyval.node)->values[0] = (yyvsp[0].ival); }
#line 1710 "bytelog.tab.c"
    break;

  case 48: /* factor: FLOAT  */
#line 430 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_FLOAT); (yyval.node)->fval = (yyvsp[0].fval); }
#line 1716 "bytelog.tab.c"
    break;

  case 49: /* factor: STRING  */
#line 431 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_STRING); (yyval.node)->str_val = (yyvsp[0].sval); }
#line 1722 "bytelog.tab.c"
    break;

  case 50: /* factor: VAR  */
#line 432 "bytelog.y"
                              { (yyval.node) = make_node(NODE_EXPR_VAR); (yyval.node)->values[0] = (yyvsp[0].ival); }
#line 1728 "bytelog.tab.c"
    break;

  case 51: /* factor: LPAREN expr RPAREN  */
#line 433 "bytelog.y"
                              { (yyval.node) = (yyvsp[-1].node); }
#line 1734 "bytelog.tab.c"
    break;

  case 52: /* factor: MINUS factor  */
#line 434 "bytelog.y"
                                { (yyval.node) = make_node(NODE_EXPR_UNARY); (yyval.node)->op = OP_NEG; (yyval.node)->left = (yyvsp[0].node); }
#line 1740 "bytelog.tab.c"
    break;

  case 53: /* factor: string_op  */
#line 435 "bytelog.y"
                              { (yyval.node) = (yyvsp[0].node); }
#line 1746 "bytelog.tab.c"
    break;

  case 54: /* factor: range_call  */
#line 436 "bytelog.y"
                              { (yyval.node) = (yyvsp[0].node); }
#line 1752 "bytelog.tab.c"
    break;

  case 55: /* factor: calc_call  */
#line 437 "bytelog.y"
                              { (yyval.node) = (yyvsp[0].node); }
#line 1758 "bytelog.tab.c"
    break;

  case 56: /* factor: math_func  */
#line 438 "bytelog.y"
                              { (yyval.node) = (yyvsp[0].node); }
#line 1764 "bytelog.tab.c"
    break;

  case 57: /* string_op: LENGTH LPAREN expr RPAREN  */
#line 442 "bytelog.y"
                                       { (yyval.node) = make_node(NODE_STRING_OP); (yyval.node)->op = OP_LENGTH; (yyval.node)->left = (yyvsp[-1].node); }
#line 1770 "bytelog.tab.c"
    break;

  case 58: /* string_op: CHAR_AT LPAREN expr COMMA expr RPAREN  */
#line 443 "bytelog.y"
                                            { (yyval.node) = make_node(NODE_STRING_OP); (yyval.node)->op = OP_CHAR_AT; (yyval.node)->left = (yyvsp[-3].node); (yyval.node)->right = (yyvsp[-1].node); }
#line 1776 "bytelog.tab.c"
    break;

  case 59: /* range_call: RANGE LPAREN expr COMMA expr RPAREN  */
#line 448 "bytelog.y"
        { (yyval.node) = make_node(NODE_RANGE); (yyval.node)->left = (yyvsp[-3].node); (yyval.node)->right = (yyvsp[-1].node); }
#line 1782 "bytelog.tab.c"
    break;

  case 60: /* condition: expr compare_op expr  */
#line 453 "bytelog.y"
        { (yyval.node) = make_node(NODE_CONDITION); (yyval.node)->op = (yyvsp[-1].ival); (yyval.node)->left = (yyvsp[-2].node); (yyval.node)->right = (yyvsp[0].node); }
#line 1788 "bytelog.tab.c"
    break;

  case 61: /* compare_op: GT  */
#line 457 "bytelog.y"
            { (yyval.ival) = OP_GT; }
#line 1794 "bytelog.tab.c"
    break;

  case 62: /* compare_op: LT  */
#line 458 "bytelog.y"
            { (yyval.ival) = OP_LT; }
#line 1800 "bytelog.tab.c"
    break;

  case 63: /* compare_op: GE  */
#line 459 "bytelog.y"
            { (yyval.ival) = OP_GE; }
#line 1806 "bytelog.tab.c"
    break;

  case 64: /* compare_op: LE  */
#line 460 "bytelog.y"
            { (yyval.ival) = OP_LE; }
#line 1812 "bytelog.tab.c"
    break;

  case 65: /* compare_op: ASSIGN  */
#line 461 "bytelog.y"
             { (yyval.ival) = OP_EQ; }
#line 1818 "bytelog.tab.c"
    break;

  case 66: /* compare_op: EQ  */
#line 462 "bytelog.y"
            { (yyval.ival) = OP_EQ; }
#line 1824 "bytelog.tab.c"
    break;

  case 67: /* compare_op: NE  */
#line 463 "bytelog.y"
            { (yyval.ival) = OP_NE; }
#line 1830 "bytelog.tab.c"
    break;

  case 68: /* break_stmt: BREAK  */
#line 476 "bytelog.y"
        { (yyval.node) = make_node(NODE_BREAK); }
#line 1836 "bytelog.tab.c"
    break;

  case 69: /* continue_stmt: CONTINUE  */
#line 481 "bytelog.y"
        { (yyval.node) = make_node(NODE_CONTINUE); }
#line 1842 "bytelog.tab.c"
    break;

  case 70: /* math_func: POW LPAREN expr COMMA expr RPAREN  */
#line 486 "bytelog.y"
        { (yyval.node) = make_node(NODE_MATH_FUNC); (yyval.node)->op = OP_POW; (yyval.node)->left = (yyvsp[-3].node); (yyval.node)->right = (yyvsp[-1].node); }
#line 1848 "bytelog.tab.c"
    break;

  case 71: /* math_func: ABS LPAREN expr RPAREN  */
#line 488 "bytelog.y"
        { (yyval.node) = make_node(NODE_MATH_FUNC); (yyval.node)->op = OP_ABS; (yyval.node)->left = (yyvsp[-1].node); }
#line 1854 "bytelog.tab.c"
    break;

  case 72: /* math_func: MIN LPAREN expr COMMA expr RPAREN  */
#line 490 "bytelog.y"
        { (yyval.node) = make_node(NODE_MATH_FUNC); (yyval.node)->op = OP_MIN; (yyval.node)->left = (yyvsp[-3].node); (yyval.node)->right = (yyvsp[-1].node); }
#line 1860 "bytelog.tab.c"
    break;

  case 73: /* math_func: MAX LPAREN expr COMMA expr RPAREN  */
#line 492 "bytelog.y"
        { (yyval.node) = make_node(NODE_MATH_FUNC); (yyval.node)->op = OP_MAX; (yyval.node)->left = (yyvsp[-3].node); (yyval.node)->right = (yyvsp[-1].node); }
#line 1866 "bytelog.tab.c"
    break;

  case 74: /* math_func: SQRT LPAREN expr RPAREN  */
#line 494 "bytelog.y"
        { (yyval.node) = make_node(NODE_MATH_FUNC); (yyval.node)->op = OP_SQRT; (yyval.node)->left = (yyvsp[-1].node); }
#line 1872 "bytelog.tab.c"
    break;


#line 1876 "bytelog.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 497 "bytelog.y"


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
