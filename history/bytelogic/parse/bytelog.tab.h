/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_BYTELOG_TAB_H_INCLUDED
# define YY_YY_BYTELOG_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    REL = 258,                     /* REL  */
    FACT = 259,                    /* FACT  */
    RULE = 260,                    /* RULE  */
    SCAN = 261,                    /* SCAN  */
    JOIN = 262,                    /* JOIN  */
    EMIT = 263,                    /* EMIT  */
    MATCH = 264,                   /* MATCH  */
    SOLVE = 265,                   /* SOLVE  */
    QUERY = 266,                   /* QUERY  */
    CALC = 267,                    /* CALC  */
    INPUT = 268,                   /* INPUT  */
    LET = 269,                     /* LET  */
    RESULT = 270,                  /* RESULT  */
    IF = 271,                      /* IF  */
    THEN = 272,                    /* THEN  */
    ELSE = 273,                    /* ELSE  */
    END = 274,                     /* END  */
    WHERE = 275,                   /* WHERE  */
    FOR = 276,                     /* FOR  */
    WHILE = 277,                   /* WHILE  */
    IN = 278,                      /* IN  */
    RANGE = 279,                   /* RANGE  */
    LENGTH = 280,                  /* LENGTH  */
    CHAR_AT = 281,                 /* CHAR_AT  */
    BREAK = 282,                   /* BREAK  */
    CONTINUE = 283,                /* CONTINUE  */
    MOD = 284,                     /* MOD  */
    POW = 285,                     /* POW  */
    ABS = 286,                     /* ABS  */
    MIN = 287,                     /* MIN  */
    MAX = 288,                     /* MAX  */
    SQRT = 289,                    /* SQRT  */
    COLON = 290,                   /* COLON  */
    COMMA = 291,                   /* COMMA  */
    WILDCARD = 292,                /* WILDCARD  */
    PLUS = 293,                    /* PLUS  */
    MINUS = 294,                   /* MINUS  */
    STAR = 295,                    /* STAR  */
    SLASH = 296,                   /* SLASH  */
    LPAREN = 297,                  /* LPAREN  */
    RPAREN = 298,                  /* RPAREN  */
    ASSIGN = 299,                  /* ASSIGN  */
    GT = 300,                      /* GT  */
    LT = 301,                      /* LT  */
    GE = 302,                      /* GE  */
    LE = 303,                      /* LE  */
    EQ = 304,                      /* EQ  */
    NE = 305,                      /* NE  */
    INTEGER = 306,                 /* INTEGER  */
    VAR = 307,                     /* VAR  */
    FLOAT = 308,                   /* FLOAT  */
    IDENT = 309,                   /* IDENT  */
    STRING = 310,                  /* STRING  */
    UMINUS = 311                   /* UMINUS  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 211 "bytelog.y"

    int ival;
    double fval;
    char *sval;
    struct ASTNode *node;
    int op;

#line 128 "bytelog.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_BYTELOG_TAB_H_INCLUDED  */
