/* ═══════════════════════════════════════════════════════════════════════════
 * demo.c - ByteLog Compiler Demo
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Demonstrates parsing a ByteLog program and displaying the AST.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "parser.h"
#include "ast.h"
#include "engine.h"
#include "wat_gen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS] <file.bl>\n", program_name);
    printf("ByteLog interpreter, analyzer, and compiler\n\n");
    printf("Options:\n");
    printf("  -v, --verbose         Show detailed parsing and execution information\n");
    printf("  -c, --compile=FORMAT  Compile to target format (wat|wasm)\n");
    printf("  -o, --output=FILE     Output file (default: input.{wat|wasm}, use '-' for stdout)\n");
    printf("  -h, --help            Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s program.bl                 # Run program, show results\n", program_name);
    printf("  %s -v program.bl              # Run with detailed output\n", program_name);
    printf("  %s --compile=wat program.bl   # Compile to WebAssembly Text\n", program_name);
    printf("  %s --compile=wasm program.bl  # Compile to WASM binary\n", program_name);
    printf("  %s -c wat -o - program.bl     # Output WAT to stdout\n", program_name);
    printf("  %s -c wat -o output.wat prog.bl # Custom output file\n", program_name);
}

static char* get_default_output_filename(const char *input_filename, const char *extension) {
    if (!input_filename) return NULL;
    
    size_t input_len = strlen(input_filename);
    size_t ext_len = strlen(extension);
    char *output = malloc(input_len + ext_len + 2); // +2 for '.' and null terminator
    if (!output) return NULL;
    
    strcpy(output, input_filename);
    
    /* Replace .bl extension or append new extension */
    char *dot = strrchr(output, '.');
    if (dot && strcmp(dot, ".bl") == 0) {
        strcpy(dot, ".");
        strcat(dot, extension);
    } else {
        strcat(output, ".");
        strcat(output, extension);
    }
    
    return output;
}

static int compile_to_wat(const char *input_file, const char *output_file, bool verbose) {
    char error_buf[512];
    bool success;
    
    /* Handle stdout output */
    if (output_file && strcmp(output_file, "-") == 0) {
        /* For stdout, we need to read the file first, then generate to stdout */
        FILE *input = fopen(input_file, "r");
        if (!input) {
            snprintf(error_buf, 512, "Cannot open input file: %s", input_file);
            if (verbose) {
                printf("❌ WAT compilation failed: %s\n", error_buf);
            } else {
                fprintf(stderr, "Error: %s\n", error_buf);
            }
            return 1;
        }
        
        /* Read entire file into memory */
        fseek(input, 0, SEEK_END);
        long file_size = ftell(input);
        fseek(input, 0, SEEK_SET);
        
        char *source = malloc(file_size + 1);
        if (!source) {
            fclose(input);
            snprintf(error_buf, 512, "Out of memory");
            if (verbose) {
                printf("❌ WAT compilation failed: %s\n", error_buf);
            }
            return 1;
        }
        
        fread(source, 1, file_size, input);
        source[file_size] = '\0';
        fclose(input);
        
        success = generate_wat_string(source, stdout, error_buf, sizeof(error_buf));
        free(source);
        
        if (!success) {
            if (verbose) {
                fprintf(stderr, "❌ WAT compilation failed: %s\n", error_buf);
            } else {
                fprintf(stderr, "Error: %s\n", error_buf);
            }
        }
        return success ? 0 : 1;
    }
    
    /* Handle file output */
    success = generate_wat_file(input_file, output_file, error_buf, sizeof(error_buf));
    
    if (success) {
        if (verbose) {
            printf("✅ WAT compilation successful!\n");
            printf("Generated: %s\n", output_file);
        } else {
            printf("Generated %s\n", output_file);
        }
        return 0;
    } else {
        if (verbose) {
            printf("❌ WAT compilation failed: %s\n", error_buf);
        } else {
            fprintf(stderr, "Compilation error: %s\n", error_buf);
        }
        return 1;
    }
}

static int compile_to_wasm(const char *input_file, const char *output_file, bool verbose) {
    /* First compile to WAT */
    char *wat_file = get_default_output_filename(input_file, "wat");
    if (!wat_file) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }
    
    /* Generate WAT file in temp location */
    char error_buf[512];
    bool success = generate_wat_file(input_file, wat_file, error_buf, sizeof(error_buf));
    
    if (!success) {
        if (verbose) {
            printf("❌ WAT generation failed: %s\n", error_buf);
        } else {
            fprintf(stderr, "WAT generation error: %s\n", error_buf);
        }
        free(wat_file);
        return 1;
    }
    
    /* Now compile WAT to WASM using wat2wasm */
    char command[1024];
    snprintf(command, sizeof(command), "wat2wasm \"%s\" -o \"%s\"", wat_file, output_file);
    
    int result = system(command);
    
    /* Clean up temporary WAT file */
    remove(wat_file);
    free(wat_file);
    
    if (result == 0) {
        if (verbose) {
            printf("✅ WASM compilation successful!\n");
            printf("Generated: %s\n", output_file);
        } else {
            printf("Generated %s\n", output_file);
        }
        return 0;
    } else {
        if (verbose) {
            printf("❌ WASM compilation failed (wat2wasm not available or failed)\n");
        } else {
            fprintf(stderr, "WASM compilation failed: wat2wasm not available or failed\n");
        }
        return 1;
    }
}

typedef enum {
    MODE_INTERPRET,
    MODE_COMPILE_WAT,
    MODE_COMPILE_WASM
} ExecutionMode;

int main(int argc, char **argv) {
    const char *filename = NULL;
    const char *output_file = NULL;
    bool verbose = false;
    ExecutionMode mode = MODE_INTERPRET;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strncmp(argv[i], "--compile=", 10) == 0) {
            const char *format = argv[i] + 10;
            if (strcmp(format, "wat") == 0) {
                mode = MODE_COMPILE_WAT;
            } else if (strcmp(format, "wasm") == 0) {
                mode = MODE_COMPILE_WASM;
            } else {
                fprintf(stderr, "Unknown compile format: %s\n", format);
                fprintf(stderr, "Supported formats: wat, wasm\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Option -c requires a format argument\n");
                return 1;
            }
            const char *format = argv[++i];
            if (strcmp(format, "wat") == 0) {
                mode = MODE_COMPILE_WAT;
            } else if (strcmp(format, "wasm") == 0) {
                mode = MODE_COMPILE_WASM;
            } else {
                fprintf(stderr, "Unknown compile format: %s\n", format);
                fprintf(stderr, "Supported formats: wat, wasm\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strncmp(argv[i], "--output=", 9) == 0) {
            if (strncmp(argv[i], "--output=", 9) == 0) {
                output_file = argv[i] + 9;
            } else {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Option -o requires a filename argument\n");
                    return 1;
                }
                output_file = argv[++i];
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            filename = argv[i];
        }
    }
    
    if (!filename) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* Handle compilation modes */
    if (mode == MODE_COMPILE_WAT || mode == MODE_COMPILE_WASM) {
        /* Determine output filename if not specified */
        char *allocated_output = NULL;
        if (!output_file) {
            const char *ext = (mode == MODE_COMPILE_WAT) ? "wat" : "wasm";
            allocated_output = get_default_output_filename(filename, ext);
            output_file = allocated_output;
            
            if (!output_file) {
                fprintf(stderr, "Error: Out of memory\n");
                return 1;
            }
        }
        
        if (verbose) {
            printf("ByteLog Compiler\n");
            printf("═══════════════════════════════════════\n\n");
            printf("Input:  %s\n", filename);
            printf("Output: %s\n", output_file);
            printf("Mode:   %s\n\n", (mode == MODE_COMPILE_WAT) ? "WebAssembly Text" : "WebAssembly Binary");
        }
        
        /* Perform compilation */
        int result;
        if (mode == MODE_COMPILE_WAT) {
            result = compile_to_wat(filename, output_file, verbose);
        } else {
            result = compile_to_wasm(filename, output_file, verbose);
        }
        
        if (allocated_output) {
            free(allocated_output);
        }
        
        return result;
    }
    
    /* Interpreter mode */
    if (verbose) {
        printf("ByteLog Interpreter\n");
        printf("═══════════════════════════════════════\n\n");
        printf("Parsing file: %s\n\n", filename);
    }
    
    /* Parse the ByteLog file */
    char error_buf[512];
    ASTNode *ast = parse_file(filename, error_buf, sizeof(error_buf));
    
    if (!ast) {
        if (verbose) {
            printf("❌ Parse failed: %s\n", error_buf);
        } else {
            fprintf(stderr, "Parse error: %s\n", error_buf);
        }
        return 1;
    }
    
    if (verbose) {
        printf("✅ Parse successful!\n\n");
        
        /* Display the Abstract Syntax Tree */
        printf("Abstract Syntax Tree:\n");
        printf("─────────────────────────────────────────\n");
        ast_print_tree(ast);
        
        printf("\nAnalysis:\n");
        printf("─────────────────────────────────────────\n");
    }
    
    /* Count different types of statements */
    int rel_count = 0, fact_count = 0, rule_count = 0;
    int solve_count = 0, query_count = 0;
    
    ASTNode *stmt = ast->data.program.statements;
    while (stmt) {
        switch (stmt->type) {
            case AST_REL_DECL: rel_count++; break;
            case AST_FACT: fact_count++; break;
            case AST_RULE: rule_count++; break;
            case AST_SOLVE: solve_count++; break;
            case AST_QUERY: query_count++; break;
            default: break;
        }
        stmt = stmt->next;
    }
    
    if (verbose) {
        printf("Relations declared: %d\n", rel_count);
        printf("Facts asserted: %d\n", fact_count);
        printf("Rules defined: %d\n", rule_count);
        printf("Solve statements: %d\n", solve_count);
        printf("Queries: %d\n", query_count);
        
        /* Show what the program does */
        printf("\nProgram Logic:\n");
        printf("─────────────────────────────────────────\n");
    }
    
    if (verbose) {
        stmt = ast->data.program.statements;
        while (stmt) {
            switch (stmt->type) {
                case AST_REL_DECL:
                    printf("• Declares relation '%s'\n", stmt->data.rel_decl.name);
                    break;
                    
                case AST_FACT:
                    printf("• Asserts fact: %s(%d, %d)\n", 
                           stmt->data.fact.relation, 
                           stmt->data.fact.a, 
                           stmt->data.fact.b);
                    break;
                    
                case AST_RULE:
                    printf("• Defines rule for '%s'\n", stmt->data.rule.target);
                    break;
                    
                case AST_SOLVE:
                    printf("• Computes fixpoint (derives all facts)\n");
                    break;
                    
                case AST_QUERY:
                    if (stmt->data.query.arg_a != -1 && stmt->data.query.arg_b != -1) {
                        printf("• Queries: Is %s(%d, %d) true?\n",
                               stmt->data.query.relation,
                               stmt->data.query.arg_a, 
                               stmt->data.query.arg_b);
                    } else if (stmt->data.query.arg_a != -1) {
                        printf("• Queries: All Y where %s(%d, Y)\n",
                               stmt->data.query.relation,
                               stmt->data.query.arg_a);
                    } else if (stmt->data.query.arg_b != -1) {
                        printf("• Queries: All X where %s(X, %d)\n",
                               stmt->data.query.relation,
                               stmt->data.query.arg_b);
                    } else {
                        printf("• Queries: All facts in %s\n",
                               stmt->data.query.relation);
                    }
                    break;
                    
                default:
                    break;
            }
            stmt = stmt->next;
        }
        
        /* Execute the program */
        printf("\nExecution:\n");
        printf("─────────────────────────────────────────\n");
    }
    
    ExecutionEngine *engine = malloc(sizeof(ExecutionEngine));
    if (!engine) {
        printf("❌ Out of memory\n");
        ast_free_tree(ast);
        return 1;
    }
    
    engine_init(engine);
    engine_set_debug(engine, false);  /* Set to true for detailed execution trace */
    
    if (!engine_execute_program(engine, ast)) {
        if (verbose) {
            printf("❌ Execution failed: %s\n", engine_get_error(engine));
        } else {
            fprintf(stderr, "Execution error: %s\n", engine_get_error(engine));
        }
        engine_cleanup(engine);
        free(engine);
        ast_free_tree(ast);
        return 1;
    }
    
    if (verbose) {
        printf("✅ Execution successful!\n\n");
        
        /* Show the derived facts */
        printf("Derived Facts:\n");
        printf("─────────────────────────────────────────\n");
        factdb_print(&engine->facts, &engine->atoms);
        
        /* Answer all queries in the program */
        printf("\nQuery Results:\n");
        printf("─────────────────────────────────────────\n");
    }
    
    stmt = ast->data.program.statements;
    int query_num = 1;
    while (stmt) {
        if (stmt->type == AST_QUERY) {
            if (verbose) {
                printf("Query %d: ", query_num++);
                
                /* Print the query */
                if (stmt->data.query.atom_a && stmt->data.query.atom_b) {
                    printf("%s(%s, %s)\n", stmt->data.query.relation,
                           stmt->data.query.arg_a == -1 ? "?" : stmt->data.query.atom_a,
                           stmt->data.query.arg_b == -1 ? "?" : stmt->data.query.atom_b);
                } else if (stmt->data.query.atom_a) {
                    printf("%s(%s, %s)\n", stmt->data.query.relation,
                           stmt->data.query.atom_a,
                           stmt->data.query.arg_b == -1 ? "?" : "?");
                } else if (stmt->data.query.atom_b) {
                    printf("%s(%s, %s)\n", stmt->data.query.relation,
                           stmt->data.query.arg_a == -1 ? "?" : "?",
                           stmt->data.query.atom_b);
                } else {
                    printf("%s(?, ?)\n", stmt->data.query.relation);
                }
            }
            
            /* Execute the query */
            QueryResult *results = engine_query(engine, stmt);
            if (results) {
                query_result_print(results, stmt->data.query.relation, &engine->atoms);
                query_result_free(results);
            } else if (verbose) {
                printf("  No results found.\n");
            }
            if (verbose) printf("\n");
        }
        stmt = stmt->next;
    }
    
    if (verbose) {
        printf("🎯 ByteLog program executed successfully!\n");
    }
    
    /* Clean up */
    engine_cleanup(engine);
    free(engine);
    ast_free_tree(ast);
    
    return 0;
}