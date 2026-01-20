/* ═══════════════════════════════════════════════════════════════════════════
 * wat_compiler.c - ByteLog to WAT Compiler
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Command-line tool to compile ByteLog programs to WebAssembly Text (WAT) format.
 * The generated WAT can be converted to WASM and run in web browsers or WASM runtimes.
 *
 * Usage: wat_compiler input.bl [output.wat]
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "wat_gen.h"
#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program_name) {
    printf("ByteLog to WAT Compiler\n");
    printf("═══════════════════════════════════════\n\n");
    printf("Usage: %s input.bl [output.wat]\n\n", program_name);
    printf("Compiles ByteLog programs to WebAssembly Text (WAT) format.\n\n");
    printf("Arguments:\n");
    printf("  input.bl    - Input ByteLog source file\n");
    printf("  output.wat  - Output WAT file (optional, defaults to input.wat)\n\n");
    printf("Examples:\n");
    printf("  %s example.bl               # Creates example.wat\n", program_name);
    printf("  %s example.bl output.wat    # Creates output.wat\n", program_name);
    printf("\nThe generated WAT file can be compiled to WASM using:\n");
    printf("  wat2wasm output.wat -o output.wasm\n");
    printf("  wasmtime output.wasm\n");
}

static char* get_output_filename(const char *input_filename) {
    if (!input_filename) return NULL;
    
    size_t len = strlen(input_filename);
    char *output = malloc(len + 5);  /* Room for ".wat" + null terminator */
    if (!output) return NULL;
    
    strcpy(output, input_filename);
    
    /* Replace .bl extension with .wat */
    char *ext = strrchr(output, '.');
    if (ext && strcmp(ext, ".bl") == 0) {
        strcpy(ext, ".wat");
    } else {
        /* No .bl extension, just append .wat */
        strcat(output, ".wat");
    }
    
    return output;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = NULL;
    char *allocated_output = NULL;
    
    if (argc >= 3) {
        output_filename = argv[2];
    } else {
        allocated_output = get_output_filename(input_filename);
        output_filename = allocated_output;
        
        if (!output_filename) {
            fprintf(stderr, "❌ Error: Out of memory\n");
            return 1;
        }
    }
    
    printf("ByteLog to WAT Compiler\n");
    printf("═══════════════════════════════════════\n\n");
    printf("Input:  %s\n", input_filename);
    printf("Output: %s\n\n", output_filename);
    
    /* Compile ByteLog to WAT */
    char error_buf[512];
    bool success = generate_wat_file(input_filename, output_filename, 
                                    error_buf, sizeof(error_buf));
    
    if (success) {
        printf("✅ Compilation successful!\n\n");
        printf("Generated WAT file: %s\n\n", output_filename);
        printf("To compile to WASM and run:\n");
        printf("  wat2wasm %s -o %s\n", output_filename, 
               strrchr(output_filename, '.') ? 
               (snprintf(error_buf, sizeof(error_buf), "%.*s.wasm", 
                        (int)(strrchr(output_filename, '.') - output_filename), 
                        output_filename), error_buf) : "output.wasm");
        printf("  wasmtime %s\n", 
               strrchr(output_filename, '.') ? 
               (snprintf(error_buf, sizeof(error_buf), "%.*s.wasm", 
                        (int)(strrchr(output_filename, '.') - output_filename), 
                        output_filename), error_buf) : "output.wasm");
    } else {
        printf("❌ Compilation failed: %s\n", error_buf);
        if (allocated_output) {
            free(allocated_output);
        }
        return 1;
    }
    
    if (allocated_output) {
        free(allocated_output);
    }
    
    return 0;
}