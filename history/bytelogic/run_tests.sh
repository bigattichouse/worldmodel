#!/bin/bash

# ByteLog Comprehensive Test Suite
# Tests parsing, execution, WAT compilation, and WASM execution

# Define the executable name
BYTELOGIC="./build/bytelogic"

echo "🧪 ByteLog Comprehensive Test Suite"
echo "═══════════════════════════════════════════════"

PASS_COUNT=0
FAIL_COUNT=0

test_result() {
    if [ $1 -eq 0 ]; then
        echo "✅ PASS: $2"
        ((PASS_COUNT++))
    else
        echo "❌ FAIL: $2"
        ((FAIL_COUNT++))
    fi
}

# Test 1: Build system
echo -e "\n📦 Testing build system..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1
test_result $? "Build compilation"

# Test 2: Basic parsing and execution
echo -e "\n🔍 Testing basic parsing and execution..."
$BYTELOGIC examples/example_family.bl > /dev/null 2>&1
test_result $? "Family example parsing and execution"

# Test 2b: Test minimal output format
echo -e "\n📝 Testing minimal output format..."
OUTPUT=$($BYTELOGIC examples/example_family.bl 2>&1)
if echo "$OUTPUT" | grep -q "parent(alice, bob)" && ! echo "$OUTPUT" | grep -q "✅" && ! echo "$OUTPUT" | grep -q "═══"; then
    echo "✅ PASS: Minimal output format working"
    ((PASS_COUNT++))
else
    echo "❌ FAIL: Minimal output format not working"
    ((FAIL_COUNT++))
fi

# Test 2c: Test verbose output format
echo -e "\n🔍 Testing verbose output format..."
VERBOSE_OUTPUT=$($BYTELOGIC -v examples/example_family.bl 2>&1)
if echo "$VERBOSE_OUTPUT" | grep -q "✅ Parse successful!" && echo "$VERBOSE_OUTPUT" | grep -q "Abstract Syntax Tree:" && echo "$VERBOSE_OUTPUT" | grep -q "parent(alice, bob)"; then
    echo "✅ PASS: Verbose output format working"
    ((PASS_COUNT++))
else
    echo "❌ FAIL: Verbose output format not working"
    ((FAIL_COUNT++))
fi

# Test 3: WAT compilation
echo -e "\n🔧 Testing WAT compilation..."
./build/wat_compiler examples/example_family.bl > /dev/null 2>&1
test_result $? "WAT compilation of family example"

# Test 4: WAT to WASM compilation
echo -e "\n⚙️  Testing WAT to WASM compilation..."
if command -v wat2wasm > /dev/null; then
    wat2wasm examples/example_family.wat -o examples/example_family.wasm > /dev/null 2>&1
    test_result $? "WAT to WASM binary compilation"
else
    echo "⚠️  SKIP: wat2wasm not available"
fi

# Test 5: WASM execution
echo -e "\n🚀 Testing WASM execution..."
if command -v node > /dev/null && [ -f examples/example_family.wasm ]; then
    node test_wasm.js > /dev/null 2>&1
    test_result $? "WASM module execution with Node.js"
else
    echo "⚠️  SKIP: Node.js not available or WASM file missing"
fi

# Test 6: Logic puzzle example
echo -e "\n🧩 Testing logic puzzle..."
$BYTELOGIC examples/logic_puzzle_simple.bl > /dev/null 2>&1
test_result $? "Logic puzzle parsing and execution"

# Test 7: WAT compilation of logic puzzle
echo -e "\n🔧 Testing WAT compilation of logic puzzle..."
./build/wat_compiler examples/logic_puzzle_simple.bl > /dev/null 2>&1
test_result $? "WAT compilation of logic puzzle"

# Test 8: Error handling - invalid input
echo -e "\n🚫 Testing error handling..."
echo "INVALID SYNTAX" | $BYTELOGIC /dev/stdin > /dev/null 2>&1
# Error handling should return 1 (failure) but not crash
if [ $? -eq 1 ]; then
    echo "✅ PASS: Error handling for invalid syntax"
    ((PASS_COUNT++))
else
    echo "❌ FAIL: Error handling for invalid syntax"
    ((FAIL_COUNT++))
fi

# Test 9: Memory management - large program
echo -e "\n💾 Testing memory management..."
# Create a larger test program
cat > /tmp/large_test.bl << EOF
REL test
$(for i in {1..100}; do echo "FACT test $i $((i+100))"; done)
SOLVE
$(for i in {1..50}; do echo "QUERY test $i ?"; done)
EOF

$BYTELOGIC /tmp/large_test.bl > /dev/null 2>&1
test_result $? "Large program execution (memory stress test)"

# Test 10: Multiple relation types
echo -e "\n🔗 Testing multiple relation types..."
cat > /tmp/multi_rel_test.bl << EOF
REL likes
REL hates  
REL knows

FACT likes alice bob
FACT hates bob charlie
FACT knows alice charlie

RULE friends: SCAN likes MATCH \$0, EMIT friends \$1 \$2

SOLVE

QUERY likes alice ?
QUERY friends alice ?
EOF

$BYTELOGIC /tmp/multi_rel_test.bl > /dev/null 2>&1
test_result $? "Multiple relation types"

# Summary
echo -e "\n📊 Test Results Summary"
echo "═══════════════════════════════════════════════"
echo "✅ Passed: $PASS_COUNT"
echo "❌ Failed: $FAIL_COUNT"
echo "📈 Success Rate: $(( PASS_COUNT * 100 / (PASS_COUNT + FAIL_COUNT) ))%"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n🎉 All tests passed! ByteLog is fully functional."
    exit 0
else
    echo -e "\n⚠️  Some tests failed. Please review the results above."
    exit 1
fi