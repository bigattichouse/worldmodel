#!/usr/bin/env python3
"""
Generate training examples for the tool-request protocol.

These examples teach the model to:
1. Call use_tool(name, **kwargs) when it needs an external capability
2. Catch ToolNotAvailableError gracefully when the tool isn't registered
3. Explain what tool it needs and why
4. Offer a useful fallback (simulation, approximation, or partial answer)

This trains three important behaviours:
  - Knowing WHEN to reach for a tool (don't brute-force in Python what
    a real tool would do better)
  - Expressing the need clearly so the runtime/user can wire it up later
  - Degrading gracefully rather than silently failing or hallucinating

Usage:
    python training/scripts/generate_tool_requests.py
    python training/scripts/generate_tool_requests.py --output training/datasets/tool_requests/basic.jsonl
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import PythonExecutor


def run_code(executor: PythonExecutor, code: str) -> str:
    result = executor.run(code)
    return result.output_text().strip()


# ---------------------------------------------------------------------------
# Helper: run code and expect ToolNotAvailableError to surface cleanly
# ---------------------------------------------------------------------------

def tool_unavailable_output(tool_name: str) -> str:
    """Produce the exact output the executor emits when a tool isn't registered."""
    executor = PythonExecutor()
    code = f"""\
try:
    result = use_tool("{tool_name}", query="test")
except ToolNotAvailableError as e:
    print(e)
"""
    return run_code(executor, code)


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------

def make_web_search_unavailable() -> dict:
    executor = PythonExecutor()

    code1 = """\
# I need to find today's exchange rate for USD→EUR to complete this conversion.
try:
    rate_data = use_tool("web_search", query="USD EUR exchange rate today")
    print(rate_data)
except ToolNotAvailableError as e:
    print(f"Tool unavailable: {e}")
    print("\\nFalling back to a recent approximate rate.")
    fallback_rate = 0.92  # approximate, may be stale
    print(f"Using fallback rate: 1 USD = {fallback_rate} EUR")
    print("WARNING: This may not reflect the current market rate.")
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Complete the conversion with the fallback rate
amount_usd = 1500
converted = amount_usd * fallback_rate
print(f"${amount_usd} USD ≈ €{converted:.2f} EUR (approximate)")
print("For an accurate result, register the 'web_search' tool in this runtime.")
"""
    output2 = run_code(executor, code2)

    return {
        "id": "tool_req_001",
        "category": "tool_requests",
        "difficulty": "basic",
        "query": "Convert $1,500 USD to EUR using the current exchange rate.",
        "response": (
            "<think>\n"
            "I need a live exchange rate. I'll try use_tool('web_search') for this.\n"
            "If it's not available I'll use an approximate fallback and be transparent about it.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "web_search isn't registered. I have a fallback rate in scope. "
            "I'll complete the conversion and tell the user the result may be stale.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "The conversion used an approximate rate. To get a live result, the runtime "
            "needs a `web_search` tool registered — for example a function that calls a "
            "currency API and returns the rate."
        ),
    }


def make_database_query_unavailable() -> dict:
    executor = PythonExecutor()

    code1 = """\
# Query sales data from the company database
try:
    rows = use_tool("sql_query",
                    query="SELECT product, SUM(revenue) FROM sales WHERE year=2025 GROUP BY product ORDER BY 2 DESC LIMIT 5")
    print("Top 5 products by 2025 revenue:")
    for row in rows:
        print(f"  {row['product']}: ${row['revenue']:,.0f}")
except ToolNotAvailableError as e:
    print(f"Tool not available: {e}")
    print("\\nI need the 'sql_query' tool to answer this question accurately.")
    print("Without it I can show you the query I would run and what the output would look like:")
    print()
    print("Query:")
    print("  SELECT product, SUM(revenue)")
    print("  FROM sales")
    print("  WHERE year = 2025")
    print("  GROUP BY product")
    print("  ORDER BY SUM(revenue) DESC")
    print("  LIMIT 5")
    print()
    print("Expected output structure:")
    print("  product        | revenue")
    print("  ---------------|----------")
    print("  <product name> | <amount>")
"""
    output1 = run_code(executor, code1)

    return {
        "id": "tool_req_002",
        "category": "tool_requests",
        "difficulty": "basic",
        "query": (
            "What were the top 5 products by revenue in 2025? "
            "Our data is in a SQL database with a 'sales' table."
        ),
        "response": (
            "<think>\n"
            "This requires a database query. I'll try use_tool('sql_query').\n"
            "If unavailable I'll show the exact query and explain what I need.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "To answer this question I need the `sql_query` tool registered in the runtime. "
            "The query above is ready to run — once `sql_query` is available, the code "
            "block will execute it and format the results automatically."
        ),
    }


def make_file_read_unavailable() -> dict:
    executor = PythonExecutor()

    code1 = """\
# Try to read a CSV from the filesystem
try:
    csv_content = use_tool("read_file", path="/data/measurements.csv")
    # parse and analyse
    import io
    lines = csv_content.strip().split("\\n")
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    values = [float(r["value"]) for r in rows]
    print(f"Loaded {len(values)} measurements")
    print(f"Mean: {sum(values)/len(values):.3f}")
    print(f"Min:  {min(values):.3f}, Max: {max(values):.3f}")
except ToolNotAvailableError as e:
    print(f"Tool not available: {e}")
    print("\\nI need the 'read_file' tool to load your CSV.")
    print("You can register it with:")
    print("  executor.register_tool('read_file', lambda path: open(path).read())")
    print("\\nAlternatively, paste the CSV content directly in your query and I can analyse it.")
except Exception as e:
    print(f"Error processing file: {e}")
"""
    output1 = run_code(executor, code1)

    return {
        "id": "tool_req_003",
        "category": "tool_requests",
        "difficulty": "basic",
        "query": "Analyse the measurements in /data/measurements.csv — compute mean, min, max.",
        "response": (
            "<think>\n"
            "I need to read a file from disk. I'll try use_tool('read_file').\n"
            "If it fails I'll tell the user exactly what to register and offer an alternative.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "The `read_file` tool isn't registered in this runtime. Either register it as "
            "shown above, or paste the CSV content directly into your message and I'll "
            "analyse it right away."
        ),
    }


def make_tool_available_and_used() -> dict:
    """Positive example: tool IS available and the model uses it correctly."""
    executor = PythonExecutor()

    # Register a fake web_search tool
    search_results = {
        "Python list comprehension tutorial": [
            {"title": "Python List Comprehensions", "snippet": "A list comprehension is a concise way to create a list: [expr for item in iterable if condition]"},
            {"title": "Real Python - List Comprehensions", "snippet": "List comprehensions provide a more compact syntax for creating a new list based on the values of an existing list."},
        ]
    }
    def fake_search(query):
        results = search_results.get(query, [{"title": "No results", "snippet": ""}])
        return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)

    executor.register_tool("web_search", fake_search)

    code1 = """\
# Search for information about the topic
results = use_tool("web_search", query="Python list comprehension tutorial")
print("Search results:")
print(results)
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Now demonstrate based on the search results
print("List comprehension syntax:")
print("  [expression for item in iterable if condition]")
print()

# Basic example
squares = [x**2 for x in range(1, 6)]
print(f"Squares 1-5: {squares}")

# With filter
evens = [x for x in range(1, 11) if x % 2 == 0]
print(f"Even numbers 1-10: {evens}")

# Nested
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(f"3x3 multiplication table: {matrix}")
"""
    output2 = run_code(executor, code2)

    return {
        "id": "tool_req_004",
        "category": "tool_requests",
        "difficulty": "basic",
        "query": (
            "I want to learn about Python list comprehensions. "
            "Can you look up a quick reference and then show me examples?"
        ),
        "response": (
            "<think>\n"
            "I can search for reference material, then demonstrate with runnable examples.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Good — the search returned useful info. Now I'll demonstrate the syntax "
            "with concrete executable examples.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "List comprehensions are a compact, readable way to build lists. "
            "The pattern is always `[expression for item in iterable if condition]` — "
            "the `if` part is optional."
        ),
    }


def make_tool_partial_fallback() -> dict:
    """Model uses tool for part of the answer and Python for the rest."""
    executor = PythonExecutor()

    code1 = """\
import math

# Try to get real stock data; fall back to demonstration with synthetic data
try:
    price_history = use_tool("stock_data", ticker="AAPL", days=30)
except ToolNotAvailableError:
    print("Note: 'stock_data' tool not available. Using synthetic price series for demonstration.")
    import random
    random.seed(42)
    price = 185.0
    price_history = []
    for _ in range(30):
        price *= (1 + random.gauss(0.0005, 0.015))
        price_history.append(round(price, 2))
    print(f"Synthetic prices (first 5): {price_history[:5]}")

# Compute statistics regardless of data source
n = len(price_history)
mean_price = sum(price_history) / n
returns = [(price_history[i] - price_history[i-1]) / price_history[i-1]
           for i in range(1, n)]
mean_ret = sum(returns) / len(returns)
vol = math.sqrt(sum((r - mean_ret)**2 for r in returns) / (len(returns)-1))
annualised_vol = vol * math.sqrt(252)

print(f"\\nPrice statistics over {n} days:")
print(f"  Mean price:        ${mean_price:.2f}")
print(f"  Start / End:       ${price_history[0]:.2f} / ${price_history[-1]:.2f}")
print(f"  Daily volatility:  {vol*100:.2f}%")
print(f"  Annualised vol:    {annualised_vol*100:.1f}%")
"""
    output1 = run_code(executor, code1)

    return {
        "id": "tool_req_005",
        "category": "tool_requests",
        "difficulty": "intermediate",
        "query": (
            "Analyse 30 days of AAPL stock prices: compute mean price, daily returns, "
            "and annualised volatility."
        ),
        "response": (
            "<think>\n"
            "I'll try use_tool('stock_data') for live prices. If unavailable I'll generate "
            "a synthetic but realistic price series so I can still demonstrate the full "
            "analysis — the statistics code is the same either way.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "The analysis ran on synthetic data since `stock_data` isn't registered. "
            "To use live prices, register a tool that returns a list of closing prices. "
            "The statistics calculation is identical regardless of data source."
        ),
    }


def make_multi_tool_workflow() -> dict:
    """Model chains two tool calls, handles one missing gracefully."""
    executor = PythonExecutor()

    # Register only the geocode tool, not the weather tool
    def fake_geocode(city):
        coords = {
            "London": (51.5074, -0.1278),
            "Paris": (48.8566, 2.3522),
            "Tokyo": (35.6762, 139.6503),
        }
        lat, lon = coords.get(city, (0, 0))
        return {"lat": lat, "lon": lon, "city": city}

    executor.register_tool("geocode", fake_geocode)

    code1 = """\
city = "Tokyo"

# Step 1: get coordinates
location = use_tool("geocode", city=city)
print(f"Location: {location['city']} at ({location['lat']:.4f}, {location['lon']:.4f})")
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Step 2: get current weather at those coordinates
try:
    weather = use_tool("weather", lat=location['lat'], lon=location['lon'])
    print(f"Current weather: {weather}")
except ToolNotAvailableError as e:
    print(f"Weather tool not available: {e}")
    print(f"\\nI have the coordinates ({location['lat']:.4f}, {location['lon']:.4f}).")
    print("To complete this query, register a 'weather' tool that accepts lat/lon")
    print("and returns current conditions (e.g. from OpenWeatherMap API).")
    print()
    # Still provide what I can: timezone estimate from longitude
    import math
    tz_offset = round(location['lon'] / 15)
    print(f"Estimated timezone: UTC{tz_offset:+d} (based on longitude)")
"""
    output2 = run_code(executor, code2)

    return {
        "id": "tool_req_006",
        "category": "tool_requests",
        "difficulty": "intermediate",
        "query": "What is the current weather in Tokyo?",
        "response": (
            "<think>\n"
            "To get weather I first need coordinates (geocode), then a weather API call.\n"
            "I'll try both tools and handle unavailability gracefully at each step.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Geocode worked. Now try the weather tool — it may not be registered.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "I have Tokyo's coordinates but the `weather` tool isn't registered. "
            "Once it's available (e.g. wrapped around OpenWeatherMap), the code will "
            "return live conditions automatically."
        ),
    }


def make_llm_tool_call() -> dict:
    """Model uses use_tool('llm') to call another model for a subtask."""
    executor = PythonExecutor()

    code1 = """\
document = \"\"\"
The new climate agreement signed in Geneva commits 47 nations to reducing
methane emissions by 35% by 2030. Key provisions include mandatory reporting,
a carbon credit trading mechanism for developing nations, and a $50B fund for
renewable energy transition. Critics argue enforcement mechanisms are weak.
\"\"\"

# Try to use an LLM tool for summarisation
try:
    summary = use_tool("llm", prompt=f"Summarise in one sentence: {document.strip()}")
    print(f"LLM summary: {summary}")
except ToolNotAvailableError as e:
    print(f"Note: {e}")
    print("Generating extractive summary from key sentences instead:")
    # Fallback: pick sentences containing key terms
    key_terms = {"commits", "provisions", "reduction", "fund", "critics"}
    sentences = [s.strip() for s in document.replace("\\n", " ").split(".") if s.strip()]
    key_sentences = [s for s in sentences
                     if any(t in s.lower() for t in key_terms)][:2]
    for s in key_sentences:
        print(f"  • {s}.")
"""
    output1 = run_code(executor, code1)

    return {
        "id": "tool_req_007",
        "category": "tool_requests",
        "difficulty": "intermediate",
        "query": (
            "Summarise this document in one sentence:\n\n"
            "The new climate agreement signed in Geneva commits 47 nations to reducing "
            "methane emissions by 35% by 2030. Key provisions include mandatory reporting, "
            "a carbon credit trading mechanism for developing nations, and a $50B fund for "
            "renewable energy transition. Critics argue enforcement mechanisms are weak."
        ),
        "response": (
            "<think>\n"
            "I can try use_tool('llm') to delegate the summarisation to a language model.\n"
            "If unavailable, I'll do an extractive summary by pulling key sentences.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "The `llm` tool isn't registered so I fell back to extractive summarisation. "
            "To enable LLM-assisted subtasks, register a tool that calls any inference "
            "endpoint and returns the generated text."
        ),
    }


def make_tool_discovery() -> dict:
    """Model discovers what tools are available before deciding approach."""
    executor = PythonExecutor()
    executor.register_tool("calculator", lambda expr: str(eval(expr)))

    code1 = """\
# Probe which tools are available before choosing an approach
available = []
for tool_name in ["web_search", "calculator", "sql_query", "read_file", "llm"]:
    try:
        # Use a harmless test call
        if tool_name == "calculator":
            use_tool(tool_name, expr="1+1")
        else:
            use_tool(tool_name, query="test")
        available.append(tool_name)
    except ToolNotAvailableError:
        pass  # not available
    except Exception:
        available.append(tool_name)  # available but our test call failed

print(f"Available tools: {available if available else '(none)'}")
print("Planning approach based on available tools...")
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Use what's available: only 'calculator' is registered here
if "calculator" in available:
    result = use_tool("calculator", expr="137 * 429 + 88")
    print(f"137 × 429 + 88 = {result}")
    print("Used: calculator tool")
else:
    # Pure Python fallback
    result = 137 * 429 + 88
    print(f"137 × 429 + 88 = {result}")
    print("Used: Python arithmetic (no tools needed)")
"""
    output2 = run_code(executor, code2)

    return {
        "id": "tool_req_008",
        "category": "tool_requests",
        "difficulty": "intermediate",
        "query": "What tools do you have available? Then compute 137 × 429 + 88.",
        "response": (
            "<think>\n"
            "I should probe which tools are available before committing to an approach. "
            "A quick try/except for each candidate tool name tells me what I can use.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Only 'calculator' is available. I'll use it for the computation, "
            "but note it's also straightforward in Python directly.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "Tool discovery lets me adapt my approach to what the runtime provides. "
            "Many calculations don't need any tool — Python handles them natively."
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_web_search_unavailable,
    make_database_query_unavailable,
    make_file_read_unavailable,
    make_tool_available_and_used,
    make_tool_partial_fallback,
    make_multi_tool_workflow,
    make_llm_tool_call,
    make_tool_discovery,
]


def generate_examples() -> list:
    examples = []
    for builder in BUILDERS:
        try:
            ex = builder()
            examples.append(ex)
            print(f"  {ex['id']}: {ex['query'][:70]}...")
        except Exception as e:
            import traceback
            print(f"  SKIP {builder.__name__}: {e}")
            traceback.print_exc()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate tool-request training examples")
    parser.add_argument("--output", default="training/datasets/tool_requests/basic.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating tool-request examples...")
    examples = generate_examples()

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
