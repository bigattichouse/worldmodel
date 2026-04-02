#!/usr/bin/env python3
"""
Generate Boolean/logic training examples (Prolog-style, Python implementation).

Re-creates the ByteLogic graph/relation problems using Python sets/dicts.
Covers: graph reachability, transitive closure, family trees, mutual relations,
inheritance chains, and logical deduction.

The model learns to represent relations as dicts of sets and write
inference rules as Python functions — equivalent to Prolog but readable.

Usage:
    python training/scripts/generate_logic.py --output training/datasets/logic/basic.jsonl --count 120
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once, PythonExecutor


def execute(code: str) -> str:
    return run_once(code).output_text()


def make_example(ex_id, category, difficulty, query, think, model_text, code):
    output = execute(code)
    parts = []
    if think:
        parts.append(f"<think>\n{think.strip()}\n</think>")
    if model_text:
        parts.append(f"<model>\n{model_text.strip()}\n</model>")
    parts.append(f"<code>\n{code.strip()}\n</code>")
    parts.append(f"<output>\n{output.strip()}\n</output>")
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": "\n".join(parts)}


# ─── Name pools ───────────────────────────────────────────────────────────────
NAMES = ["alice", "bob", "charlie", "diana", "eve", "frank",
         "grace", "henry", "iris", "jack", "karen", "leo"]
NODES = list("abcdefghij")


# ─── Graph reachability ───────────────────────────────────────────────────────

def gen_reachability(rng):
    """Can you reach node X from node Y? (transitive closure via BFS)"""
    n = rng.randint(5, 8)
    nodes = NODES[:n]
    # Build a connected-ish DAG
    edges = []
    for i in range(1, n):
        src = nodes[rng.randint(0, i-1)]
        edges.append((src, nodes[i]))
    # Maybe a few extra edges
    for _ in range(rng.randint(1, 3)):
        a, b = rng.sample(nodes, 2)
        if (a, b) not in edges and a != b:
            edges.append((a, b))

    start, end = nodes[0], nodes[-1]
    query = f"Can you reach node '{end}' from node '{start}'? Edges: {edges}"
    think = "Build the graph as an adjacency dict, then use BFS to check reachability."
    model_text = "Store edges as dict of sets. BFS from start, return True if end found."
    code = (
        f"from collections import defaultdict, deque\n\n"
        f"edges = {edges}\n"
        f"graph = defaultdict(set)\n"
        f"for src, dst in edges:\n"
        f"    graph[src].add(dst)\n\n"
        f"def can_reach(graph, start, end):\n"
        f"    visited = set()\n"
        f"    queue = deque([start])\n"
        f"    while queue:\n"
        f"        node = queue.popleft()\n"
        f"        if node == end:\n"
        f"            return True\n"
        f"        if node not in visited:\n"
        f"            visited.add(node)\n"
        f"            queue.extend(graph[node] - visited)\n"
        f"    return False\n\n"
        f"result = can_reach(graph, '{start}', '{end}')\n"
        f"print(f\"Can reach '{end}' from '{start}': {{result}}\")"
    )
    return query, think, model_text, code


def gen_all_reachable(rng):
    """Find all nodes reachable from X"""
    n = rng.randint(6, 9)
    nodes = NODES[:n]
    edges = []
    for i in range(1, n):
        src = nodes[rng.randint(0, i-1)]
        edges.append((src, nodes[i]))
    for _ in range(2):
        a, b = rng.sample(nodes, 2)
        if (a, b) not in edges:
            edges.append((a, b))

    start = nodes[0]
    query = f"Find all nodes reachable from node '{start}'. Edges: {edges}"
    think = "BFS/DFS from the start node, collect all visited nodes."
    code = (
        f"from collections import defaultdict, deque\n\n"
        f"edges = {edges}\n"
        f"graph = defaultdict(set)\n"
        f"for src, dst in edges:\n"
        f"    graph[src].add(dst)\n\n"
        f"visited = set()\n"
        f"queue = deque(['{start}'])\n"
        f"while queue:\n"
        f"    node = queue.popleft()\n"
        f"    if node not in visited:\n"
        f"        visited.add(node)\n"
        f"        queue.extend(graph[node])\n\n"
        f"visited.discard('{start}')  # exclude start itself\n"
        f"print(f\"Reachable from '{start}': {{sorted(visited)}}\")"
    )
    return query, think, "", code


# ─── Family tree ──────────────────────────────────────────────────────────────

def gen_family_ancestors(rng):
    """Who are all the ancestors of X?"""
    names = rng.sample(NAMES, 6)
    # Build parent -> children
    gen0 = names[:2]   # grandparents
    gen1 = names[2:4]  # parents
    gen2 = names[4:6]  # children
    parent_of = {gen1[0]: gen0[0], gen1[1]: gen0[1], gen2[0]: gen1[0], gen2[1]: gen1[0]}
    target = gen2[0]

    parent_list = list(parent_of.items())
    query = f"Who are all the ancestors of {target}? Parent relationships: {parent_list}"
    think = "Walk up the parent chain repeatedly until no more parents found."
    code = (
        f"parent_of = dict({parent_list})\n\n"
        f"def ancestors(person, parent_of):\n"
        f"    result = []\n"
        f"    current = person\n"
        f"    while current in parent_of:\n"
        f"        current = parent_of[current]\n"
        f"        result.append(current)\n"
        f"    return result\n\n"
        f"result = ancestors('{target}', parent_of)\n"
        f"print(f\"Ancestors of '{target}': {{result}}\")"
    )
    return query, think, "", code


def gen_family_descendants(rng):
    """Who are all the descendants of X?"""
    names = rng.sample(NAMES, 7)
    root = names[0]
    children = {names[0]: [names[1], names[2]],
                names[1]: [names[3], names[4]],
                names[2]: [names[5]],
                names[5]: [names[6]]}
    children_list = [(k, v) for k, v in children.items()]

    query = f"Who are all the descendants of {root}? Children map: {children_list}"
    think = "BFS/DFS down the children tree to find all descendants."
    code = (
        f"from collections import deque\n\n"
        f"children = dict({children_list})\n\n"
        f"def all_descendants(person, children):\n"
        f"    result = []\n"
        f"    queue = deque(children.get(person, []))\n"
        f"    while queue:\n"
        f"        child = queue.popleft()\n"
        f"        result.append(child)\n"
        f"        queue.extend(children.get(child, []))\n"
        f"    return result\n\n"
        f"result = all_descendants('{root}', children)\n"
        f"print(f\"Descendants of '{root}': {{result}}\")"
    )
    return query, think, "", code


def gen_mutual_relation(rng):
    """Who do A and B both know? (mutual friends)"""
    names = rng.sample(NAMES, 8)
    a, b = names[0], names[1]
    shared = rng.sample(names[2:], rng.randint(1, 3))
    a_friends = shared + rng.sample([n for n in names[2:] if n not in shared], rng.randint(1, 2))
    b_friends = shared + rng.sample([n for n in names[2:] if n not in shared], rng.randint(1, 2))
    rng.shuffle(a_friends)
    rng.shuffle(b_friends)

    query = f"Who do both {a} and {b} know? {a} knows: {a_friends}. {b} knows: {b_friends}."
    think = "Intersection of both people's friend sets."
    code = (
        f"a_friends = set({a_friends})\n"
        f"b_friends = set({b_friends})\n"
        f"mutual = sorted(a_friends & b_friends)\n"
        f"print(f\"Both {a} and {b} know: {{mutual}}\")"
    )
    return query, think, "", code


# ─── Property inheritance ──────────────────────────────────────────────────────

def gen_inheritance_chain(rng):
    """What properties does X have? (through isa chain)"""
    # Build an isa hierarchy
    hierarchy = [
        ("poodle", "dog"),
        ("dog", "mammal"),
        ("mammal", "animal"),
    ]
    properties = {
        "animal": ["alive", "moves"],
        "mammal": ["warm-blooded", "has-fur"],
        "dog": ["barks", "four-legged"],
        "poodle": ["curly-fur", "intelligent"],
    }
    target = "poodle"
    query = f"What properties does a poodle have? ISA chain: {hierarchy}. Properties: {list(properties.items())}"
    think = (
        "Walk the isa chain upward, collecting properties at each level. "
        "This is like Prolog's property inheritance through isa/1."
    )
    model_text = (
        "For each class in the isa chain (including the class itself),\n"
        "collect all properties. Return the union."
    )
    code = (
        f"isa = dict({hierarchy})\n"
        f"properties = {dict(properties)}\n\n"
        f"def all_properties(entity, isa, properties):\n"
        f"    result = []\n"
        f"    current = entity\n"
        f"    while current:\n"
        f"        result.extend(properties.get(current, []))\n"
        f"        current = isa.get(current)\n"
        f"    return result\n\n"
        f"props = all_properties('{target}', isa, properties)\n"
        f"print(f\"Properties of poodle: {{props}}\")"
    )
    return query, think, model_text, code


def gen_sibling_detection(rng):
    """Who are X's siblings? (same parent)"""
    names = rng.sample(NAMES, 6)
    parent_of = {names[1]: names[0], names[2]: names[0],
                 names[3]: names[0], names[4]: names[5]}
    target = names[1]
    parent_list = list(parent_of.items())

    query = f"Who are {target}'s siblings (share the same parent)? Parent relationships: {parent_list}"
    think = "Find target's parent, then find all others with the same parent."
    code = (
        f"parent_of = dict({parent_list})\n\n"
        f"def siblings(person, parent_of):\n"
        f"    parent = parent_of.get(person)\n"
        f"    if not parent:\n"
        f"        return []\n"
        f"    return [p for p, par in parent_of.items() if par == parent and p != person]\n\n"
        f"result = siblings('{target}', parent_of)\n"
        f"print(f\"Siblings of '{target}': {{sorted(result)}}\")"
    )
    return query, think, "", code


def gen_set_logic(rng):
    """Boolean set operations: union, intersection, difference"""
    names_a = sorted(rng.sample(NAMES, rng.randint(3, 5)))
    names_b = sorted(rng.sample(NAMES, rng.randint(3, 5)))
    op = rng.choice(["union", "intersection", "difference", "symmetric_difference"])
    op_sym = {"union": "|", "intersection": "&", "difference": "-", "symmetric_difference": "^"}
    op_words = {
        "union": "belong to either set",
        "intersection": "belong to both sets",
        "difference": "are in A but not B",
        "symmetric_difference": "belong to exactly one set"
    }
    query = f"A = {names_a}, B = {names_b}. Find all elements that {op_words[op]}."
    think = f"Use set {op} operation."
    code = (
        f"A = set({names_a})\n"
        f"B = set({names_b})\n"
        f"result = sorted(A {op_sym[op]} B)\n"
        f"print(f'Result: {{result}}')"
    )
    return query, think, "", code


def gen_topological_sort(rng):
    """Topological ordering of tasks with dependencies"""
    n = rng.randint(4, 6)
    tasks = [f"task_{i}" for i in range(n)]
    # Build a DAG (no cycles)
    deps = {}
    for i in range(1, n):
        # each task depends on some earlier tasks
        num_deps = rng.randint(0, min(i, 2))
        deps[tasks[i]] = rng.sample(tasks[:i], num_deps)
    deps[tasks[0]] = []

    dep_list = [(t, d) for t, d in deps.items()]
    query = f"What order should these tasks run given dependencies? Dependencies: {dep_list}"
    think = "Topological sort: tasks with no pending dependencies run first."
    model_text = "Use Kahn's algorithm: track in-degrees, repeatedly emit zero-in-degree nodes."
    code = (
        f"from collections import defaultdict, deque\n\n"
        f"deps = dict({[(t, list(d)) for t, d in deps.items()]})\n"
        f"all_tasks = list(deps.keys())\n\n"
        f"# Build in-degree map\n"
        f"in_degree = {{t: 0 for t in all_tasks}}\n"
        f"dependents = defaultdict(list)\n"
        f"for task, prereqs in deps.items():\n"
        f"    for prereq in prereqs:\n"
        f"        in_degree[task] += 1\n"
        f"        dependents[prereq].append(task)\n\n"
        f"queue = deque([t for t in all_tasks if in_degree[t] == 0])\n"
        f"order = []\n"
        f"while queue:\n"
        f"    task = queue.popleft()\n"
        f"    order.append(task)\n"
        f"    for dep in dependents[task]:\n"
        f"        in_degree[dep] -= 1\n"
        f"        if in_degree[dep] == 0:\n"
        f"            queue.append(dep)\n\n"
        f"print(f'Execution order: {{order}}')"
    )
    return query, think, model_text, code


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("reachability",    "intermediate", gen_reachability,        0.15),
    ("all_reachable",   "intermediate", gen_all_reachable,       0.12),
    ("ancestors",       "basic",        gen_family_ancestors,    0.12),
    ("descendants",     "basic",        gen_family_descendants,  0.12),
    ("mutual",          "basic",        gen_mutual_relation,     0.12),
    ("inheritance",     "intermediate", gen_inheritance_chain,   0.10),
    ("siblings",        "basic",        gen_sibling_detection,   0.10),
    ("set_logic",       "basic",        gen_set_logic,           0.10),
    ("topo_sort",       "intermediate", gen_topological_sort,    0.07),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0
    total_w = sum(g[3] for g in GENERATORS)

    for _ in range(count):
        r = rng.random()
        cumulative = 0.0
        chosen = GENERATORS[0]
        for gen in GENERATORS:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break
        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        try:
            query, think, model_text, code = fn(rng)
            ex = make_example(f"logic_{idx:04d}", "logic", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping {name} (error: {e})")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate logic/Prolog-style training examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} logic examples...")
    examples = generate_examples(args.count, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")
    errors = sum(1 for ex in examples if "<code>" in ex["response"] and "<output>" not in ex["response"])
    print("All examples validated OK." if errors == 0 else f"WARNING: {errors} missing outputs")


if __name__ == "__main__":
    main()
