#!/usr/bin/env python3
"""
Generate programming / algorithms training examples.

Covers:
- Sorting: bubble, insertion, merge, quicksort — trace & complexity
- Searching: linear, binary search — trace & complexity
- String algorithms: palindrome, anagram, frequency, longest common substring
- Recursion: factorial, fibonacci, Tower of Hanoi
- Data structures: stack/queue operations, linked list traversal, binary tree
- Dynamic programming: coin change, LCS, knapsack (0/1)
- Graph algorithms: BFS/DFS distance, shortest path (Dijkstra)

These examples show the model thinking algorithmically, tracing through
execution, and analysing complexity.

Usage:
    python training/scripts/generate_programming.py
    python training/scripts/generate_programming.py --output training/datasets/programming/basic.jsonl
    python training/scripts/generate_programming.py --count 120
"""

import sys
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import PythonExecutor


def run_code(executor: PythonExecutor, code: str) -> str:
    result = executor.run(code)
    return result.output_text().strip()


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def make_bubble_sort(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    arr = random.sample(range(1, 30), random.choice([6, 7, 8]))
    executor = PythonExecutor()
    code = f"""\
def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    comparisons = 0
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            comparisons += 1
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                swaps += 1
    return a, comparisons, swaps

arr = {arr}
sorted_arr, comps, swaps = bubble_sort(arr)
print(f"Input:  {{arr}}")
print(f"Sorted: {{sorted_arr}}")
print(f"Comparisons: {{comps}}  (worst case n(n-1)/2 = {{len(arr)*(len(arr)-1)//2}})")
print(f"Swaps:       {{swaps}}")
print(f"Complexity: O(n²) time, O(1) space")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "basic",
        "query": f"Sort the array {arr} using bubble sort. Count comparisons and swaps. What is the time complexity?",
        "response": (
            "<think>\n"
            "Bubble sort: repeatedly compare adjacent pairs and swap if out of order.\n"
            "After i passes, the i largest elements are in their final positions.\n"
            "Time: O(n²) worst/average. Space: O(1).\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "Bubble sort is O(n²) — inefficient for large arrays, but easy to implement."
        ),
    }


def make_merge_sort(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    arr = random.sample(range(1, 50), random.choice([7, 8, 9]))
    executor = PythonExecutor()
    code = f"""\
def merge_sort(arr, depth=0):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid],  depth+1)
    right = merge_sort(arr[mid:], depth+1)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

import math
arr = {arr}
n   = len(arr)
result = merge_sort(arr)
print(f"Input:  {{arr}}")
print(f"Sorted: {{result}}")
print(f"n = {{n}}")
print(f"Comparisons (upper bound): n·log₂n ≈ {{n * math.log2(n):.1f}}")
print(f"Complexity: O(n log n) time, O(n) space")
print(f"Stable: yes (equal elements keep original order)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "intermediate",
        "query": (
            f"Sort {arr} using merge sort. Explain the divide-and-conquer approach "
            "and state the time and space complexity."
        ),
        "response": (
            "<think>\n"
            "Merge sort: divide array in half recursively, sort each half, merge.\n"
            "Merge step: O(n). Depth of recursion: O(log n). Total: O(n log n).\n"
            "Space: O(n) for merge buffers. Stable sort.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "Merge sort guarantees O(n log n) even in the worst case — unlike quicksort."
        ),
    }


def make_binary_search(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    n = random.choice([15, 20, 30])
    arr = sorted(random.sample(range(1, 100), n))
    target = random.choice(arr + [random.randint(1, 100)])  # sometimes miss
    executor = PythonExecutor()
    code = f"""\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    steps  = []
    while lo <= hi:
        mid = (lo + hi) // 2
        steps.append(f"  lo={{lo}}, hi={{hi}}, mid={{mid}}, arr[mid]={{arr[mid]}}")
        if arr[mid] == target:
            return mid, steps
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, steps

arr    = {arr}
target = {target}
idx, steps = binary_search(arr, target)

print(f"Array (sorted, n={{len(arr)}}): {{arr}}")
print(f"Searching for: {{target}}")
print(f"Steps ({{len(steps)}} comparisons):")
for s in steps:
    print(s)
if idx >= 0:
    print(f"Found at index {{idx}} ✓")
else:
    print(f"Not found")

import math
print(f"Max comparisons needed: ⌈log₂({{len(arr)}})⌉ = {{math.ceil(math.log2(len(arr)))}}")
print(f"Linear search would need up to {{len(arr)}} comparisons")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "basic",
        "query": (
            f"Search for {target} in the sorted array {arr} using binary search. "
            "Trace each comparison and compare efficiency to linear search."
        ),
        "response": (
            "<think>\n"
            "Binary search: maintain lo/hi pointers; check midpoint; discard half at each step.\n"
            "Requires sorted input. O(log n) time vs O(n) for linear.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# String algorithms
# ---------------------------------------------------------------------------

def make_palindrome(ex_id: str) -> dict:
    words = ["racecar", "hello", "madam", "python", "level", "openai", "civic",
             "kayak", "noon", "world", "radar", "deed"]
    random.seed(random.randint(1, 999))
    samples = random.sample(words, 5)
    executor = PythonExecutor()
    code = f"""\
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]

def longest_palindrome_substr(s):
    \"\"\"Expand around centre approach — O(n²).\"\"\"
    n   = len(s)
    best = s[0]
    for centre in range(n):
        for lo, hi in [(centre, centre), (centre, centre+1)]:  # odd, even
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                lo -= 1; hi += 1
            candidate = s[lo+1:hi]
            if len(candidate) > len(best):
                best = candidate
    return best

words = {samples}
print("Palindrome check:")
for w in words:
    print(f"  {{w!r:<12}} → {{is_palindrome(w)}}")

print()
sentence = "racecarabcba"
lps = longest_palindrome_substr(sentence)
print(f"Longest palindromic substring in {{sentence!r!r}}: {{lps!r}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "basic",
        "query": (
            f"Check whether each of {samples} is a palindrome. "
            "Also find the longest palindromic substring in 'racecarabcba'."
        ),
        "response": (
            "<think>\n"
            "Palindrome check: compare string to its reverse. O(n).\n"
            "Longest palindromic substring: expand-around-centre approach. O(n²).\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_anagram(ex_id: str) -> dict:
    pairs = [("listen", "silent"), ("hello", "world"), ("triangle", "integral"),
             ("python", "typhon"), ("cat", "act"), ("abc", "bca"), ("dog", "god")]
    random.seed(random.randint(1, 999))
    samples = random.sample(pairs, 4)
    executor = PythonExecutor()
    pairs_code = str(samples)
    code = f"""\
from collections import Counter

def are_anagrams(a, b):
    return Counter(a.lower()) == Counter(b.lower())

def group_anagrams(words):
    groups = {{}}
    for w in words:
        key = tuple(sorted(w.lower()))
        groups.setdefault(key, []).append(w)
    return list(groups.values())

pairs = {pairs_code}
print("Anagram pairs:")
for a, b in pairs:
    result = are_anagrams(a, b)
    freq_a = dict(sorted(Counter(a).items()))
    print(f"  {{a!r}} vs {{b!r}}: {{result}}  (chars: {{freq_a}})")

print()
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
groups = group_anagrams(words)
print(f"Group anagrams from {{words}}:")
for g in groups:
    print(f"  {{sorted(g)}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "basic",
        "query": (
            f"Check if these pairs are anagrams: {samples}. "
            "Also group the words ['eat', 'tea', 'tan', 'ate', 'nat', 'bat'] by anagram family."
        ),
        "response": (
            "<think>\n"
            "Anagram test: two strings are anagrams if they have the same character frequency. "
            "Use Counter for O(n) check.\n"
            "Grouping: sort each word's characters as a key → same key = same anagram family.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------

def make_fibonacci(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    n = random.choice([10, 15, 20, 25])
    executor = PythonExecutor()
    code = f"""\
import functools, sys
sys.setrecursionlimit(10000)

# Naive recursive (exponential) — only for small n
@functools.lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1: return n
    return fib_memo(n-1) + fib_memo(n-2)

# Iterative (O(n) time, O(1) space)
def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

n = {n}
result = fib_iter(n)
print(f"Fibonacci({{n}}) = {{result}}")
print(f"First 15 Fibonacci numbers:")
print(f"  {{[fib_iter(i) for i in range(15)]}}")
print()
print(f"Naive recursion: O(2^n) — calls grow exponentially")
print(f"Memoised / iterative: O(n) — each value computed once")
# Verify
assert fib_memo(n) == result
print(f"Memo check: fib_memo({{n}}) = {{fib_memo(n)}} ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "basic",
        "query": (
            f"Compute Fibonacci({n}). Show the first 15 Fibonacci numbers, "
            "explain the complexity difference between naive recursion and the iterative approach."
        ),
        "response": (
            "<think>\n"
            "Naive recursive fib: T(n) = T(n-1) + T(n-2) → O(2^n) calls.\n"
            "Memoisation or iteration: O(n) time, O(1) space for iterative.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------

def make_coin_change(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    coins = random.choice([[1, 5, 10, 25], [1, 3, 4, 7], [2, 5, 10]])
    amount = random.choice([11, 15, 23, 30, 41])
    executor = PythonExecutor()
    code = f"""\
def coin_change(coins, amount):
    # dp[i] = minimum coins to make amount i
    INF = float('inf')
    dp  = [INF] * (amount + 1)
    dp[0] = 0
    used = [[] for _ in range(amount + 1)]

    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a-c] + 1 < dp[a]:
                dp[a] = dp[a-c] + 1
                used[a] = used[a-c] + [c]

    return dp[amount], used[amount]

coins  = {coins}
amount = {amount}
min_coins, combo = coin_change(coins, amount)

print(f"Coins: {{coins}}")
print(f"Target: {{amount}}")
if min_coins == float('inf'):
    print(f"Cannot make {{amount}} with given coins")
else:
    print(f"Minimum coins: {{min_coins}}")
    print(f"Combination:   {{sorted(combo, reverse=True)}}")
    print(f"Check: sum = {{sum(combo)}} ✓")
    print()
    print(f"DP table (0 to {{amount}}):")
    dp = [float('inf')] * (amount + 1); dp[0] = 0
    for a in range(1, amount+1):
        for c in coins:
            if c <= a and dp[a-c]+1 < dp[a]: dp[a] = dp[a-c]+1
    print(f"  {{list(enumerate(dp))}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "intermediate",
        "query": (
            f"Find the minimum number of coins from {coins} that sum to {amount}. "
            "Use dynamic programming and show the DP table."
        ),
        "response": (
            "<think>\n"
            "Coin change (minimum coins): classic unbounded knapsack DP.\n"
            "dp[0] = 0; dp[a] = min(dp[a-c]+1 for c in coins if c <= a)\n"
            "O(amount × len(coins)) time and space.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_lcs(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    pairs = [("ABCBDAB", "BDCAB"), ("AGGTAB", "GXTXAYB"), ("ABCDE", "ACE"),
             ("longest", "stone"), ("python", "typhoon")]
    s1, s2 = random.choice(pairs)
    executor = PythonExecutor()
    code = f"""\
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    # dp[i][j] = length of LCS of s1[:i] and s2[:j]
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find the actual LCS
    lcs_str = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs_str.append(s1[i-1]); i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return dp[m][n], ''.join(reversed(lcs_str))

s1 = {repr(s1)}
s2 = {repr(s2)}
length, subsequence = lcs(s1, s2)

print(f"LCS of {{s1!r}} and {{s2!r}}")
print(f"Length: {{length}}")
print(f"Subsequence: {{subsequence!r}}")
print(f"Complexity: O(m×n) = O({{len(s1)}}×{{len(s2)}}) = O({{len(s1)*len(s2)}})")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "intermediate",
        "query": (
            f"Find the Longest Common Subsequence (LCS) of '{s1}' and '{s2}'. "
            "Use dynamic programming. State the time complexity."
        ),
        "response": (
            "<think>\n"
            "LCS DP: dp[i][j] = LCS length for s1[:i] and s2[:j].\n"
            "Recurrence: if s1[i]==s2[j]: dp[i][j] = dp[i-1][j-1]+1\n"
            "            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "Backtrack from dp[m][n] to recover the actual subsequence.\n"
            "O(m×n) time and space.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def make_dijkstra(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    # Small weighted graph
    nodes = ['A', 'B', 'C', 'D', 'E']
    edges = [
        ('A', 'B', random.randint(1, 5)),
        ('A', 'C', random.randint(3, 8)),
        ('B', 'C', random.randint(1, 4)),
        ('B', 'D', random.randint(2, 6)),
        ('C', 'E', random.randint(1, 5)),
        ('D', 'E', random.randint(1, 4)),
    ]
    src = 'A'
    executor = PythonExecutor()
    edges_code = str(edges)
    code = f"""\
import heapq

edges = {edges_code}
nodes = ['A','B','C','D','E']

# Build adjacency list
graph = {{n: [] for n in nodes}}
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))

def dijkstra(graph, src):
    dist = {{n: float('inf') for n in graph}}
    prev = {{n: None for n in graph}}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev

dist, prev = dijkstra(graph, '{src}')

print("Graph edges:")
for u,v,w in edges:
    print(f"  {{u}}-{{v}}: {{w}}")
print()
print(f"Shortest distances from {{'{src}'}}:")
for node in sorted(dist):
    # Reconstruct path
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path_str = ' → '.join(reversed(path))
    print(f"  {{'{src}'}}→{{node}}: {{dist[node]}}  (path: {{path_str}})")
"""
    output = run_code(executor, code)
    edges_display = ", ".join(f"{u}-{v}:{w}" for u,v,w in edges)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "intermediate",
        "query": (
            f"Find shortest paths from A in this weighted graph: {edges_display}. "
            "Use Dijkstra's algorithm and show the path to each node."
        ),
        "response": (
            "<think>\n"
            "Dijkstra's algorithm: use a min-heap (priority queue) to always expand "
            "the nearest unvisited node. Greedy approach works for non-negative weights.\n"
            "Time: O((V+E) log V) with a binary heap.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_bfs_levels(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    # Generate a simple undirected graph
    n = 7
    edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(3,6),(4,6)]
    src = 0
    executor = PythonExecutor()
    code = f"""\
from collections import deque

edges = {edges}
n = {n}
src = {src}

# Build adjacency list
graph = [[] for _ in range(n)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

def bfs(graph, src):
    dist   = [-1] * len(graph)
    parent = [-1] * len(graph)
    dist[src] = 0
    queue = deque([src])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return dist, parent, order

dist, parent, order = bfs(graph, src)

print(f"BFS from node {{src}} ({{n}} nodes)")
print(f"Edges: {{edges}}")
print(f"Visit order: {{order}}")
print()
print(f"Node | Distance | Path from {{src}}")
print("-" * 40)
for node in range(n):
    path = []
    cur = node
    while cur != -1:
        path.append(cur); cur = parent[cur]
    print(f"  {{node:2d}} |    {{dist[node]:2d}}    | {{' → '.join(map(str, reversed(path)))}}")

# BFS tree levels
levels = {{}}
for node, d in enumerate(dist):
    levels.setdefault(d, []).append(node)
print()
print("BFS levels (distance from source):")
for lvl in sorted(levels):
    print(f"  Level {{lvl}}: {{levels[lvl]}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "intermediate",
        "query": (
            f"Run BFS from node 0 on this graph (7 nodes): edges {edges}. "
            "Show the visit order, distance to each node, and BFS level structure."
        ),
        "response": (
            "<think>\n"
            "BFS: use a queue, process nodes in FIFO order.\n"
            "Track distance (level) from source. Each level = one hop further.\n"
            "Time: O(V+E). BFS gives shortest paths in unweighted graphs.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------

def make_algorithm_comparison(ex_id: str) -> dict:
    """Compare sorting algorithms on the same array."""
    random.seed(random.randint(1, 999))
    arr = random.sample(range(1, 100), 12)
    executor = PythonExecutor()
    code1 = f"""\
import time, random

arr = {arr}
n   = len(arr)

# Count comparisons for different algorithms
def insertion_sort_counted(a):
    a = a[:]
    comps = 0
    for i in range(1, len(a)):
        key = a[i]; j = i-1
        while j >= 0:
            comps += 1
            if a[j] > key:
                a[j+1] = a[j]; j -= 1
            else:
                break
        a[j+1] = key
    return a, comps

def bubble_sort_counted(a):
    a = a[:]; comps = 0
    for i in range(len(a)):
        for j in range(len(a)-i-1):
            comps += 1
            if a[j] > a[j+1]: a[j], a[j+1] = a[j+1], a[j]
    return a, comps

_, is_comps  = insertion_sort_counted(arr)
_, bs_comps  = bubble_sort_counted(arr)
py_sorted    = sorted(arr)  # Timsort

print(f"Sorting {{arr}}")
print(f"n = {{n}}")
print(f"Sorted: {{py_sorted}}")
print()
print(f"Algorithm         | Comparisons | Big-O     | Best for")
print(f"------------------|-------------|-----------|------------------")
print(f"Bubble sort       |  {{bs_comps:>4}}       | O(n²)     | Teaching")
print(f"Insertion sort    |  {{is_comps:>4}}       | O(n²)     | Small/nearly sorted")
print(f"Merge sort        |   ~{{n * __import__('math').ceil(__import__('math').log2(n))*2:>2}}       | O(n log n)| General purpose")
print(f"Python (Timsort)  |   optimal   | O(n log n)| Everything")
"""
    output1 = run_code(executor, code1)
    code2 = """\
import math
print("Scaling comparison (approximate comparisons for n elements):")
print(f"{'n':>8} | {'O(n²)':>12} | {'O(n log n)':>12} | {'speedup':>10}")
print("-" * 50)
for n in [10, 100, 1000, 10000, 100000]:
    n2      = n*n
    nlogn   = int(n * math.log2(n))
    speedup = n2 / nlogn
    print(f"{n:>8,} | {n2:>12,} | {nlogn:>12,} | {speedup:>10.1f}x")
"""
    output2 = run_code(executor, code2)
    return {
        "id": ex_id,
        "category": "programming",
        "difficulty": "advanced",
        "query": (
            f"Compare bubble sort, insertion sort, and merge sort on {arr}. "
            "Count comparisons for the O(n²) algorithms, then show how the "
            "O(n log n) advantage grows with input size."
        ),
        "response": (
            "<think>\n"
            "I'll instrument bubble sort and insertion sort to count comparisons, "
            "then show the theoretical scaling difference as n grows.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Now show the scaling table to illustrate why O(n log n) dominates at scale.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "The speedup ratio grows linearly with n: at 100k elements, "
            "O(n log n) does ~6000x fewer comparisons than O(n²)."
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_bubble_sort,
    make_merge_sort,
    make_binary_search,
    make_palindrome,
    make_anagram,
    make_fibonacci,
    make_coin_change,
    make_lcs,
    make_dijkstra,
    make_bfs_levels,
    make_algorithm_comparison,
]


def generate_examples(count: int) -> list:
    examples = []
    idx = 1
    per_builder = max(1, count // len(BUILDERS))
    remainder = count - per_builder * len(BUILDERS)

    random.seed(42)
    for i, builder in enumerate(BUILDERS):
        n = per_builder + (1 if i < remainder else 0)
        for _ in range(n):
            ex_id = f"prog_{idx:03d}"
            try:
                ex = builder(ex_id)
                examples.append(ex)
                idx += 1
                print(f"  {ex_id}: {ex['query'][:70]}...")
            except Exception as e:
                import traceback
                print(f"  SKIP {ex_id} ({builder.__name__}): {e}")
                traceback.print_exc()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate programming training examples")
    parser.add_argument("--output", default="training/datasets/programming/basic.jsonl")
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} programming examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
