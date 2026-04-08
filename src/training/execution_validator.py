import torch
"""
Execution-Based Validation for WorldModel Training
===================================================

Validates generated code by:
1. Extracting <code> blocks from generated responses
2. Checking syntactic validity via ast.parse()
3. Optionally executing code and comparing output
"""

import ast
import sys
import io
import re
import logging
from typing import Optional, List, Dict, Any, Tuple

# Extract code between <code>...</code> tags
_CODE_BLOCK_RE = re.compile(r"<code>\s*\n?(.*?)\n?\s*</code>", re.DOTALL)

# Qwen3 may emit <|fim_suffix|> in place of custom <code>/<output> tokens when
# the new token embeddings were not learned for output generation.
# After </think>, the first <|fim_suffix|>...<|fim_suffix|> block is the code.
_FIM_BLOCK_RE = re.compile(r"<\|fim_suffix\|>\s*\n?(.*?)\n?\s*<\|fim_suffix\|>", re.DOTALL)

# Detect Python-like code
_PYTHON_LINE_RE = re.compile(
    r"^\s*(import |from |def |class |if |for |while |try:|except|with |"
    r"print\(|return |raise |yield |assert |elif |else:|pass$|break$|continue$|"
    r"[a-zA-Z_][a-zA-Z0-9_]*\s*=|\#\s)",
    re.MULTILINE,
)

TAG_CLOSE = "</think>"
TAG_OPEN_PART = "/think>"


class CodeCheckResult:
    """Result of checking a single code block."""
    def __init__(self, code: Optional[str], syntax_valid: bool,
                 syntax_error: Optional[str] = None,
                 executed: bool = False, execution_error: Optional[str] = None,
                 stdout: str = "", stderr: str = "",
                 expected_output_match: bool = False):
        self.code = code
        self.syntax_valid = syntax_valid
        self.syntax_error = syntax_error
        self.executed = executed
        self.execution_error = execution_error
        self.stdout = stdout
        self.stderr = stderr
        self.expected_output_match = expected_output_match

    def summary(self) -> str:
        parts = []
        if self.code:
            parts.append(f"code={len(self.code)}c")
        else:
            parts.append("no_code")
        if self.syntax_valid:
            parts.append("syntax=OK")
        elif self.syntax_error:
            parts.append(f"syntax=ERR:{self.syntax_error[:40]}")
        if self.executed:
            parts.append("exec=OK")
        elif self.execution_error:
            parts.append(f"exec=ERR:{self.execution_error[:40]}")
        return " ".join(parts)


def extract_code(text: str) -> Optional[str]:
    """Extract code from generated text."""
    # Primary: tagged <code> blocks
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # Secondary: <|fim_suffix|> blocks — Qwen3 uses FIM tokens when custom <code>
    # embeddings were not trained for output. After </think>, the first
    # <|fim_suffix|>...<|fim_suffix|> block is code.
    if TAG_CLOSE in text:
        after_think = text.split(TAG_CLOSE, 1)[-1]
        fim_matches = _FIM_BLOCK_RE.findall(after_think)
        if fim_matches:
            code = fim_matches[0].strip()
            if code and _PYTHON_LINE_RE.search(code):
                return code

    # Fallback: Python code after </think>
    if TAG_CLOSE not in text:
        return None

    parts = text.split(TAG_CLOSE)
    after = parts[-1].strip()

    # Strip any leftover tags
    after = after.replace(TAG_CLOSE, "")
    after = after.replace(TAG_OPEN_PART, "")
    after = after.strip()

    if not after:
        return None

    if not _PYTHON_LINE_RE.search(after):
        return None

    # Collect consecutive Python lines
    code_lines = []
    for line in after.split("\n"):
        line = line.rstrip()
        if not line:
            if code_lines:
                code_lines.append(line)
        elif _PYTHON_LINE_RE.match(line):
            code_lines.append(line)
        elif code_lines and line.startswith((" ", "\t")):
            code_lines.append(line)
        else:
            break

    code = "\n".join(code_lines).strip()
    if len(code) >= 10:
        return code
    return None


_PRELOAD_SCRIPT = """\
import math, random, statistics, functools, heapq, itertools, fractions, re, json, sys, os
from collections import defaultdict, deque, Counter, OrderedDict, namedtuple
try:
    import numpy as np
    import numpy
except ImportError:
    pass
try:
    import scipy
except ImportError:
    pass
try:
    import sympy
    from sympy import symbols, solve, simplify, expand, factor, diff, integrate
except ImportError:
    pass
try:
    import networkx as nx
except ImportError:
    pass
"""


class ExecutionValidator:
    """Validates that generated code is syntactically correct and optionally executes it."""

    def __init__(self, executor=None, execution_timeout: float = 5.0,
                 validate_output: bool = True):
        self.executor = executor
        self.execution_timeout = execution_timeout
        self.validate_output = validate_output
        if self.executor is not None:
            self.executor.run(_PRELOAD_SCRIPT)

    @staticmethod
    def check_syntax(code: str) -> tuple[bool, Optional[str]]:
        """Check if code is syntactically valid Python using ast.parse()."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            msg = f"SyntaxError: {e.msg}"
            if e.lineno:
                msg += f" (line {e.lineno})"
            return False, msg

    def execute_code(self, code: str) -> tuple[bool, str, str, Optional[str]]:
        """Execute code and capture output."""
        if self.executor is not None:
            result = self.executor.run(code)
            return (
                result.success,
                result.stdout,
                result.stderr if not result.success else "",
                None if result.success else f"Execution failed: {result.stderr[:200]}",
            )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            exec(code, {"__builtins__": __builtins__})
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return (success, stdout_capture.getvalue(), stderr_capture.getvalue(), error_msg)

    def check_code(self, code: str, expected_output: Optional[str] = None) -> CodeCheckResult:
        """Check a single code block: syntax + execution + output matching."""
        if not code:
            return CodeCheckResult(code=None, syntax_valid=False)

        syntax_valid, syntax_error = self.check_syntax(code)
        if not syntax_valid:
            return CodeCheckResult(code=code, syntax_valid=False, syntax_error=syntax_error)

        executed = False
        execution_error = None
        stdout = ""
        stderr = ""
        expected_match = False

        success, out, err, err_msg = self.execute_code(code)
        stdout = out
        stderr = err

        if success:
            executed = True
            if self.validate_output and expected_output is not None:
                actual = out.strip().lower()
                expected = expected_output.strip().lower()
                expected_match = (
                    expected == actual or
                    expected in actual or
                    actual in expected
                )
        else:
            execution_error = err_msg

        return CodeCheckResult(
            code=code, syntax_valid=True, executed=executed,
            execution_error=execution_error, stdout=stdout, stderr=stderr,
            expected_output_match=expected_match,
        )

    @torch.no_grad()
    def validate_batch(
        self, model, tokenizer, examples: List[Dict], device: str,
        max_new_tokens: int = 512, temperature: float = 0.0,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate responses for examples and validate the code."""
        if sample_size is not None:
            examples = examples[:sample_size]

        total = len(examples)
        results = []
        code_found = 0
        syntax_valid = 0
        executed_ok = 0
        output_match = 0

        for i, ex in enumerate(examples):
            query = ex.get("query", "")
            expected_response = ex.get("response", "")

            expected_output = None
            if self.validate_output:
                out_match = re.search(r'<output>\s*\n?(.*?)\n?\s*</output>',
                                     expected_response, re.DOTALL)
                if out_match:
                    expected_output = out_match.group(1).strip()

            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False, add_generation_prompt=True,
            )

            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            gen_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=False,
            )

            code = extract_code(gen_text)
            if code:
                code_found += 1
                result = self.check_code(code, expected_output)
                results.append(result)

                if result.syntax_valid:
                    syntax_valid += 1
                if result.executed:
                    executed_ok += 1
                if result.expected_output_match:
                    output_match += 1
            else:
                results.append(CodeCheckResult(
                    code=None, syntax_valid=False,
                    syntax_error="No <code> block found in response"
                ))

        return {
            "total": total,
            "code_extract_rate": code_found / max(total, 1),
            "syntax_valid_rate": syntax_valid / max(total, 1),
            "execution_success_rate": executed_ok / max(total, 1),
            "expected_output_match": output_match / max(total, 1),
            "code_found": code_found,
            "syntax_valid": syntax_valid,
            "executed_ok": executed_ok,
            "output_match": output_match,
            "results": results,
        }

    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """Format validation metrics as a human-readable string."""
        return (
            f"Code: {metrics['code_found']}/{metrics['total']} "
            f"({metrics['code_extract_rate']:.0%}) | "
            f"Syntax: {metrics['syntax_valid']}/{metrics['total']} "
            f"({metrics['syntax_valid_rate']:.0%}) | "
            f"Exec: {metrics['executed_ok']}/{metrics['total']} "
            f"({metrics['execution_success_rate']:.0%}) | "
            f"Output: {metrics['output_match']}/{metrics['total']} "
            f"({metrics['expected_output_match']:.0%})"
        )

    @staticmethod
    def format_errors(metrics: Dict[str, Any]) -> str:
        """Format syntax/execution errors for logging."""
        errors = []
        for r in metrics.get("results", []):
            if r.code and not r.syntax_valid:
                errors.append(f"  SYNTAX: {r.syntax_error}")
                lines = (r.code or "").split("\n")[:5]
                for line in lines:
                    errors.append(f"    {line}")
                if r.code and len(r.code.split("\n")) > 5:
                    errors.append("    ...")
                errors.append("")
            elif r.code and r.syntax_valid and not r.executed:
                errors.append(f"  EXEC: {r.execution_error}")
                errors.append("")
            elif r.code is None:
                errors.append(f"  NO_CODE: {r.syntax_error}")
                errors.append("")
        return "\n".join(errors)


class CodeRewardFunction:
    """Composite reward for geometric candidate evaluation based on code validity."""

    def __init__(self, reward_bonus: float = 0.2,
                 syntax_penalty: float = 0.5, exec_bonus: float = 0.1):
        self.reward_bonus = reward_bonus
        self.syntax_penalty = syntax_penalty
        self.exec_bonus = exec_bonus

    def compute_score(self, token_loss: float, results: List[CodeCheckResult]) -> float:
        """Compute composite score from token loss + code check results. Lower is better."""
        if not results:
            return token_loss

        total = len(results)
        syntax_ok = sum(1 for r in results if r.syntax_valid)
        exec_ok = sum(1 for r in results if r.executed)
        no_code = sum(1 for r in results if r.code is None)

        score = token_loss
        score -= self.reward_bonus * syntax_ok / total
        score -= self.exec_bonus * exec_ok / total
        score += self.syntax_penalty * no_code / total
        return score

    def score_to_metrics(self, score: float, token_loss: float,
                         results: List[CodeCheckResult]) -> Dict[str, float]:
        """Break down how score was computed."""
        if not results:
            return {"raw_loss": token_loss, "adjusted_score": score}
        total = len(results)
        return {
            "raw_loss": token_loss,
            "adjusted_score": score,
            "syntax_bonus": self.reward_bonus * sum(1 for r in results if r.syntax_valid) / total,
            "exec_bonus": self.exec_bonus * sum(1 for r in results if r.executed) / total,
            "no_code_penalty": self.syntax_penalty * sum(1 for r in results if r.code is None) / total,
        }
