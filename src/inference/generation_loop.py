"""
Custom generation loop for WorldModel inference.

Two code paths:

  Retrained model (target):
    Generates <code>...</code> natively. Loop stops on </code> via
    StoppingCriteria, executes, injects <output>, continues.

  v1 model (compatibility):
    Forces </think> as a mid-generation stop, then forces <code> into
    context so the model generates isolated code content. Python is
    extracted by stripping non-ASCII trailing garbage (proxy tokens),
    executed for real, and the clean structure is reconstructed.
"""

import re
import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from ..executor.python_exec import PythonExecutor


THINK_CLOSE = "</think>"
CODE_OPEN   = "<code>"
CODE_CLOSE  = "</code>"
OUT_OPEN    = "<output>"
OUT_CLOSE   = "</output>"
MAX_CODE_CYCLES = 8

# Suppress Qwen3 tokens that compete with code-boundary positions:
#   Multimodal: <|object_ref_start|>…<|video_pad|>  (151646–151656)
#   FIM:        <|fim_prefix|>…<|fim_pad|>          (151659–151662)
_SUPPRESS_TOKEN_IDS = (
    [[i] for i in range(151646, 151657)] +
    [[i] for i in range(151659, 151663)]
)

_PYTHON_HINT_RE = re.compile(
    r'(print\s*\(|^\s*import |^\s*from |^\s*def |^\s*\w+\s*=)',
    re.MULTILINE,
)


class _StopOnTokens(StoppingCriteria):
    """Stop generation when any of the given token IDs is produced."""
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = set(stop_ids)
        self.triggered_id: Optional[int] = None

    def __call__(self, input_ids, scores, **kwargs):
        last = input_ids[0, -1].item()
        if last in self.stop_ids:
            self.triggered_id = last
            return True
        return False


def _generate(model, tokenizer, context: str, max_new_tokens: int,
              temperature: float, device: str,
              stop_ids: List[int] = None) -> tuple[str, Optional[int]]:
    """
    Generate tokens, stopping at EOS or any token in stop_ids.
    Returns (decoded_new_text, triggered_stop_id_or_None).
    """
    stopper = _StopOnTokens(stop_ids or [])
    input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=_SUPPRESS_TOKEN_IDS,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return new_text, stopper.triggered_id


def _extract_python_code(raw: str) -> Optional[str]:
    """
    Extract clean Python from raw model output that may end with
    non-ASCII proxy tokens. Takes lines up to the first non-ASCII line.
    """
    lines = []
    for line in raw.split('\n'):
        if any(ord(c) > 127 for c in line):
            break
        lines.append(line)
    code = '\n'.join(lines).strip()
    return code if (code and _PYTHON_HINT_RE.search(code)) else None


def extract_last_code_block(text: str) -> Optional[str]:
    """Extract the most recent <code>...</code> block."""
    last_open = text.rfind(CODE_OPEN)
    last_close = text.rfind(CODE_CLOSE)
    if last_open == -1 or last_close <= last_open:
        return None
    return text[last_open + len(CODE_OPEN):last_close].strip()


def generate_with_execution(
    model,
    tokenizer,
    prompt: str,
    executor: Optional[PythonExecutor] = None,
    max_new_tokens_per_step: int = 512,
    max_cycles: int = MAX_CODE_CYCLES,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    if executor is None:
        executor = PythonExecutor()

    think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    code_close_id  = tokenizer.convert_tokens_to_ids(CODE_CLOSE)

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    context = formatted_prompt
    cycles = 0

    while cycles < max_cycles:
        new_text, stop_reason = _generate(
            model, tokenizer, context,
            max_new_tokens_per_step, temperature, device,
            stop_ids=[think_close_id, code_close_id],
        )
        context += new_text
        hit_eos = tokenizer.eos_token in new_text or not new_text.strip()

        # ── Retrained model: proper </code> tag emitted ────────────────────
        if stop_reason == code_close_id or (CODE_CLOSE in new_text and CODE_OPEN in context):
            code = extract_last_code_block(context)
            if code:
                result = executor.run(code)
                context += f"\n{OUT_OPEN}\n{result.output_text()}\n{OUT_CLOSE}\n"
                cycles += 1
                if hit_eos:
                    break
                continue
            break

        # ── v1 model: </think> seen, no <code> yet ─────────────────────────
        if stop_reason == think_close_id:
            # Force <code> and generate the code content in isolation
            context += f"\n{CODE_OPEN}\n"
            code_raw, _ = _generate(
                model, tokenizer, context,
                max_new_tokens_per_step, temperature, device,
                # also stop on </think> so prose doesn't bleed into code content
                stop_ids=[code_close_id, think_close_id],
            )
            # v1 path: use ASCII extraction (not tag-based) to skip proxy-token garbage
            code = _extract_python_code(code_raw)
            if not code:
                # No executable code — remove the <code> we prematurely added and stop
                context = context[:context.rfind(f"\n{CODE_OPEN}\n")]
                break

            result = executor.run(code)
            # Rebuild context cleanly: drop proxy-token garbage from code_raw
            code_section_start = context.rfind(CODE_OPEN)
            context = (
                context[:code_section_start]
                + f"{CODE_OPEN}\n{code}\n{CODE_CLOSE}\n"
                + f"{OUT_OPEN}\n{result.output_text()}\n{OUT_CLOSE}\n"
            )
            cycles += 1
            continue  # let the loop generate closing prose (or another code cycle)

        if hit_eos:
            break

        # No code block and no recognised stop — stalled
        break

    response = context[len(formatted_prompt):]
    response = response.replace("<|im_end|>", "").strip()
    return response


def format_response_for_display(response: str) -> str:
    """Strip think blocks for user-facing display."""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()
