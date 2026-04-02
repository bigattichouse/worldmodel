"""
Custom generation loop for WorldModel inference.

Intercepts after </code> tokens, executes the code block, injects
<output>...</output> into context, then continues generation.

This is what makes the model "think with code" rather than just predict tokens.
"""

import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..executor.python_exec import PythonExecutor


# Tags used by the system
CODE_OPEN  = "<code>"
CODE_CLOSE = "</code>"
OUT_OPEN   = "<output>"
OUT_CLOSE  = "</output>"
MAX_CODE_CYCLES = 8  # safety limit on code/output cycles per response


def extract_last_code_block(text: str) -> Optional[str]:
    """Extract the most recent <code>...</code> block from text."""
    # Find the last occurrence
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
    """
    Generate a response with interleaved code execution.

    Args:
        model: Loaded HuggingFace causal LM
        tokenizer: Matching tokenizer
        prompt: User input (will be formatted as "User: ...\n\nAssistant: ")
        executor: PythonExecutor instance (fresh per call for isolated state,
                  or shared for multi-turn state persistence)
        max_new_tokens_per_step: Tokens to generate before checking for </code>
        max_cycles: Maximum code/output cycles before forcing stop
        temperature: Sampling temperature
        device: "cuda" or "cpu"

    Returns:
        Full response string including think/model/code/output blocks
    """
    if executor is None:
        executor = PythonExecutor()

    formatted_prompt = f"User: {prompt}\n\nAssistant: "
    context = formatted_prompt
    cycles = 0

    while cycles < max_cycles:
        input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

        # Stop tokens: </code> and end-of-sequence
        stop_token_ids = []
        for stop_str in [CODE_CLOSE, tokenizer.eos_token]:
            ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if ids:
                stop_token_ids.extend(ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens_per_step,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        context += new_text

        # Check if we hit end of sequence
        if tokenizer.eos_token in new_text or not new_text.strip():
            break

        # Check if a code block just closed
        if CODE_CLOSE in new_text:
            code = extract_last_code_block(context)
            if code:
                result = executor.run(code)
                output_text = result.output_text()
                context += f"\n{OUT_OPEN}\n{output_text}\n{OUT_CLOSE}\n"
                cycles += 1
            else:
                break
        else:
            # No code block and no EOS — generation stalled or hit token limit
            break

    # Strip the prompt prefix, return only the assistant response
    response = context[len(formatted_prompt):]
    return response.strip()


def format_response_for_display(response: str) -> str:
    """
    Strip think blocks for user-facing display.
    Keeps model, code, output, and prose.
    """
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()
