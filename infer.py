#!/usr/bin/env python3
"""
WorldModel Inference CLI
=========================
Interactive shell using the custom generation loop with code execution.

Usage:
    # With trained LoRA adapter:
    python infer.py --model ./output/worldmodel/final

    # With base model (no fine-tuning yet, for testing the loop):
    python infer.py --model Qwen/Qwen3-1.7B --base-only

    # Single query:
    python infer.py --model ./output/worldmodel/final --query "What is 15% of 240?"

Options:
    --model         Path to fine-tuned model directory, or HF model name
    --base-only     Load as base model (no LoRA adapter merging)
    --query         Run a single query and exit
    --show-think    Show <think> blocks in output (default: hidden)
    --temperature   Sampling temperature (default: 0.7, 0 = greedy)
    --max-tokens    Max new tokens per generation step (default: 512)
    --vm            Use scratchpad QEMU VM for code execution instead of inline exec
"""

import sys
import argparse
import logging
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.generation_loop import generate_with_execution, format_response_for_display
from src.executor.python_exec import PythonExecutor
from src.executor.vm_exec import VMExecutor, is_available as vm_available

logging.basicConfig(level=logging.WARNING)  # suppress transformers noise in interactive mode
logger = logging.getLogger(__name__)


ROCM_ENV_HINT = """
ROCm environment variables (set before running if on MI50):
    export HSA_OVERRIDE_GFX_VERSION=9.0.6
    export PYTORCH_ROCM_ARCH=gfx906
    export TOKENIZERS_PARALLELISM=false
    export HIP_VISIBLE_DEVICES=0
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
"""


def load_model(model_path: str, base_only: bool = False):
    """Load model and tokenizer. Handles both LoRA adapters and base models."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # float32 for ROCm gfx906 stability

    if base_only or not (Path(model_path) / "adapter_config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load base model name from adapter config
        import json
        adapter_cfg = json.load(open(Path(model_path) / "adapter_config.json"))
        base_name = adapter_cfg.get("base_model_name_or_path", model_path)
        print(f"  Base model: {base_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        # Training resized embeddings to fit the extended tokenizer; match that here.
        adapter_vocab_size = len(tokenizer)
        base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
        if adapter_vocab_size != base_vocab_size:
            base_model.resize_token_embeddings(adapter_vocab_size)
            print(f"  Resized embeddings: {base_vocab_size} → {adapter_vocab_size}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("  LoRA adapter merged")

    model.eval()
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}  ({used:.1f}/{vram:.1f}GB)")

    return model, tokenizer


def run_interactive(model, tokenizer, executor, args):
    """Interactive REPL."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nWorldModel ready. Type your question, or 'quit' to exit.")
    print("Commands: /reset (clear Python state), /think (toggle think visibility)")
    print("-" * 60)

    show_think = args.show_think

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if query == "/reset":
            executor.reset()
            print("Python state cleared.")
            continue

        if query == "/think":
            show_think = not show_think
            print(f"Think blocks: {'visible' if show_think else 'hidden'}")
            continue

        print()
        response = generate_with_execution(
            model=model,
            tokenizer=tokenizer,
            prompt=query,
            executor=executor,
            max_new_tokens_per_step=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )

        if show_think:
            print(response)
        else:
            print(format_response_for_display(response))


def main():
    parser = argparse.ArgumentParser(description="WorldModel inference CLI")
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--vm", action="store_true",
                        help="Use scratchpad QEMU VM for code execution")
    args = parser.parse_args()

    if args.vm:
        if not vm_available():
            print("WARNING: scratchpad-cli not found; falling back to inline exec")
            executor = PythonExecutor()
        else:
            executor = VMExecutor()
            print("Using scratchpad QEMU VM for execution")
    else:
        executor = PythonExecutor()

    model, tokenizer = load_model(args.model, args.base_only)

    if args.query:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        response = generate_with_execution(
            model=model,
            tokenizer=tokenizer,
            prompt=args.query,
            executor=executor,
            max_new_tokens_per_step=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )
        if args.show_think:
            print(response)
        else:
            print(format_response_for_display(response))
    else:
        run_interactive(model, tokenizer, executor, args)


if __name__ == "__main__":
    main()
