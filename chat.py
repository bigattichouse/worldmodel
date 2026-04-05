#!/usr/bin/env python3
"""
WorldModel Chat
===============
Interactive chat with a WorldModel fine-tuned on Qwen3-1.7B.
The model reasons with <think> blocks, writes Python code, and the
runtime executes it — injecting real results back into the response.

Usage:
    python chat.py                          # auto-detect latest model
    python chat.py --model ./output/worldmodel_v2/final
    python chat.py --show-think             # show internal reasoning
    python chat.py --vm                     # use QEMU sandbox for execution

Commands during chat:
    /think      toggle visibility of <think> blocks
    /reset      clear Python execution state (variable namespace)
    quit        exit
"""

import sys
import json
import argparse
import logging
import warnings
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

BASE_MODEL = Path.home() / "workspace/model/Qwen3-1.7B"
OUTPUT_DIR = Path(__file__).parent / "output"


def find_latest_model() -> Path:
    """Return the most recently modified 'final' adapter directory."""
    candidates = sorted(
        OUTPUT_DIR.glob("*/final"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {OUTPUT_DIR}.\n"
            "Run ./train_rocm.sh to train one first."
        )
    return candidates[0]


def load_model(model_path: Path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    adapter_cfg_path = model_path / "adapter_config.json"
    is_lora = adapter_cfg_path.exists()

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # float32 for ROCm gfx906 stability

    if is_lora:
        cfg = json.loads(adapter_cfg_path.read_text())
        base_name = cfg.get("base_model_name_or_path", str(BASE_MODEL))
        print(f"  Base model : {base_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=dtype, device_map="auto", trust_remote_code=True
        )
        # Match the vocab size used during training (resize may have shrunk it)
        adapter_vocab = len(tokenizer)
        base_vocab = base.get_input_embeddings().weight.shape[0]
        if adapter_vocab != base_vocab:
            base.resize_token_embeddings(adapter_vocab)
            print(f"  Embeddings : {base_vocab} → {adapter_vocab}")
        model = PeftModel.from_pretrained(base, str(model_path))
        model = model.merge_and_unload()
        print("  LoRA merged")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, device_map="auto", trust_remote_code=True
        )

    model.eval()
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used  = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU        : {torch.cuda.get_device_name(0)}  "
              f"({vram_used:.1f}/{vram_total:.1f} GB)")
    return model, tokenizer, device


def make_executor(use_vm: bool):
    from src.executor.python_exec import PythonExecutor
    from src.executor.vm_exec import VMExecutor, is_available as vm_available

    if use_vm:
        if vm_available():
            print("  Executor   : QEMU scratchpad VM")
            return VMExecutor()
        print("  Executor   : scratchpad-cli not found, falling back to inline exec")
    return PythonExecutor()


def run(args):
    from src.inference.generation_loop import (
        generate_with_execution,
        format_response_for_display,
    )

    model_path = Path(args.model) if args.model else find_latest_model()
    model, tokenizer, device = load_model(model_path)
    executor = make_executor(args.vm)

    show_think = args.show_think
    print()
    print("WorldModel ready. Type a question, or a command:")
    print("  /think  — toggle <think> block visibility")
    print("  /reset  — clear Python variable state")
    print("  quit    — exit")
    print("-" * 60)

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
        if query == "/think":
            show_think = not show_think
            print(f"Think blocks: {'visible' if show_think else 'hidden'}")
            continue
        if query == "/reset":
            executor.reset()
            print("Python state cleared.")
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

        print(response if show_think else format_response_for_display(response))


def main():
    parser = argparse.ArgumentParser(
        description="WorldModel interactive chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to fine-tuned model directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--show-think", action="store_true",
        help="Show <think> reasoning blocks (hidden by default)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature — 0 for greedy/deterministic (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per generation step (default: 512)",
    )
    parser.add_argument(
        "--vm", action="store_true",
        help="Use QEMU scratchpad VM for code execution instead of inline exec",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
