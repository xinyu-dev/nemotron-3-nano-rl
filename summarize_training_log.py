#!/usr/bin/env python3
"""Summarize training throughput, FLOPS, and MFU from a GRPO training output log.

Usage:
    python summarize_training_log.py <log_file> [--warmup-steps N]
    e.g.
    python summarize_training_log.py /workspace/nemotron-3-nano-rl/logs/exp_002/wandb/wandb/latest-run/files/output.log --warmup-steps 4

Arguments:
    log_file        Path to the output.log file
    --warmup-steps  Number of initial steps to exclude (default: 1)
"""

import argparse
import re
from pathlib import Path

# ANSI color codes
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
ORANGE  = "\033[38;5;208m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"

PATTERNS = {
    "e2e_tokens_per_sec_gpu":              r"E2E \(Tokens/sec/gpu\):\s+([\d.]+)",
    "policy_training_tokens_per_sec_gpu":  r"Policy Training \(Tokens/sec/gpu\):\s+([\d.]+)",
    "logprobs_tokens_per_sec_gpu":         r"Policy and Reference Logprobs \(Tokens/sec/gpu\):\s+([\d.]+)",
    "training_group_tokens_per_sec_gpu":   r"Training Worker Group \(Tokens/sec/gpu\):\s+([\d.]+)",
    "generation_group_tokens_per_sec_gpu": r"Generation Worker Group \(Tokens/sec/gpu\):\s+([\d.]+)",
    "training_flops_tflops":               r"Training FLOPS:\s+([\d.]+) TFLOPS",
    "training_flops_per_rank":             r"Training FLOPS:.*\(([\d.]+) TFLOPS per rank\)",
    "mfu_pct":                             r"Training Model Floating Point Utilization:\s+([\d.]+)%",
    "mean_gen_length":                     r"Mean Generation Length:\s+([\d.]+)",
}


def parse_log(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    block_starts = [m.start() for m in re.finditer(r"📊 Training Results:", text)]
    if not block_starts:
        raise ValueError("No training steps found in log file.")

    steps = []
    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] if i + 1 < len(block_starts) else len(text)
        block = text[start:end]
        metrics = {}
        for key, pattern in PATTERNS.items():
            m = re.search(pattern, block)
            if m:
                metrics[key] = float(m.group(1))
        if metrics:
            metrics["step"] = i + 1
            steps.append(metrics)
    return steps


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(steps: list[dict], warmup_steps: int):
    stable_steps = [s for s in steps if s["step"] > warmup_steps]

    def avg_metric(key, src=stable_steps):
        vals = [s[key] for s in src if key in s]
        return average(vals)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'━'*72}{RESET}")
    print(f"{BOLD}{CYAN}  📊  Training Log Summary{RESET}")
    print(f"{BOLD}{CYAN}{'━'*72}{RESET}")
    print(f"  {WHITE}Total steps found{RESET} : {BOLD}{len(steps)}{RESET}")
    print(f"  {WHITE}Warmup excluded  {RESET} : {BOLD}{len(steps) - len(stable_steps)} step(s){RESET}")
    print(f"  {WHITE}Steps averaged   {RESET} : {BOLD}{len(stable_steps)} "
          f"(steps {warmup_steps+1}–{steps[-1]['step']}){RESET}")
    print(f"{BOLD}{CYAN}{'━'*72}{RESET}\n")

    # ── Per-step table ────────────────────────────────────────────────────────
    h1 = f"{'Step':>5}  {'E2E':>8}  {'Train':>9}  {'Logprob':>8}  {'TrainGrp':>9}  {'GenGrp':>8}  {'TFLOPS':>8}  {'TFLOPS/rank':>11}  {'MFU%':>7}"
    print(f"{BOLD}{WHITE}{h1}{RESET}")
    print(f"{GRAY}{'─'*len(h1)}{RESET}")

    for s in steps:
        is_warmup = s["step"] <= warmup_steps
        c = GRAY if is_warmup else WHITE
        marker = f"{GRAY} *{RESET}" if is_warmup else "  "

        print(
            f"{c}{BOLD}{s['step']:>5}{RESET}{marker}"
            f"  {c}{s.get('e2e_tokens_per_sec_gpu', float('nan')):>8.1f}{RESET}"
            f"  {c}{s.get('policy_training_tokens_per_sec_gpu', float('nan')):>9.1f}{RESET}"
            f"  {c}{s.get('logprobs_tokens_per_sec_gpu', float('nan')):>8.1f}{RESET}"
            f"  {c}{s.get('training_group_tokens_per_sec_gpu', float('nan')):>9.1f}{RESET}"
            f"  {c}{s.get('generation_group_tokens_per_sec_gpu', float('nan')):>8.1f}{RESET}"
            f"  {c}{s.get('training_flops_tflops', float('nan')):>8.1f}{RESET}"
            f"  {c}{s.get('training_flops_per_rank', float('nan')):>11.2f}{RESET}"
            f"  {c}{BOLD}{s.get('mfu_pct', float('nan')):>7.2f}{RESET}"
        )

    print(f"\n  {GRAY}* = warmup step(s), excluded from averages{RESET}\n")

    # ── Averages ──────────────────────────────────────────────────────────────
    print(f"{BOLD}{CYAN}{'━'*72}{RESET}")
    print(f"{BOLD}{CYAN}  ⚡  Averages  (steps {warmup_steps+1}–{steps[-1]['step']}){RESET}")
    print(f"{BOLD}{CYAN}{'━'*72}{RESET}")

    def row(label, value, unit=""):
        print(f"    {WHITE}{label:<38}{RESET}{BOLD}{value}{RESET} {GRAY}{unit}{RESET}")

    print(f"\n  {BOLD}{CYAN}🚀 Throughputs (per GPU){RESET}")
    row("E2E Tokens/sec/gpu",              f"{avg_metric('e2e_tokens_per_sec_gpu'):>8.2f}", "tok/s/gpu")
    row("Policy Training Tokens/sec/gpu",  f"{avg_metric('policy_training_tokens_per_sec_gpu'):>8.2f}", "tok/s/gpu")
    row("Logprobs Tokens/sec/gpu",         f"{avg_metric('logprobs_tokens_per_sec_gpu'):>8.2f}", "tok/s/gpu")
    row("Training Worker Group T/s/gpu",   f"{avg_metric('training_group_tokens_per_sec_gpu'):>8.2f}", "tok/s/gpu")
    row("Generation Worker Group T/s/gpu", f"{avg_metric('generation_group_tokens_per_sec_gpu'):>8.2f}", "tok/s/gpu")

    print(f"\n  {BOLD}{CYAN}⚙️  Training FLOPS{RESET}")
    row("Total TFLOPS",                    f"{avg_metric('training_flops_tflops'):>8.2f}", "TFLOPS")
    row("TFLOPS per rank",                 f"{avg_metric('training_flops_per_rank'):>8.2f}", "TFLOPS/rank")

    print(f"\n  {BOLD}{CYAN}🎯 Model FLOP Utilization (MFU){RESET}")
    row("Avg MFU",                         f"{avg_metric('mfu_pct'):>8.2f}", "%")

    print(f"\n  {BOLD}{CYAN}📝 Generation Length{RESET}")
    row("Avg Output Tokens",               f"{avg_metric('mean_gen_length'):>8.1f}", "tokens")
    all_lens = [s["mean_gen_length"] for s in stable_steps if "mean_gen_length" in s]
    if all_lens:
        row("Min / Max",                   f"{min(all_lens):>8.1f} / {max(all_lens):.1f}", "tokens")
    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize GRPO training log metrics.")
    parser.add_argument("log_file", type=str, help="Path to output.log")
    parser.add_argument(
        "--warmup-steps", type=int, default=1,
        help="Number of initial steps to exclude from averages (default: 1)"
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    steps = parse_log(log_path)
    summarize(steps, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
