#!/usr/bin/env python3
"""
Run LinUCB with multiple alpha values and compare the evaluation metrics.

This script sequentially launches `examples/usermodel/run_LinUCB.py` for each
alpha, parses the generated log file to collect the final (Epoch 0) metrics,
persists them to JSON, and produces a comparison plot.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "saved_models" / "MovieLensEnv-v0" / "LinUCB" / "logs"
RESULT_DIR = ROOT / "visual_results"


def run_linucb(alpha: float, base_args: List[str], message_prefix: str) -> Path:
    """Launch LinUCB training for a single alpha and return the log path."""
    msg = f"{message_prefix}-alpha-{str(alpha).replace('.', 'p')}"
    cmd = [
        sys.executable,
        "examples/usermodel/run_LinUCB.py",
        "--env",
        "MovieLensEnv-v0",
        "--seed",
        "2023",
        "--epoch",
        "1",
        "--batch_size",
        "512",
        "--num_workers",
        "0",
        "--ucb_alpha",
        str(alpha),
        "--message",
        msg,
    ]
    if base_args:
        cmd.extend(base_args)

    print(f"\n=== Running LinUCB with alpha={alpha} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)

    target_prefix = f"[{msg}]"
    matches = sorted(
        (p for p in LOG_DIR.glob("*.log") if p.name.startswith(target_prefix)),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No log file found for message '{msg}'.")
    return matches[-1]


def parse_metrics(log_path: Path) -> Dict[str, float]:
    """Parse the final Epoch [0] metrics dict from a log file."""
    content = log_path.read_text()
    matches = re.findall(r"Epoch:\s*\[0\],\s*Info:\s*(\[.*?\])", content, re.S)
    if not matches:
        raise ValueError(f"Could not find Epoch [0] metrics in {log_path}")

    raw = matches[-1]
    cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
    metrics_list = ast.literal_eval(cleaned)
    if not metrics_list:
        raise ValueError(f"No metrics parsed from {log_path}")
    metrics = metrics_list[0]
    return {
        "loss": float(metrics["loss"]),
        "MAE": float(metrics["MAE"]),
        "ctr": float(metrics["ctr"]),
        "click_loss": float(metrics["click_loss"]),
    }


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    """Create a comparison plot for CTR and click loss across alphas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(df.index, df["ctr"], marker="o")
    axes[0].set_title("CTR vs Alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("CTR (average reward)")

    axes[1].plot(df.index, df["click_loss"], marker="o", color="C3")
    axes[1].set_title("Click Loss vs Alpha")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Average |prediction - reward|")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LinUCB alpha values.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0],
        help="Alpha values to evaluate.",
    )
    parser.add_argument(
        "--message-prefix",
        type=str,
        default="linucb-sweep",
        help="Prefix for run messages/log files.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to run_LinUCB.py.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=RESULT_DIR / "linucb_alpha_results.json",
        help="Where to save the aggregated metrics.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=RESULT_DIR / "linucb_alpha_comparison.png",
        help="Where to save the comparison plot.",
    )
    args = parser.parse_args()

    metrics_by_alpha = {}
    for alpha in args.alphas:
        log_path = run_linucb(alpha, args.extra_args or [], args.message_prefix)
        metrics = parse_metrics(log_path)
        metrics_by_alpha[str(alpha)] = metrics
        print(f"Alpha {alpha}: {metrics}")

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(metrics_by_alpha, indent=2))
    print(f"\nSaved metrics to {args.results_json}")

    df = pd.DataFrame(metrics_by_alpha).T.astype(float)
    df.index = df.index.astype(float)
    df = df.sort_index()
    plot_results(df, args.plot_path)


if __name__ == "__main__":
    main()
