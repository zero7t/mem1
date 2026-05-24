#!/usr/bin/env python3
"""
Parse grpo_full.log and import metrics into SwanLab as a new offline run.

Usage:
    python import_log_to_swanlab.py [log_file] [experiment_name]

Default:
    log_file = /root/paddlejob/workspace/mem1/MEM1/logs/grpo_full.log
    experiment_name = BASELINE-2D-GRPO-bs60
"""

import re
import sys
import os

def parse_step_metrics(line):
    """Parse a step: line into (step_num, metrics_dict)."""
    # Extract step number
    step_match = re.search(r'step:(\d+)', line)
    if not step_match:
        return None, None
    step = int(step_match.group(1))

    # Extract all key:value pairs
    metrics = {}
    # Pattern: key:value where value is a number (int or float)
    pairs = re.findall(r'(\S+?):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
    for key, value in pairs:
        if key == 'step':
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return step, metrics


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/root/paddlejob/workspace/mem1/MEM1/logs/grpo_full.log"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "BASELINE-2D-GRPO-bs60"

    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing: {log_file}")
    print(f"Experiment: {experiment_name}")

    # Parse all steps
    all_steps = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'step:' in line and 'timing_s' in line:
                step, metrics = parse_step_metrics(line)
                if step is not None and metrics:
                    all_steps.append((step, metrics))

    print(f"Found {len(all_steps)} steps")

    if not all_steps:
        print("No steps found in log file!")
        sys.exit(1)

    # Print first and last step for verification
    print(f"  First step: {all_steps[0][0]}")
    print(f"  Last step:  {all_steps[-1][0]}")
    print(f"  Sample metrics: {list(all_steps[0][1].keys())[:10]}...")

    # Import to SwanLab
    try:
        import swanlab
    except ImportError:
        print("Error: swanlab not installed. Run: pip install swanlab")
        sys.exit(1)

    print(f"\nInitializing SwanLab (offline mode)...")
    swanlab.init(
        project="MEM1",
        experiment_name=experiment_name,
        config={
            "source": "imported_from_log",
            "log_file": log_file,
            "total_steps": len(all_steps),
            "batch_size": 60,
            "model": "Qwen2.5-7B",
            "improvements": "none (baseline)",
        },
        mode="offline"
    )

    print(f"Logging {len(all_steps)} steps to SwanLab...")
    for step, metrics in all_steps:
        swanlab.log(data=metrics, step=step)

    swanlab.finish()
    print(f"\nDone! Imported {len(all_steps)} steps into SwanLab experiment '{experiment_name}'")
    print(f"SwanLab logs saved in: {os.getcwd()}/swanlog/")
    print(f"\nTo view: swanlab watch swanlog/")


if __name__ == "__main__":
    main()
