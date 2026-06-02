"""
Sync first 100 steps from V3-grpo-n4-bs96.log to SwanLab.
Usage: python sync_to_swanlab.py [--api-key YOUR_KEY]
"""
import re
import argparse
import swanlab

def parse_step_metrics(line):
    """Parse a step line into dict of metrics."""
    metrics = {}
    # Extract step number
    step_match = re.search(r'step:(\d+)', line)
    if not step_match:
        return None, None
    step = int(step_match.group(1))

    # Extract all key:value pairs
    pairs = re.findall(r'([\w/]+):(-?[\d.]+)', line)
    for key, val in pairs:
        if key == 'step':
            continue
        try:
            metrics[key] = float(val)
        except ValueError:
            pass
    return step, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--log-file', type=str,
                       default='/root/paddlejob/workspace/new_mem1/mem1/mem1/Mem1/train/V3-grpo-n4-bs96.log')
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    if args.api_key:
        swanlab.login(api_key=args.api_key)

    # Parse log file
    steps_data = []
    with open(args.log_file, 'r') as f:
        for line in f:
            if 'timing_s/step' in line:
                step, metrics = parse_step_metrics(line)
                if step is not None and step <= args.max_steps:
                    steps_data.append((step, metrics))

    print(f"Parsed {len(steps_data)} steps from log")

    # Init swanlab run
    run = swanlab.init(
        project="MEM1",
        experiment_name="V3-curriculum-n4-bs96",
        config={
            "model": "Qwen2.5-7B",
            "algorithm": "GRPO",
            "train_batch_size": 96,
            "n_agent": 4,
            "max_turns": 6,
            "lr": 2e-7,
            "param_offload": True,
            "gpu": "A800x6",
            "curriculum": "warmup→convergence→transition→refinement",
            "data": "nq_hotpotqa_train_multi_2",
        }
    )

    # Log all steps
    for step, metrics in steps_data:
        swanlab.log(metrics, step=step)

    print(f"Synced {len(steps_data)} steps to SwanLab")
    print(f"Project: MEM1, Experiment: V3-curriculum-n4-bs96")

    swanlab.finish()

if __name__ == '__main__':
    main()
