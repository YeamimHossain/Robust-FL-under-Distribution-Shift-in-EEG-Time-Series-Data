"""
Load saved results from each strategy and produce the comparison
table/plots for your paper -- convergence curves, per-client accuracy
distribution, unseen-client generalization, and wall-clock time.

Usage:
    python scripts/compare_results.py
(run after you have results/fedavg_results.json, fedprox_results.json,
 and fedsplit_results.json from run_experiment.py)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["fedavg", "fedprox", "fedsplit"]
RESULTS_DIR = "results"


def load_results():
    results = {}
    for strat in STRATEGIES:
        path = os.path.join(RESULTS_DIR, f"{strat}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[strat] = json.load(f)
        else:
            print(f"[skip] {path} not found -- run run_experiment.py --strategy {strat} first")
    return results


def print_summary_table(results):
    print(f"{'Strategy':<10} {'Rounds':<8} {'Wall(s)':<10} "
          f"{'Unseen Mean Acc':<18} {'Unseen Std':<12}")
    for strat, r in results.items():
        print(f"{strat:<10} {r['rounds']:<8} {r['wall_clock_seconds']:<10.1f} "
              f"{r['unseen_mean_accuracy']:<18.4f} {r['unseen_std_accuracy']:<12.4f}")


def plot_convergence(results):
    plt.figure(figsize=(6, 4))
    for strat, r in results.items():
        rounds = [x[0] for x in r["history_losses_distributed"]]
        losses = [x[1] for x in r["history_losses_distributed"]]
        plt.plot(rounds, losses, marker="o", label=strat)
    plt.xlabel("Communication round")
    plt.ylabel("Distributed evaluation loss")
    plt.title("Convergence comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_comparison.png"), dpi=150)
    print(f"Saved {RESULTS_DIR}/convergence_comparison.png")


def plot_unseen_client_distribution(results):
    plt.figure(figsize=(6, 4))
    data, labels = [], []
    for strat, r in results.items():
        if "unseen_client_accuracy" in r:
            accuracies = list(r["unseen_client_accuracy"].values())
        elif "unseen_client_metrics" in r:
            accuracies = [metrics["accuracy"]
                          for metrics in r["unseen_client_metrics"].values()
                          if "accuracy" in metrics]
        else:
            print(f"[skip] {strat} has no per-unseen-client accuracy data")
            continue

        if not accuracies:
            print(f"[skip] {strat} has no usable per-unseen-client accuracy data")
            continue

        data.append(accuracies)
        labels.append(strat)

    if not data:
        plt.close()
        print("[skip] No per-unseen-client accuracy data available for spread plot")
        return

    plt.boxplot(data, labels=labels)
    plt.ylabel("Accuracy on unseen clients")
    plt.title("Per-client generalization spread (Step 6.2/6.3)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unseen_client_spread.png"), dpi=150)
    print(f"Saved {RESULTS_DIR}/unseen_client_spread.png")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results found yet.")
    else:
        print_summary_table(results)
        plot_convergence(results)
        plot_unseen_client_distribution(results)
