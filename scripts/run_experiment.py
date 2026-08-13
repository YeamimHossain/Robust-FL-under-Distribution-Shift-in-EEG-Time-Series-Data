"""
Run the full federated simulation.

Usage:
    python run_experiment.py --strategy fedavg   --rounds 30 --clients 45
    python run_experiment.py --strategy fedprox  --rounds 30 --clients 45 --mu 1.0
    python run_experiment.py --strategy fedsplit --rounds 30 --clients 15   # smaller scale

This script:
  1. Loads + preprocesses each client's data
  2. Builds Flower clients
  3. Runs the chosen aggregation strategy for N rounds
  4. Evaluates on held-out UNSEEN clients at the end
  5. Saves per-round + per-client results to results/<strategy>_results.json
  6. Saves the final trained global model to results/<strategy>_model.pt
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TF/absl C++ logs
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("PYTHONWARNINGS", "ignore")    # inherited by actor processes

import warnings
warnings.filterwarnings("ignore")  # covers the driver process itself

import logging
import argparse
import json
import time
import numpy as np
import torch
import flwr as fl

logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    make_client_split, load_subject_data_local,
    train_test_split_per_client, normalize_per_client,
)
from scripts.client import EEGClient, get_params
from models.eegnet import EEGNet
from strategies.fedsplit import FedSplit
from strategies.tracking_fedavg import TrackingFedAvg
from strategies.logging_utils import RoundLogger

import ray

def preload_and_share(subject_ids, data_dir):
    """Load + preprocess every client's data ONCE, then place each into
    Ray's shared object store. Returns {subject_id: ObjectRef} plus the
    common (n_channels, n_times) shape."""
    refs = {}
    n_channels, n_times = None, None
    print(f"Loading and preprocessing {len(subject_ids)} clients once "
          f"(this is the only time each .mat file is read from disk)...")
    for i, sid in enumerate(subject_ids, 1):
        X, y = load_subject_data_local(sid, data_dir)
        X_train, y_train, X_test, y_test = train_test_split_per_client(X, y)
        X_train, X_test = normalize_per_client(X_train, X_test)
        refs[sid] = ray.put((X_train, y_train, X_test, y_test))
        if n_channels is None:
            n_channels, n_times = X.shape[1], X.shape[2]
        if i % 10 == 0 or i == len(subject_ids):
            print(f"  loaded {i}/{len(subject_ids)} clients")
    return refs, n_channels, n_times


def build_client_fn(subject_ids, data_refs, n_channels, n_times, device,
                     local_epochs, batch_size, lr=1e-3):
    def client_fn(context):
        cid = context.node_config.get("partition-id", context.node_id)
        subject_id = subject_ids[int(cid)]
        X_train, y_train, X_test, y_test = ray.get(data_refs[subject_id])
        return EEGClient(
            subject_id, X_train, y_train, X_test, y_test,
            n_channels=n_channels, n_times=n_times, device=device,
            local_epochs=local_epochs, batch_size=batch_size, lr=lr,
        ).to_client()
    return client_fn


def evaluate_on_unseen_clients(global_params, unseen_subject_ids, n_channels,
                                n_times, data_dir, device="cpu"):
    """the generalization-to-a-new-client result. Reports the
    same four metrics as training (accuracy, precision, recall, f1) so
    unseen-client results are directly comparable to in-training rounds."""
    model = EEGNet(n_channels=n_channels, n_times=n_times).to(device)
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, global_params)}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    per_subject_metrics = {}
    for sid in unseen_subject_ids:
        X, y = load_subject_data_local(sid, data_dir)
        # unseen clients: no local fine-tuning, evaluate global model as-is
        X_norm = (X - X.mean(axis=(0, 2), keepdims=True)) / (
            X.std(axis=(0, 2), keepdims=True) + 1e-6)
        y_t = torch.tensor(y, dtype=torch.long)
        with torch.no_grad():
            logits = model(torch.tensor(X_norm, dtype=torch.float32).to(device))
            preds = logits.argmax(dim=1)
            acc = (preds == y_t).float().mean().item()

            tp = ((preds == 1) & (y_t == 1)).sum().item()
            fp = ((preds == 1) & (y_t == 0)).sum().item()
            fn = ((preds == 0) & (y_t == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_subject_metrics[int(sid)] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return per_subject_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fedavg", "fedprox", "fedsplit"],
                         required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=50,
                         help="number of TRAINING clients to include "
                              "(use a smaller number for fedsplit, Step 5.2)")
    parser.add_argument("--mu", type=float, default=0.01,
                         help="proximal mu for fedprox (ignored otherwise)")
    parser.add_argument("--fedsplit_mu", type=float, default=0.1,
                         help="FedSplit's internal local proximal strength "
                              "(previously hardcoded at 1.0, never tuned -- "
                              "your own FedProx sweep showed mu=0.1 was "
                              "already too strong, so 1.0 was likely far "
                              "too strong for FedSplit too; try smaller "
                              "values like 0.01 as well)")
    parser.add_argument("--fedsplit_steps", type=int, default=3,
                         help="FedSplit's local proximal gradient steps per "
                              "round (more steps = closer to the exact "
                              "solve the original algorithm calls for, at "
                              "higher per-round cost)")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                         help="fraction of available clients sampled each "
                              "round (1.0 = all clients every round). "
                              "Lowering this is a real time-complexity "
                              "lever: fewer clients per round means less "
                              "wall-clock time per round, and is also "
                              "standard practice in real deployments where "
                              "not every client is online every round.")
    parser.add_argument("--local_epochs", type=int, default=5,
                         help="epochs over mini-batches per client per "
                              "round (previously this flag existed but "
                              "was never actually wired through -- fixed)")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="mini-batch size for local training")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="base learning rate (the fixed 'gradient "
                              "step size' -- Adam adapts around this "
                              "internally per-parameter, but this base "
                              "value itself doesn't change unless "
                              "--lr_schedule is also set)")
    parser.add_argument("--lr_schedule", action="store_true",
                         help="opt-in: halve the learning rate whenever "
                              "the round loss stops improving for "
                              "--lr_patience rounds in a row. Off by "
                              "default -- existing results are unaffected "
                              "unless you explicitly add this flag.")
    parser.add_argument("--lr_patience", type=int, default=5,
                         help="rounds of no improvement before shrinking "
                              "lr (only used if --lr_schedule is set)")
    parser.add_argument("--lr_min", type=float, default=1e-5,
                         help="floor -- lr will never shrink below this "
                              "(only used if --lr_schedule is set)")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="directory containing s1.mat, s2.mat, ... "
                              "downloaded locally from GigaDB")
    parser.add_argument("--ray_tmp_dir", type=str, default=None,
                         help="optional: redirect Ray's temp/spill directory "
                              "here (e.g. a path on a drive with more free "
                              "space than /tmp). Use this if you see 'Local "
                              "disk is full' errors referencing /tmp/ray/...")
    parser.add_argument("--actor_cpus", type=int, default=1,
                         help="CPUs reserved per simulated client. Flower "
                              "creates roughly (total CPUs / this number) "
                              "parallel actors. Raise this (e.g. to 2 or 4) "
                              "on a memory-constrained machine to reduce how "
                              "many clients run simultaneously.")
    args = parser.parse_args()

    training_clients, unseen_clients = make_client_split()
    n_available = len(training_clients)
    if args.clients > n_available:
        print(f"[warning] --clients {args.clients} requested, but only "
              f"{n_available} training clients are available "
              f"({len(unseen_clients)} of the 50 usable subjects are held "
              f"out as unseen/generalization clients). Using all "
              f"{n_available} available training clients instead. "
              f"Report {n_available} training clients in your paper, "
              f"not {args.clients}.")
    training_clients = training_clients[:args.clients]

    ray_init_args = {"ignore_reinit_error": True, "include_dashboard": False}
    if args.ray_tmp_dir:
        os.makedirs(args.ray_tmp_dir, exist_ok=True)
        ray_init_args["_temp_dir"] = args.ray_tmp_dir
    ray.init(**ray_init_args)

    data_refs, n_channels, n_times = preload_and_share(training_clients, args.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client_fn = build_client_fn(training_clients, data_refs, n_channels,
                                 n_times, device, args.local_epochs,
                                 args.batch_size, args.lr)

    init_model = EEGNet(n_channels=n_channels, n_times=n_times)
    init_params = fl.common.ndarrays_to_parameters(get_params(init_model))

    lr_state = {"current_lr": args.lr, "best_loss": None, "bad_rounds": 0}

    def maybe_shrink_lr(latest_loss):
        if not args.lr_schedule or latest_loss is None:
            return
        if lr_state["best_loss"] is None or latest_loss < lr_state["best_loss"] - 1e-4:
            lr_state["best_loss"] = latest_loss
            lr_state["bad_rounds"] = 0
        else:
            lr_state["bad_rounds"] += 1
            if lr_state["bad_rounds"] >= args.lr_patience:
                old_lr = lr_state["current_lr"]
                lr_state["current_lr"] = max(old_lr * 0.5, args.lr_min)
                lr_state["bad_rounds"] = 0
                print(f"[lr_schedule] loss plateaued for {args.lr_patience} "
                      f"rounds -- reducing lr {old_lr:.6f} -> "
                      f"{lr_state['current_lr']:.6f}", flush=True)

    def fit_config_fn(server_round):
        config = {"server_round": server_round, "lr": lr_state["current_lr"]}
        if args.strategy == "fedprox":
            config["proximal_mu"] = args.mu
        else:
            config["proximal_mu"] = 0.0
        return config

    def evaluate_config_fn(server_round):
        return {"server_round": server_round}

    def weighted_average(metrics):
        accs = [n * m["accuracy"] for n, m in metrics]
        total = sum(n for n, _ in metrics)
        return {"accuracy": sum(accs) / total}

    start_time = time.time()

    os.makedirs("results", exist_ok=True)
    progress_path = f"results/{args.strategy}_progress.json"
    round_logger = RoundLogger(
        results_path=progress_path,
        on_round_end=maybe_shrink_lr,
        run_metadata={
            "strategy": args.strategy,
            "rounds_planned": args.rounds,
            "n_training_clients": len(training_clients),
            "n_unseen_clients": len(unseen_clients),
            "mu": args.mu if args.strategy == "fedprox" else None,
            "fedsplit_mu": args.fedsplit_mu if args.strategy == "fedsplit" else None,
            "fedsplit_steps": args.fedsplit_steps if args.strategy == "fedsplit" else None,
            "fraction_fit": args.fraction_fit,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
        },
    )
    print(f"Live progress will be saved after every round to {progress_path} "
          f"-- if this run gets interrupted, whatever rounds completed so "
          f"far will still be there.\n")

    checkpoint_path = f"results/{args.strategy}_checkpoint.npz"

    if args.strategy in ("fedavg", "fedprox"):
        strategy = TrackingFedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=max(2, int(len(training_clients) * args.fraction_fit)),
            min_available_clients=len(training_clients),
            initial_parameters=init_params,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            evaluate_metrics_aggregation_fn=weighted_average,
            round_logger=round_logger,
            checkpoint_path=checkpoint_path,
        )
    else:  # fedsplit
        strategy = FedSplit(
            initial_parameters=init_params,
            fraction_fit=args.fraction_fit,
            min_fit_clients=max(2, int(len(training_clients) * args.fraction_fit)),
            min_available_clients=len(training_clients),
            local_prox_steps=args.fedsplit_steps,
            fedsplit_mu=args.fedsplit_mu,
            round_logger=round_logger,
            checkpoint_path=checkpoint_path,
        )
    print(f"A model checkpoint will be saved to {checkpoint_path} after "
          f"every round -- if interrupted, load it with np.load() to get "
          f"the latest completed round's weights (see README).\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(training_clients),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.actor_cpus, "num_gpus": 0},
        keep_initialised=True,  # reuse OUR already-initialized Ray session
                                  # (with data_refs already in the shared
                                  # object store) instead of Flower
                                  # shutting it down and starting fresh
    )

    elapsed = time.time() - start_time

    # Final global params for unseen-client evaluation (Step 6.3)
    if args.strategy == "fedsplit":
        final_params = strategy.global_params
    else:
        final_params = strategy.latest_ndarrays

    if final_params is None:
        raise RuntimeError(
            "No global model was produced -- every client's fit/evaluate "
            "call failed in every round (check the printed Flower/Ray logs "
            "above for the actual per-client error; look for 'ERROR' or "
            "'received 0 results and N failures'). Fix that root cause "
            "first; this is not a bug in the final-model retrieval step."
        )

    unseen_metrics = evaluate_on_unseen_clients(
        final_params, unseen_clients, n_channels, n_times, args.data_dir, device
    )

    model_path = f"results/{args.strategy}_model.pt"
    final_model = EEGNet(n_channels=n_channels, n_times=n_times)
    keys = list(final_model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, final_params)}
    final_model.load_state_dict(state_dict, strict=True)
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "n_channels": n_channels,
        "n_times": n_times,
        "strategy": args.strategy,
        "rounds": args.rounds,
        "mu": args.mu if args.strategy == "fedprox" else None,
    }, model_path)
    print(f"Saved trained model to {model_path}")

    unseen_summary = {}
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        values = [m[metric_name] for m in unseen_metrics.values()]
        unseen_summary[f"unseen_mean_{metric_name}"] = float(np.mean(values))
        unseen_summary[f"unseen_std_{metric_name}"] = float(np.std(values))

    out = {
        "strategy": args.strategy,
        "rounds": args.rounds,
        "n_training_clients": len(training_clients),
        "n_unseen_clients": len(unseen_clients),
        "wall_clock_seconds": elapsed,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed": history.metrics_distributed,
        "unseen_client_metrics": unseen_metrics,
        **unseen_summary,
        "model_path": model_path,
    }
    out_path = f"results/{args.strategy}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Results saved to {out_path}")
    print(f"Unseen-client mean accuracy:  {out['unseen_mean_accuracy']:.4f} (+/- {out['unseen_std_accuracy']:.4f})")
    print(f"Unseen-client mean precision: {out['unseen_mean_precision']:.4f} (+/- {out['unseen_std_precision']:.4f})")
    print(f"Unseen-client mean recall:    {out['unseen_mean_recall']:.4f} (+/- {out['unseen_std_recall']:.4f})")
    print(f"Unseen-client mean f1:        {out['unseen_mean_f1']:.4f} (+/- {out['unseen_std_f1']:.4f})")


if __name__ == "__main__":
    main()
