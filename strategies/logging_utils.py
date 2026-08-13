"""
Shared round-by-round console logging + incremental results persistence,
used by both TrackingFedAvg (FedAvg/FedProx) and FedSplit strategies.
"""

import json
import os
import time


def weighted_average(items, key):
    """items: list of (num_examples, metrics_dict). Weighted mean of
    metrics_dict[key] across items, weighted by num_examples."""
    total = sum(n for n, _ in items)
    if total == 0:
        return 0.0
    return sum(n * m.get(key, 0.0) for n, m in items) / total


class RoundLogger:
    def __init__(self, results_path, run_metadata=None, on_round_end=None):
        self.results_path = results_path
        self.run_metadata = run_metadata or {}
        self.start_time = time.time()
        self.rounds = []  # list of per-round dicts, appended in order
        # Optional callback(avg_loss) invoked after every completed round --
        # used by the opt-in loss-adaptive LR scheduler (Q1) to decide
        # whether to shrink the learning rate, without RoundLogger needing
        # to know anything about learning rates itself.
        self.on_round_end = on_round_end
        os.makedirs(os.path.dirname(self.results_path) or ".", exist_ok=True)

    def _persist(self):
        payload = {
            **self.run_metadata,
            "elapsed_seconds_so_far": time.time() - self.start_time,
            "rounds_completed": len(self.rounds),
            "rounds": self.rounds,
        }
        # Write to a temp file then replace -- avoids a half-written JSON
        # file if interrupted mid-write.
        tmp_path = self.results_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.results_path)

    def record_fit_round(self, server_round, results):
        """results: list of (ClientProxy, FitRes) from aggregate_fit."""
        client_ids = sorted(
            r.metrics.get("subject_id", "?") for _, r in results
        )
        items = [(r.num_examples, r.metrics) for _, r in results]
        train_loss = weighted_average(items, "train_loss")

        print(f"[Round {server_round}] FIT complete -- "
              f"{len(results)} clients: {client_ids} -- "
              f"avg train_loss={train_loss:.4f}", flush=True)

        self._pending_fit = {
            "round": server_round,
            "phase": "fit",
            "n_clients": len(results),
            "client_ids": client_ids,
            "train_loss": train_loss,
        }

    def record_evaluate_round(self, server_round, results):
        """results: list of (ClientProxy, EvaluateRes) from aggregate_evaluate."""
        client_ids = sorted(
            r.metrics.get("subject_id", "?") for _, r in results
        )
        items = [(r.num_examples, r.metrics) for _, r in results]
        avg_loss = sum(r.num_examples * r.loss for _, r in results) / max(
            1, sum(r.num_examples for _, r in results)
        )
        avg_acc = weighted_average(items, "accuracy")
        avg_precision = weighted_average(items, "precision")
        avg_recall = weighted_average(items, "recall")
        avg_f1 = weighted_average(items, "f1")

        elapsed = time.time() - self.start_time
        print(f"[Round {server_round}] EVAL complete -- "
              f"{len(results)} clients: {client_ids} -- "
              f"loss={avg_loss:.4f} acc={avg_acc:.4f} "
              f"precision={avg_precision:.4f} recall={avg_recall:.4f} "
              f"f1={avg_f1:.4f} (elapsed {elapsed/60:.1f} min)", flush=True)

        record = getattr(self, "_pending_fit", {
            "round": server_round, "phase": "fit_missing"
        })
        record.update({
            "eval_n_clients": len(results),
            "eval_client_ids": client_ids,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        })
        self.rounds.append(record)
        self._persist()

        if self.on_round_end is not None:
            self.on_round_end(avg_loss)

        return avg_loss, {"accuracy": avg_acc, "precision": avg_precision,
                           "recall": avg_recall, "f1": avg_f1}
