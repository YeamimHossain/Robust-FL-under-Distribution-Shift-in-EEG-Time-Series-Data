"""
Flower client. One instance of this class = one subject
(one federated client). Handles the FedProx proximal term directly in the
local training loop, controlled by a `proximal_mu` config value sent from
the server each round -- setting mu=0 makes this behave as plain FedAvg,
which is how we produce the FedAvg baseline without a separate codepath.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple

from models.eegnet import EEGNet


def get_params(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_params(model: nn.Module, params: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)


class EEGClient(fl.client.NumPyClient):
    def __init__(self, subject_id, X_train, y_train, X_test, y_test,
                 n_channels, n_times, device="cpu", local_epochs=2, lr=1e-3,
                 batch_size=32):
        self.subject_id = int(subject_id)
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.model = EEGNet(n_channels=n_channels, n_times=n_times).to(device)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_params(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]
            ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_params(self.model, parameters)

        server_round = config.get("server_round", "?")

        anchor_params = copy.deepcopy(list(self.model.parameters()))

        fedsplit_mode = bool(config.get("fedsplit_mode", False))
        if fedsplit_mode:
            mu = float(config.get("fedsplit_mu", 0.1))
            n_steps = int(config.get("local_prox_steps", 3))
        else:
            mu = float(config.get("proximal_mu", 0.0))  # 0 -> plain FedAvg
            n_steps = self.local_epochs

        print(f"  [Round {server_round}] Client (subject {self.subject_id}) "
              f"training: {n_steps} local epochs, "
              f"{len(self.X_train)} trials, batch_size={self.batch_size}",
              flush=True)

        
        effective_lr = float(config.get("lr", self.lr))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=effective_lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        n = len(self.X_train)
        bs = min(self.batch_size, n)

        running_loss_sum = 0.0
        running_loss_count = 0
        for _ in range(n_steps):
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                X = self.X_train[idx].to(self.device)
                y = self.y_train[idx].to(self.device)

                optimizer.zero_grad()
                logits = self.model(X)
                loss = criterion(logits, y)

                if mu > 0:
                    prox_term = 0.0
                    for w, w_anchor in zip(self.model.parameters(), anchor_params):
                        prox_term += torch.sum((w - w_anchor) ** 2)
                    loss = loss + (mu / 2.0) * prox_term

                loss.backward()
                optimizer.step()

                running_loss_sum += loss.item() * len(idx)
                running_loss_count += len(idx)

        train_loss = running_loss_sum / max(1, running_loss_count)

        return get_params(self.model), len(self.X_train), {
            "subject_id": self.subject_id,
            "train_loss": float(train_loss),
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
                 ) -> Tuple[float, int, Dict[str, Scalar]]:
        set_params(self.model, parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        server_round = config.get("server_round", "?")

        with torch.no_grad():
            X = self.X_test.to(self.device)
            y = self.y_test.to(self.device)
            logits = self.model(X)
            loss = criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            tp = ((preds == 1) & (y == 1)).sum().item()
            fp = ((preds == 1) & (y == 0)).sum().item()
            fn = ((preds == 0) & (y == 1)).sum().item()
            tn = ((preds == 0) & (y == 0)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  [Round {server_round}] Client (subject {self.subject_id}) "
              f"evaluating: loss={loss:.4f} acc={acc:.4f} "
              f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}",
              flush=True)

        return loss, len(self.X_test), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "subject_id": self.subject_id,
        }
