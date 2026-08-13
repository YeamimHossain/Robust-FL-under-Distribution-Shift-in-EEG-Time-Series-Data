"""
Thin wrapper around Flower's built-in FedAvg so we can retrieve the final
global model after start_simulation() returns. Also used for FedProx, since
FedProx in Flower is just FedAvg + the proximal_mu passed via on_fit_config_fn
(the actual proximal math lives client-side, in scripts/client.py's fit()).

Also handles round-by-round console logging and incremental results
persistence (via logging_utils.RoundLogger), if a logger is provided --
so progress (loss/accuracy/f1, which clients ran) is visible live and
survives a kernel interruption.
"""

from typing import List, Tuple, Optional, Dict
import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class TrackingFedAvg(FedAvg):
    def __init__(self, *args, round_logger=None, checkpoint_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_ndarrays = None
        self.round_logger = round_logger
        self.checkpoint_path = checkpoint_path

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_params is not None:
            self.latest_ndarrays = parameters_to_ndarrays(aggregated_params)
            if self.checkpoint_path is not None:
                import numpy as np
                np.savez(self.checkpoint_path, *self.latest_ndarrays,
                         round=server_round)
        if self.round_logger is not None and results:
            self.round_logger.record_fit_round(server_round, results)
        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss_agg, metrics_agg = super().aggregate_evaluate(
            server_round, results, failures
        )
        if self.round_logger is not None and results:
            loss_agg, metrics_agg = self.round_logger.record_evaluate_round(
                server_round, results
            )
        return loss_agg, metrics_agg
