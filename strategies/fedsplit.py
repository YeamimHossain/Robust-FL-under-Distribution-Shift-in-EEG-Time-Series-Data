"""
FedSplit, implemented as a custom Flower Strategy.

There is no built-in FedSplit in Flower/FedML, so this is written from
scratch based on the Peaceman-Rachford splitting update in Pathak & Wainwright
(2020), "FedSplit: An algorithmic framework for fast federated optimization."


Per-client update:
    x_i^{t+1} = prox_{f_i}(2*x^t - z_i^t)      # approximated via K local SGD steps
    z_i^{t+1} = z_i^t + x_i^{t+1} - x^t         # reflection update
Server update:
    x^{t+1}   = average_i( z_i^{t+1} )
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


def _add(a: NDArrays, b: NDArrays, scale_b: float = 1.0) -> NDArrays:
    return [x + scale_b * y for x, y in zip(a, b)]


def _sub(a: NDArrays, b: NDArrays) -> NDArrays:
    return [x - y for x, y in zip(a, b)]


def _scale(a: NDArrays, s: float) -> NDArrays:
    return [x * s for x in a]


class FedSplit(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters,
                 fraction_fit: float = 1.0,
                 min_fit_clients: int = 2,
                 min_available_clients: int = 2,
                 local_prox_steps: int = 3,
                 fedsplit_mu: float = 0.1,
                 round_logger=None,
                 checkpoint_path=None):
        super().__init__()
        self.global_params: NDArrays = parameters_to_ndarrays(initial_parameters)
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.local_prox_steps = local_prox_steps
        self.fedsplit_mu = fedsplit_mu
        self.round_logger = round_logger
        self.checkpoint_path = checkpoint_path
        # z_i state per client, keyed by client id -- persists across rounds,
        # which is what makes this "splitting" rather than plain averaging.
        self.z_state: Dict[str, NDArrays] = {}

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(self.global_params)

    def configure_fit(self, server_round, parameters, client_manager):
        sample_size = max(self.min_fit_clients,
                           int(self.fraction_fit * client_manager.num_available()))
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )

        x_t = parameters_to_ndarrays(parameters)
        fit_configs = []
        for client in clients:
            cid = client.cid
            if cid not in self.z_state:
                self.z_state[cid] = [p.copy() for p in x_t]
            z_i = self.z_state[cid]

            # target = 2*x_t - z_i  -- this is what the client trains "toward"
            reflect_target = _sub(_scale(x_t, 2.0), z_i)

            config = {
                "fedsplit_mode": True,
                "local_prox_steps": self.local_prox_steps,
                "fedsplit_mu": self.fedsplit_mu,
                "server_round": server_round,
            }
            fit_configs.append((
                client,
                fl.common.FitIns(ndarrays_to_parameters(reflect_target), config)
            ))
        return fit_configs

    def aggregate_fit(self, server_round, results: List[Tuple[ClientProxy, FitRes]],
                       failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        x_t = self.global_params
        z_new_list = []

        for client, fit_res in results:
            cid = client.cid
            x_i_new = parameters_to_ndarrays(fit_res.parameters)  # prox_{f_i}(2x_t - z_i)
            z_i = self.z_state[cid]

            # z_i^{t+1} = z_i^t + x_i^{t+1} - x_t
            z_i_new = _add(z_i, _sub(x_i_new, x_t))
            self.z_state[cid] = z_i_new
            z_new_list.append(z_i_new)

        # x^{t+1} = average of z_i^{t+1} across participating clients
        n = len(z_new_list)
        avg = z_new_list[0]
        for z in z_new_list[1:]:
            avg = _add(avg, z)
        avg = _scale(avg, 1.0 / n)

        self.global_params = avg

        if self.checkpoint_path is not None:
            import numpy as np
            np.savez(self.checkpoint_path, *self.global_params,
                     round=server_round)

        if self.round_logger is not None:
            self.round_logger.record_fit_round(server_round, results)

        return ndarrays_to_parameters(avg), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=self.min_available_clients,
        )
        config = {"server_round": server_round}
        return [(c, fl.common.EvaluateIns(parameters, config)) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        if self.round_logger is not None:
            return self.round_logger.record_evaluate_round(server_round, results)

        # fallback (no logger provided): plain weighted average, no f1
        losses = [r.loss * r.num_examples for _, r in results]
        accs = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        total_examples = sum(r.num_examples for _, r in results)
        agg_loss = sum(losses) / total_examples
        agg_acc = sum(accs) / total_examples
        return agg_loss, {"accuracy": agg_acc}

    def evaluate(self, server_round, parameters):
        return None
