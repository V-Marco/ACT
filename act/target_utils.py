import numpy as np
import torch

from act.optim import ACTOptimizer


def get_voltage_trace_from_params(simulation_constants: object) -> torch.Tensor:
    optim = ACTOptimizer(simulation_constants=simulation_constants)
    target_V = []
    for amp in simulation_constants.amps:
        target_V.append(
            optim.simulate(
                amp, simulation_constants.params, simulation_constants.target_params
            ).reshape(1, -1)
        )
    target_V = torch.cat(target_V, dim=0)
    return target_V
