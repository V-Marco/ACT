import numpy as np
import torch

from act.act_types import SimulationConstants
from act.optim import ACTOptimizer


def get_voltage_trace_from_params(
    simulation_constants: SimulationConstants,
) -> torch.Tensor:
    optim = ACTOptimizer(simulation_constants=simulation_constants)
    target_V = []
    for amp in simulation_constants["optimization_parameters"]["amps"]:
        params = [
            p["channel"]
            for p in simulation_constants["optimization_parameters"]["params"]
        ]
        target_params = simulation_constants["segregation"]["target_params"]
        target_V.append(optim.simulate(amp, params, target_params).reshape(1, -1))
    target_V = torch.cat(target_V, dim=0)
    return target_V
