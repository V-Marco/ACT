import numpy as np
import torch

from act.act_types import SimulationConfig
from act.optim import ACTOptimizer


def get_voltage_trace_from_params(
    simulation_config: SimulationConfig,
) -> torch.Tensor:
    optim = ACTOptimizer(simulation_config=simulation_config)
    target_V = []
    for amp in simulation_config["optimization_parameters"]["amps"]:
        params = [
            p["channel"] for p in simulation_config["optimization_parameters"]["params"]
        ]
        target_params = simulation_config["optimization_parameters"]["target_params"]
        target_V.append(optim.simulate(amp, params, target_params).reshape(1, -1))
    target_V = torch.cat(target_V, dim=0)
    return target_V
