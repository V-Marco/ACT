import os

import numpy as np
import torch

from act.act_types import SimulationConfig
from act.optim import ACTOptimizer
from act.analysis import save_plot


def get_voltage_trace_from_params(
    simulation_config: SimulationConfig,
) -> torch.Tensor:
    optim = ACTOptimizer(simulation_config=simulation_config, set_passive_properties=False)
    target_V = []
    output_folder = os.path.join(simulation_config["output"]["folder"],
                                 simulation_config["run_mode"],
                                 "target")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for i, amp in enumerate(simulation_config["optimization_parameters"]["amps"]):
        params = [
            p["channel"] for p in simulation_config["optimization_parameters"]["params"]
        ]
        target_params = simulation_config["optimization_parameters"]["target_params"]
        tv = optim.simulate(amp, params, target_params).reshape(1, -1)
        target_V.append(tv)
        # write to output folder / mode / target
        if i % 5 == 0: # 5 should be user defined
            save_plot(
                amp, output_folder, simulated_data = None, target_V=tv.detach().numpy(), output_file=f"target_{(amp * 1000):.0f}nA.png"
            )
    target_V = torch.cat(target_V, dim=0)
    return target_V
