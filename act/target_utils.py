import json
import os

import numpy as np
import torch

from act.act_types import SimulationConfig
from act.analysis import save_plot
from act.optim import ACTOptimizer
from act.cell_model import CellModel


def get_voltage_trace_from_params(
    simulation_config: SimulationConfig,
) -> torch.Tensor:
    target_cell = CellModel(
            hoc_file=simulation_config["cell"]["hoc_file"],
            cell_name=simulation_config["cell"]["name"],
        )
    optim = ACTOptimizer(
        simulation_config=simulation_config, set_passive_properties=False, cell_override=target_cell
    )
    target_V = []
    output_folder = os.path.join(
        simulation_config["output"]["folder"], simulation_config["run_mode"], "target"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if simulation_config["optimization_parameters"].get("target_cell_params"):
        target_params = simulation_config["optimization_parameters"].get("target_cell_params")
    else:
        target_params = simulation_config["optimization_parameters"]["target_params"]

    dt = simulation_config["simulation_parameters"]["h_dt"]
    simulated_label = simulation_config["output"].get("simulated_label", "Simulated")
    target_label = simulation_config["output"].get("target_label", "Target")
    for i, amp in enumerate(simulation_config["optimization_parameters"]["amps"]):
        params = [
            p["channel"] for p in simulation_config["optimization_parameters"]["params"]
        ]
        tv = optim.simulate(amp, params, target_params).reshape(1, -1)
        target_V.append(tv)
        # write to output folder / mode / target
        save_plot(
            amp,
            output_folder,
            simulated_data=None,
            target_V=tv.cpu().detach().numpy(),
            output_file=f"target_{(amp * 1000):.0f}nA.png",
            dt=dt,
            simulated_label=simulated_label,
            target_label=target_label,
        )
    target_V = torch.cat(target_V, dim=0)

    # save passive properties
    passive_properties, passive_v = optim.calculate_passive_properties(
        params, target_params
    )
    with open(
        os.path.join(output_folder, "target_passive_properties.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(passive_properties, fp, indent=2)

    save_plot(
        -0.1,
        output_folder,
        simulated_data=passive_v,
        output_file="passive_-100nA.png",
        dt=dt,
        simulated_label=simulated_label,
    )

    return target_V
