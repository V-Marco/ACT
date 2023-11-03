import json
import os

import numpy as np
import torch

from act.act_types import SimulationConfig
from act.analysis import save_plot
from act.optim import ACTOptimizer
from act.cell_model import CellModel

DEFAULT_TARGET_V_FILE = "./target_v.json"

def get_voltage_trace_from_params(
    simulation_config: SimulationConfig,
) -> torch.Tensor:
    # If we specify a target cell then we should simulate that target 
    if simulation_config["optimization_parameters"].get("target_cell"):
        target_cell = CellModel(
            hoc_file=simulation_config["optimization_parameters"]["target_cell"][
                "hoc_file"
            ],
            cell_name=simulation_config["optimization_parameters"]["target_cell"][
                "name"
            ],
        )
    # otherwise, just use the default config["cell"], loaded by ACTOptimizer
    else:
        target_cell = None

    # get our parameters and targets
    if simulation_config["optimization_parameters"].get("target_cell_params"):
        params = simulation_config["optimization_parameters"]["target_cell_params"]
    else:
        params = simulation_config["optimization_parameters"]["params"]

    if simulation_config["optimization_parameters"].get("target_cell_target_params"):
        target_params = simulation_config["optimization_parameters"].get(
            "target_cell_target_params"
        )
    else:
        target_params = simulation_config["optimization_parameters"]["target_params"]

    # create the optimizer
    optim = ACTOptimizer(
        simulation_config=simulation_config,
        set_passive_properties=False,
        cell_override=target_cell,
    )
    target_V = []

    # create output folders
    output_folder = os.path.join(
        simulation_config["output"]["folder"], simulation_config["run_mode"], "target"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # get extra sim info
    dt = simulation_config["simulation_parameters"]["h_dt"]
    simulated_label = simulation_config["output"].get("simulated_label", "Simulated")
    target_label = simulation_config["output"].get("target_label", "Target")

    # generate data per amp
    for i, amp in enumerate(simulation_config["optimization_parameters"]["amps"]):
        print(f"Generating trace for {float(amp)*1000} nA")
        parameters = [
            p["channel"] for p in params
        ]
        tv = optim.simulate(amp, parameters, target_params).reshape(1, -1)
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
        parameters, target_params
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

def save_target_traces(
        simulation_config: SimulationConfig,
) -> torch.Tensor:
    target_V = get_voltage_trace_from_params(simulation_config)

    target_v_file = simulation_config["optimization_parameters"].get("target_V_file", DEFAULT_TARGET_V_FILE)
    target_v_dict = {"traces":target_V.cpu().detach().tolist()}

    with open(target_v_file, "w") as fp:
        json.dump(target_v_dict, fp) 

    return target_V


def load_target_traces(simulation_config: SimulationConfig,
) -> torch.Tensor:
    target_v_file = simulation_config["optimization_parameters"].get("target_V_file", DEFAULT_TARGET_V_FILE)
    
    with open(target_v_file, "r") as fp:
        target_v_dict = json.load(fp) 
    print(f"Loading {target_v_file} for target traces")
    return torch.tensor(target_v_dict["traces"])
