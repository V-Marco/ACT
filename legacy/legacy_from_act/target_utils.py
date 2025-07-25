import json
import os

import numpy as np
import torch
import sys
sys.path.append("../")
from act import simulator

from act.act_types import SimulationConfig
from legacy.analysis import save_plot
from act.optim import ACTOptimizer
from act.cell_model import CellModel
from legacy import utils

DEFAULT_TARGET_V_FILE = "target_v.json"
DEFAULT_TARGET_V_LTO_FILE = "target_v_lto.json"
DEFAULT_TARGET_V_HTO_FILE = "target_v_hto.json"


def get_voltage_trace_from_params(
    simulation_config: SimulationConfig,
    ignore_segregation=False,
) -> torch.Tensor:
    if not ignore_segregation:
        segregation_index = utils.get_segregation_index(simulation_config)
        segregated_and_lto = simulation_config[
            "run_mode"
        ] == "segregated" and simulation_config["segregation"][segregation_index].get(
            "use_lto_amps", False
        )
        segregated_and_hto = simulation_config[
            "run_mode"
        ] == "segregated" and simulation_config["segregation"][segregation_index].get(
            "use_hto_amps", False
        )
    else:
        segregated_and_lto = False
        segregated_and_hto = False

    # If we specify a target cell then we should simulate that target
    if (
        simulation_config["optimization_parameters"].get("target_cell")
        and not segregated_and_lto
        and not segregated_and_hto
    ):
        target_cell = CellModel(
            hoc_file=simulation_config["optimization_parameters"]["target_cell"][
                "hoc_file"
            ],
            cell_name=simulation_config["optimization_parameters"]["target_cell"][
                "name"
            ],
        )
        if simulation_config["optimization_parameters"].get("target_cell_params"):
            params = simulation_config["optimization_parameters"]["target_cell_params"]
        else:
            params = simulation_config["optimization_parameters"]["params"]
    # otherwise, just use the default config["cell"], loaded by ACTOptimizer
    else:
        target_cell = None
        params = simulation_config["optimization_parameters"]["params"]

    if segregated_and_lto or segregated_and_hto:
        target_params = simulation_config["optimization_parameters"]["target_params"]
        if segregated_and_lto:
            print("Checking to see if channels need blocked for lto")
            if simulation_config["optimization_parameters"].get("lto_block_channels"):
                block_channels = simulation_config["optimization_parameters"].get(
                    "lto_block_channels"
                )
                target_params_new = []
                params_list = [p["channel"] for p in params]
                for p, name in zip(target_params, params_list):
                    if name not in block_channels:
                        target_params_new.append(p)
                    else:
                        target_params_new.append(0)  # block the channel
                        print(f"blocking channel {name} | {name} = 0.0")
                target_params = target_params_new
        if segregated_and_hto:
            print("Checking to see if channels need blocked for hto")
            if simulation_config["optimization_parameters"].get("hto_block_channels"):
                block_channels = simulation_config["optimization_parameters"].get(
                    "hto_block_channels"
                )
                target_params_new = []
                params_list = [p["channel"] for p in params]
                for p, name in zip(target_params, params_list):
                    if name not in block_channels:
                        target_params_new.append(p)
                    else:
                        target_params_new.append(0)  # block the channel
                        print(f"blocking channel {name} | {name} = 0.0")
                target_params = target_params_new

    elif simulation_config["optimization_parameters"].get("target_cell_target_params"):
        target_params = simulation_config["optimization_parameters"].get(
            "target_cell_target_params"
        )
    else:
        target_params = simulation_config["optimization_parameters"]["target_params"]
    # create the optimizer
    optim = ACTOptimizer(
        simulation_config=simulation_config,
        set_passive_properties=False if (target_cell or segregated_and_lto or segregated_and_hto) else True,  # we only want to set passive properties if we're using the original cell, not target
        cell_override=target_cell,
        ignore_segregation=ignore_segregation,
    )
    target_V = []

    # create output folders
    output_folder = os.path.join(
        utils.get_output_folder_name(simulation_config), "target"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # get extra sim info
    dt = simulation_config["simulation_parameters"]["h_dt"]
    simulated_label = simulation_config["output"].get("simulated_label", "Simulated")
    target_label = simulation_config["output"].get("target_label", "Target")

    if (
        not ignore_segregation
        and simulation_config["run_mode"] == "segregated"
        and simulation_config["segregation"][segregation_index].get(
            "use_lto_amps", False
        )
    ):
        print(f"Using LTO Amps for current segregation (use_lto_amps set)")
        amps = simulation_config["optimization_parameters"]["lto_amps"]
    elif (
        not ignore_segregation
        and simulation_config["run_mode"] == "segregated"
        and simulation_config["segregation"][segregation_index].get(
            "use_hto_amps", False
        )
    ):
        print(f"Using HTO Amps for current segregation (use_hto_amps set)")
        amps = simulation_config["optimization_parameters"]["hto_amps"]
    else:
        amps = simulation_config["optimization_parameters"]["amps"]

    # generate data per amp
    for i, amp in enumerate(amps):
        print(f"Generating trace for {float(amp)*1000} pA")
        parameters = [p["channel"] for p in params]
        tv = optim.simulate(amp, parameters, target_params).reshape(1, -1)
        target_V.append(tv)
        # write to output folder / mode / target
        save_plot(
            amp,
            output_folder,
            simulated_data=None,
            target_V=tv.cpu().detach().numpy(),
            output_file=f"target_{(amp * 1000):.0f}pA.png",
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
        output_file="passive_-100pA.png",
        dt=dt,
        simulated_label=simulated_label,
    )

    return target_V


def save_target_traces(
    simulation_config: SimulationConfig,
    ignore_segregation=False,
    save_lto=False,  # useful for plotting final plots
    save_hto=False,
) -> torch.Tensor:
    target_V = get_voltage_trace_from_params(
        simulation_config, ignore_segregation=ignore_segregation
    )
    output_folder = utils.get_output_folder_name(simulation_config) + "target/"

    target_v_file = simulation_config["optimization_parameters"].get(
        "target_V_file", DEFAULT_TARGET_V_FILE
    )

    target_v_file = output_folder + target_v_file
    print(f"Saving target voltage data to: {target_v_file}")
    target_v_dict = {"traces": target_V.cpu().detach().tolist()}

    with open(target_v_file, "w") as fp:
        json.dump(target_v_dict, fp)

    if save_lto:
        target_v_lto_file = output_folder + DEFAULT_TARGET_V_LTO_FILE
        print(f"saving additional lto file {target_v_lto_file}...")
        with open(target_v_lto_file, "w") as fp:
            json.dump(target_v_dict, fp)
    if save_hto:
        target_v_hto_file = output_folder + DEFAULT_TARGET_V_HTO_FILE
        print(f"saving additional hto file {target_v_hto_file}...")
        with open(target_v_hto_file, "w") as fp:
            json.dump(target_v_dict, fp)

    return target_V


def load_target_traces(
    simulation_config: SimulationConfig, target_v_file=None,
    ignore_segregation=False,
    ramp_time = -1,
    cut_ramp=False
) -> torch.Tensor:
    output_folder = utils.get_output_folder_name(simulation_config) + "target/"
    if not target_v_file:
        target_v_file = simulation_config["optimization_parameters"].get(
            "target_V_file", DEFAULT_TARGET_V_FILE
        )

    target_v_file = output_folder + target_v_file

    with open(target_v_file, "r") as fp:
        target_v_dict = json.load(fp)
    print(f"Loading {target_v_file} for target traces")
    dt = simulation_config["simulation_parameters"]["h_dt"]
    traces = torch.tensor(target_v_dict["traces"])
    if simulation_config["run_mode"] == "segregated" and not ignore_segregation:
        segregation_index = utils.get_segregation_index(simulation_config)
        if ramp_time == -1: # this was not set
            ramp_time = simulation_config["segregation"][segregation_index].get(
                "ramp_time", 0
            )
    else:
        ramp_time = 0
    if ramp_time and cut_ramp:
        print(f"cutting ramp time {ramp_time} from beginning of trace")
        return traces[:, int(ramp_time / dt) :] 
    else:
        return traces
