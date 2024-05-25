import json
import os
import shutil
import h5py
import shutil
from io import StringIO

import numpy as np
from bmtk.builder.networks import NetworkBuilder
from bmtk.simulator import bionet
from bmtk.utils.sim_setup import build_env_bionet
from neuron import h
import pandas as pd
import torch
import multiprocessing as mp
from scipy import signal
import itertools

from act.act_types import SimulationConfig
from act.cell_model import CellModel
from act.DataProcessor import DataProcessor

import warnings

pc = h.ParallelContext()  # object to access MPI methods
MPI_RANK = int(pc.id())

def get_random_seed(config: SimulationConfig) -> str:
    return f"{config['optimization_parameters']['random_seed']}"

def create_output_folder(config: SimulationConfig, overwrite=True) -> str:
    if(config["output"]["auto_structure"] == True):
        print("AUTO STRUCTURED")
        output_folder = get_output_folder_name(config)
    else:
        output_folder = f"{config['output']['folder']}"

    if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    # delete old results
    if os.path.exists(output_folder) and overwrite:
        shutil.rmtree(output_folder, ignore_errors=True)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    return output_folder

def get_output_folder_name(config: SimulationConfig) -> str:
    cell_name = config["cell"]["name"]
    num_slices = config['optimization_parameters']['parametric_distribution']['n_slices']
    num_slices_name = ""
    if type(num_slices) is list:
        for x in num_slices:
            num_slices_name = num_slices_name + f"{x}-"
    else:
        num_slices_name = f"{num_slices}"
    run_mode = f"{config['run_mode']}"  #"segregated" "origin
    if(run_mode == "segregated"):
        run_mode_name = "seg"
    else:
        run_mode_name = "orig"

    return f"./output/{cell_name}_{run_mode_name}_{num_slices_name}slice/"

def get_sim_data_folder_name(config: SimulationConfig) -> str:
    segregation_index = get_segregation_index(config)
    if config["run_mode"] == "segregated":
        sim_dir = get_output_folder_name(config) + "sim_data/" + f"module_{segregation_index+1}/"
    else:
        sim_dir = get_output_folder_name(config) + "sim_data/" 
    return sim_dir

def get_param_values_file(config: SimulationConfig) -> str:
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    return get_output_folder_name(config) + "sim_data/"  + f"parameter_values_{random_seed}-seed.json"

def get_sim_output_folder_name(config: SimulationConfig) -> str:
    return get_sim_data_folder_name(config) + "output/"

def create_model_data_folder(config: SimulationConfig) -> str:
    segregation_index = get_segregation_index(config)
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    if config["run_mode"] == "segregated":
        
        output_folder = get_output_folder_name(config) + "model_data/" 

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = output_folder + f"{random_seed}-seed/"

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = output_folder + f"module_{segregation_index+1}/"
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    else:
        output_folder = get_output_folder_name(config) + "model_data/" 

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = output_folder + f"{random_seed}-seed/"

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    
    return output_folder

def get_model_data_folder_name(config: SimulationConfig) -> str:
    output_dir = get_output_folder_name(config)
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    segregation_index = get_segregation_index(config)
    if(config["run_mode"] == "segregated"):
        model_data_dir = output_dir + "model_data/"+ f"{random_seed}-seed/" + f"module_{segregation_index+1}/"
    else:
        model_data_dir = output_dir + "model_data/"+ f"{random_seed}-seed/"
    return model_data_dir

def get_last_model_data_folder_name(config: SimulationConfig) -> str:
    # This is used for the plotting scripts because the module # was incremented, so we need the "last run"
    output_dir = get_output_folder_name(config)
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    segregation_index = get_segregation_index(config)
    if(config["run_mode"] == "segregated"):
        model_data_dir = output_dir + "model_data/"+ f"{random_seed}-seed/" + f"module_{segregation_index}/"
    else:
        model_data_dir = output_dir + "model_data/"+ f"{random_seed}-seed/"
    return model_data_dir

def get_final_folder(config: SimulationConfig) -> str:
    output_dir = get_output_folder_name(config)
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    return output_dir + "final/" + f"{random_seed}-seed/"

def set_cell_parameters(cell, parameter_list: list, parameter_values: list) -> None:
    for sec in cell.all:
        for index, key in enumerate(parameter_list):
            setattr(sec, key, parameter_values[index])


def get_params(param_dist):
    def _param(params, n_params, n_splits):
        if len(params) == n_params:
            return params
        p_list = []
        for i in range(n_splits):
            current_params = params + [i]
            p = _param(current_params, n_params, n_splits)
            p_list = p_list + [p]
        return p_list

    n_splits, n_params = param_dist.shape
    p_inds = _param([], n_params, n_splits)
    p_inds = np.array(p_inds).reshape(
        n_splits**n_params, n_params
    )  # reshape because appending is weird

    params = []
    for p_ind in p_inds:
        param_set = list(param_dist.T[range(param_dist.shape[1]), p_ind])
        params.append(param_set)
    return params


def get_param_dist(
    config: SimulationConfig,
    preset_params={},
    learned_params={},
    learned_variability=0,
    learned_variability_params=[],
    n_slices=0,
    block_channels=[],
):
    # Deterimine the number of cells and their parameters to be set
    preset_channels = [k for k, v in preset_params.items()]
    all_channels = [p["channel"] for p in config["optimization_parameters"]["params"]]

    # only learned parameters to adjust are ones that are in learned_variability_params if set
    if len(learned_variability_params) > 0:
        learned_params = {
            k: v for k, v in learned_params.items() if k in learned_variability_params
        }

    if (
        learned_variability > 0
    ):  # we should remove them from learned since we're going to want to re-learn them
        preset_channels = [p for p in preset_channels if p not in learned_params]

    channels = [
        p["channel"]
        for p in config["optimization_parameters"]["params"]
        if p["channel"] not in preset_channels and p["channel"] not in block_channels
    ]
    lows = [
        p["low"]
        for p in config["optimization_parameters"]["params"]
        if p["channel"] not in preset_channels and p["channel"] not in block_channels
    ]
    highs = [
        p["high"]
        for p in config["optimization_parameters"]["params"]
        if p["channel"] not in preset_channels and p["channel"] not in block_channels
    ]

    if learned_variability > 0:
        print(f"learned variability set to {learned_variability}")
        # we need to adjust the high and low values to be around the learned value by a percentage
        new_lows = []
        new_highs = []
        for channel, low, high in zip(channels, lows, highs):
            if channel in learned_params:
                new_lows.append(learned_params[channel] * (1 - learned_variability))
                new_highs.append(learned_params[channel] * (1 + learned_variability))
            else:
                new_lows.append(low)
                new_highs.append(high)
        print(f"channels set this segregation {channels}")
        print(f"old lows: {lows}")
        print(f"old highs: {highs}")
        lows = new_lows
        highs = new_highs
        print(f"new lows: {lows}")
        print(f"new highs: {highs}")

    if not n_slices:
        n_slices = (
            config["optimization_parameters"]
            .get("parametric_distribution", {})
            .get("n_slices", 0)
        )
    if type(n_slices) is int:
        if n_slices <= 1:
            raise ValueError(
                'config["optimization_parameters"]["parametric_distribution"]["n_slices"] must be > 2 to generate a distribution.'
            )
        param_dist = np.array(
            [
                np.arange(
                    low, high - 1e-15, (high - low) / (n_slices - 1)
                )  # -1e-15 because we really don't want the last one since we're adding it back in, can lead to inhomogenous shapes if ommited
                for low, high in zip(lows, highs)
            ]
        ).T
        param_dist = np.array(param_dist.tolist() + [highs])  # add on the highes values
        parameter_values_list = get_params(param_dist)
    else:
        if len(n_slices) == 0:
            raise ValueError(
                'config["optimization_parameters"]["parametric_distribution"]["n_slices"] requires each element to be > 2 to generate a distribution.'
            )
        param_dist = []
        count = 0
        # Generate list of lists of the increments of gbar for each ion channel
        for low, high, n_slice in zip(lows, highs, n_slices):
            count=count+1
            slice_range = np.arange(
                low, high - 1e-15, (high - low) / (n_slice - 1)
            ).tolist() # -1e-15 because we really don't want the last one since we're adding it back in, can lead to inhomogenous shapes if ommited
            slice_range.append(high)
            param_dist.append(slice_range)

        # Generate all of the gbar combinations in a list of lists
        parameter_values_list = [list(x) for x in itertools.product(*param_dist)]

    if config["run_mode"] == "segregated":
        # need to add in the preset channels to each of the generated parametric sets
        parameter_values_list_updated = []
        for parameter_values in parameter_values_list:  # for each parameter set
            for channel, value in zip(
                channels, parameter_values
            ):  # update the preset_channels dict to include the current parameter set
                preset_params[channel] = value
            for channel in block_channels:
                preset_params[channel] = 0.0
            current_parameter_values_list = []
            for channel in all_channels:
                current_parameter_values_list.append(preset_params[channel])
            parameter_values_list_updated.append(current_parameter_values_list)
        return parameter_values_list_updated
    else:
        return parameter_values_list


def cleanup_simulation():
    files = [
        "act_simulation_config.json",
        "circuit_act_simulation_config.json",
        "node_sets.json",
        "run_bionet.py",
        "simulation_act_simulation_config.json",
    ]
    folders = ["components"]
    # TODO Remove


def get_segregation_index(config: SimulationConfig):
    parameter_values_file = get_param_values_file(config)

    if config["run_mode"] != "segregated":
        return -1
    if not os.path.exists(parameter_values_file):
        return 0
    with open(parameter_values_file, "r") as fp:
        parameter_values_dict = json.load(fp)
    segregation_index = parameter_values_dict["segregation_index"]

    return segregation_index


def load_preset_params(config: SimulationConfig):
    # Returns a dict of learned params from segregation
    # if segregation is not used then returns an empty dict
    parameter_values_file = get_param_values_file(config)

    if config["run_mode"] != "segregated":
        return {}
    if not os.path.exists(parameter_values_file):
        return {}
    with open(parameter_values_file, "r") as fp:
        parameter_values_dict = json.load(fp)
    segregation_index = parameter_values_dict["segregation_index"]

    preset_params = parameter_values_dict["learned_params"]
    # everything after the current index should be zero
    total_segregations = len(config["segregation"])
    for i in range(segregation_index + 1, total_segregations):
        for seg_p in config["segregation"][i]["params"]:
            # print(f"Setting {seg_p} = 0 for future segregation")
            preset_params[seg_p] = 0

    return preset_params


def load_learned_params(config: SimulationConfig):
    # Returns a dict of learned params from segregation
    # if segregation is not used then returns an empty dict
    parameter_values_file = get_param_values_file(config)

    if not os.path.exists(parameter_values_file):
        return {}
    with open(parameter_values_file, "r") as fp:
        parameter_values_dict = json.load(fp)
    learned_params = parameter_values_dict.get("learned_params", {})
    return learned_params


def get_learned_variability(config: SimulationConfig):
    parameter_values_file = get_param_values_file(config)
    lv = 0
    if os.path.exists(parameter_values_file):
        with open(parameter_values_file, "r") as fp:
            parameter_values_dict = json.load(fp)
        segregation_index = parameter_values_dict["segregation_index"]
        lv = config["segregation"][segregation_index].get("learned_variability", 0)
    return lv


def get_learned_variability_params(config: SimulationConfig):
    parameter_values_file = get_param_values_file(config)
    lvp = []
    if os.path.exists(parameter_values_file):
        with open(parameter_values_file, "r") as fp:
            parameter_values_dict = json.load(fp)
        segregation_index = parameter_values_dict["segregation_index"]
        lvp = config["segregation"][segregation_index].get(
            "learned_variability_params", []
        )
    return lvp


def update_segregation(config: SimulationConfig, learned_params):
    # This function accepts the learned parameters for the network
    # And updates the parameter_values.json if that parameter was
    # in the current segregation index
    # learned_params = {'channel'(str):value(float),}
    output_dir = get_sim_data_folder_name(config)
    parameter_values_file = get_param_values_file(config)
    if os.path.exists(parameter_values_file):
        print(f"Updating {parameter_values_file} for learned parameters")
        with open(parameter_values_file, "r") as fp:
            parameter_values_dict = json.load(fp)
        segregation_index = parameter_values_dict["segregation_index"]

        current_segregation_params = []
        if config["segregation"][segregation_index].get("learned_variability", 0) > 0:
            for i in range(
                segregation_index
            ):  # we should add previous params to our list
                current_segregation_params = current_segregation_params + [
                    p
                    for p in config["segregation"][i]["params"]
                    if p not in current_segregation_params
                ]

            learned_variability_params = []
            if config["segregation"][segregation_index].get(
                "learned_variability_params", []
            ):
                # if there are only specific parameters that we want to learn, update those only, remove from the current_seg params
                learned_variability_params = config["segregation"][
                    segregation_index
                ].get("learned_variability_params")
                current_segregation_params = [
                    p
                    for p in current_segregation_params
                    if p in learned_variability_params
                ]

        current_segregation_params = (
            current_segregation_params
            + config["segregation"][segregation_index]["params"]
        )

        # don't want to update anything that was blocked
        if config["segregation"][segregation_index].get("use_hto_amps", False):
            hto_block_channels = config["optimization_parameters"]["hto_block_channels"]
            new_current_params = []
            for p in current_segregation_params:
                if p not in hto_block_channels:
                    new_current_params.append(p)
                else:
                    print(f"Not updating {p} because it was blocked")
            current_segregation_params = new_current_params

        for learned_param, value in learned_params.items():
            if learned_param in current_segregation_params:
                parameter_values_dict["learned_params"][learned_param] = value
                print(f"Segregation parameter {learned_param} updated to {value}")

        parameter_values_dict["segregation_index"] += 1
        print(
            f"Segregation stage {segregation_index+1}/{len(config['segregation'])} complete."
        )
        with open(parameter_values_file, "w") as fp:
            json.dump(parameter_values_dict, fp, indent=4)
        if segregation_index+1 == len(config["segregation"]):
            final_folder = get_final_folder(config)
            final_parameter_file = f"final/{final_folder}/parameter_values_seg_complete.json"
            print(f"Saving final learned parameters to: {final_parameter_file}")
            shutil.move(parameter_values_file, final_parameter_file)
    else:
        print(
            f"{parameter_values_file} file not found - unable to update learned params"
        )


def save_learned_params(learned_params, config: SimulationConfig):
    parameter_values_file = get_param_values_file(config)
    if os.path.exists(parameter_values_file):
        print(f"Updating {parameter_values_file} for learned parameters")
        with open(parameter_values_file, "r") as fp:
            parameter_values_dict = json.load(fp)
        parameter_values_dict["learned_params"] = {}
        for learned_param, value in learned_params.items():
            parameter_values_dict["learned_params"][learned_param] = value
        with open(parameter_values_file, "w") as fp:
            json.dump(parameter_values_dict, fp, indent=4)


def build_parametric_network(config: SimulationConfig):
    output_dir = get_sim_data_folder_name(config)
    config_file = output_dir + "simulation_act_simulation_config.json"
    parameter_values_file = get_param_values_file(config)

    params = [p["channel"] for p in config["optimization_parameters"]["params"]]

    if os.path.exists("components"):
        print(f"./components dir exists, removing")
        shutil.rmtree("components")

    learned_params = {}  # keep track for future, not writing anything in this script
    preset_params = {}  # will set some to 0 and others to what was determined before
    segregation_index = 0  # looping through each of the segregations in the config
    n_slices = 0
    learned_variability = 0
    learned_variability_params = []
    block_channels = []
    # check if we're running in segregated mode
    # we keep track of the segregated state
    if config["run_mode"] == "segregated":
        print("Segregation mode selected.")
        total_segregations = len(config["segregation"])
        # we're segregated and have run the simulation before
        if os.path.exists(parameter_values_file):
            print(f"Loading {parameter_values_file} for learned parameters")
            with open(parameter_values_file, "r") as fp:
                parameter_values_dict = json.load(fp)
            segregation_index = parameter_values_dict["segregation_index"]

            if segregation_index > 0 and segregation_index < total_segregations:
                learned_params = parameter_values_dict["learned_params"]
                preset_params = load_preset_params(config)
            else:
                print(f"Segregation stage 1/{total_segregations} started.")
                segregation_index = 0
                segregation_params = config["segregation"][segregation_index]["params"]
                preset_params = {p: 0 for p in params if p not in segregation_params}
                print(f"Segregation parameters selected: {segregation_params}")
                print(f"Setting all else to zero: {preset_params}")
        else:  # this is the first run, and we've not loaded before
            print(f"Segregation stage 1/{total_segregations} started.")
            # Everything but the first set of parameters should be set to zero
            segregation_params = config["segregation"][segregation_index]["params"]
            preset_params = {p: 0 for p in params if p not in segregation_params}
            print(f"Segregation parameters selected: {segregation_params}")
            print(f"Setting all else to zero: {preset_params}")

        learned_variability = config["segregation"][segregation_index].get(
            "learned_variability", 0
        )
        learned_variability_params = config["segregation"][segregation_index].get(
            "learned_variability_params", []
        )
        n_slices = config["segregation"][segregation_index].get("n_slices", 0)
        if config["segregation"][segregation_index].get("use_hto_amps", False):
            print("hto block channels loaded")
            block_channels = config["optimization_parameters"].get(
                "hto_block_channels", []
            )
            # for hbc in hto_block_channels:
            #    preset_params[hbc] = 0.0

    param_dist = get_param_dist(
        config,
        preset_params=preset_params,
        learned_params=learned_params,
        learned_variability=learned_variability,
        learned_variability_params=learned_variability_params,
        n_slices=n_slices,
        block_channels=block_channels,
    )    

    network_dir = output_dir + "network"
    # remove everything from prior runs
    if os.path.exists(network_dir):
        for f in os.listdir(network_dir):
            os.remove(os.path.join(network_dir, f))

    cell_name = config["cell"]["name"]
    hoc_file = config["cell"]["hoc_file"]
    modfiles_folder = config["cell"]["modfiles_folder"]

    if config["run_mode"] == "segregated" and config["segregation"][
        segregation_index
    ].get("use_lto_amps", False):
        print(f"Using LTO Amps for current segregation (use_lto_amps set)")
        amps = config["optimization_parameters"]["lto_amps"]
    elif config["run_mode"] == "segregated" and config["segregation"][
        segregation_index
    ].get("use_hto_amps", False):
        print(f"Using HTO Amps for current segregation (use_hto_amps set)")
        amps = config["optimization_parameters"]["hto_amps"]
    else:
        amps = config["optimization_parameters"]["amps"]

    # see if there needs to be a ramp
    ramp_time = 0
    ramp_splits = 1
    if config["run_mode"] == "segregated":
        ramp_time = config["segregation"][segregation_index].get("ramp_time", 0.0)
        ramp_splits = config["segregation"][segregation_index].get("ramp_splits", 1)

    amp_delay = config["simulation_parameters"]["h_i_delay"]
    amp_duration = config["simulation_parameters"]["h_i_dur"]

    dt = config["simulation_parameters"]["h_dt"]
    tstop = config["simulation_parameters"]["h_tstop"]
    v_init = config["simulation_parameters"]["h_v_init"]
    celsius = config["simulation_parameters"].get("h_celsius")

    if (
        config["run_mode"] == "segregated"
    ):  # sometimes segregated modules have different params
        amp_delay = config["segregation"][segregation_index].get("h_i_delay", amp_delay)
        amp_duration = config["segregation"][segregation_index].get(
            "h_i_dur", amp_duration
        )
        tstop = config["segregation"][segregation_index].get("h_tstop", tstop)

    if not celsius:
        print("Setting celsius to default value of 31.0")
        celsius = 31.0
    else:
        print(f"Celsius set to {celsius}")

    n_cells_per_amp = len(param_dist)
    n_cells = len(amps) * n_cells_per_amp

    print(
        f"Number of cells to be generated: {n_cells}  ({len(amps)} amps * {n_cells_per_amp} cells per amp (parameters ^ splits))"
    )

    # Build a BMTK config to inject and record

    clamps = {}
    net = NetworkBuilder("biocell")
    # Since these need specified in the config, we split by amp delivered
    for i, amp in enumerate(amps):
        pop = f"amp{i}"
        net.add_nodes(
            N=n_cells_per_amp,
            pop_name=pop,
            model_type="biophysical",
            model_template="hoc:" + cell_name,
            morphology=None,
        )
        if ramp_time:
            for index in range(ramp_splits):
                ramp_amp = amp / ramp_splits
                ramp_time_split = ramp_time / ramp_splits
                ramp_duration = ramp_time + amp_duration - index * ramp_time_split
                ramp_delay = amp_delay + index * ramp_time_split
                clamps[f"clamp_{i}_{index}"] = {
                    "input_type": "current_clamp",
                    "module": "IClamp",
                    "node_set": pop,
                    "amp": ramp_amp,
                    "delay": ramp_delay,
                    "duration": ramp_duration,
                }

        else:  # regular amps
            clamps[pop] = {
                "input_type": "current_clamp",
                "module": "IClamp",
                "node_set": pop,
                "amp": amp,
                "delay": amp_delay,
                "duration": amp_duration,
            }

    params_list = []
    amps_list = []
    for amp in amps:
        amps_list = amps_list + [amp for _ in range(n_cells_per_amp)]
        params_list = params_list + param_dist

    net.build()
    net.save_nodes(output_dir=network_dir)
    net.save_edges(output_dir=network_dir)

    build_env_bionet(
        base_dir=output_dir,
        network_dir=network_dir,
        tstop=tstop
        + ramp_time,  # ramp time is normally zero but when this is all done then we need to cut the first "ramp_time" off each trace to keep it consistent
        dt=dt,
        dL=9999999.9,
        report_vars=["v"],
        v_init=v_init,
        celsius=celsius,
        components_dir= "components",
        config_file="act_simulation_config.json",
        compile_mechanisms=False,
        overwrite_config=True,
    )

    # copy the files to the correct network directories
    shutil.copy(hoc_file, output_dir + "components/templates/")

    new_mods_folder = output_dir + "components/mechanisms/"
    src_files = os.listdir(modfiles_folder)
    for file_name in src_files:
        full_file_name = os.path.join(modfiles_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, new_mods_folder)

    # compile mod files
    os.system(f"cd {new_mods_folder} && nrnivmodl && cd -")

    # Update the generated config to include clamps
    conf_dict = None

    with open(config_file, "r") as json_file:
        conf_dict = json.load(json_file)
        conf_dict["inputs"] = clamps
        conf_dict["manifest"]["$NETWORK_DIR"] = "$BASE_DIR/" + network_dir
        conf_dict["network"] = "$BASE_DIR/circuit_act_simulation_config.json"

    with open(config_file, "w") as f:
        json.dump(conf_dict, f, indent=2)

    nodesets_file = output_dir + "node_sets.json"
    node_dict = None
    with open(nodesets_file) as json_file:
        node_dict = json.load(json_file)
        for i, amp in enumerate(amps):
            pop = f"amp{i}"
            node_dict[pop] = [
                i
                for i in range(
                    n_cells_per_amp * i, n_cells_per_amp * i + n_cells_per_amp
                )
            ]
    with open(nodesets_file, "w") as f:
        json.dump(node_dict, f, indent=2)

    with open(parameter_values_file, "w") as json_file:
        param_dict = {
            "parameters": params,
            "parameter_values_list": params_list,
            "amps": list(amps_list),
            "segregation_index": segregation_index,
            "learned_params": learned_params,
        }
        json.dump(param_dict, json_file, indent=2)


def generate_parametric_traces(config: SimulationConfig):
    """
    This function utilizes BMTK and MPI to generate voltage
    traces for a large collection of cells and generates an h5
    file for injestion later.
    """
    output_dir = get_sim_data_folder_name(config)
    passive_properties = config.get("cell", {}).get("passive_properties", None)
    config_file = output_dir + "simulation_act_simulation_config.json"
    parameter_values_file = get_param_values_file(config)
    with open(parameter_values_file) as f:
        param_dict = json.load(f)
        params = param_dict["parameters"]
        parameter_values_list = param_dict["parameter_values_list"]

    pc.barrier()

    # Build network, modify cell parameters
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)

    cells = graph.get_local_cells()

    n_param_values = len(parameter_values_list)
    for c_ind in cells:
        cell = cells[c_ind]
        # since we create n_amps number of cells * the parametric distribution we just want to loop around each time the amps change
        parameter_values = parameter_values_list[cell.gid]
        if passive_properties:
            CellModel.set_passive_props(
                cell.hobj.all, passive_properties, cell.hobj.soma[0]
            )
        print(f"Setting cell {cell.gid} parameters {params} = {parameter_values}")
        set_cell_parameters(cell.hobj, params, parameter_values)
    pc.barrier()

    # Run the Simulation
    sim.run()
    #bionet.nrn.quit_execution()

    # Save traces/parameters and cleanup


def apply_decimate_factor(config: SimulationConfig, traces):
    decimate_factor = config["optimization_parameters"].get("decimate_factor")
    if decimate_factor:
        if isinstance(traces, torch.Tensor):
            traces = traces.cpu().detach().numpy()
        print(f"decimate_factor set - reducing dataset by {decimate_factor}x")
        from scipy import signal

        traces = signal.decimate(
            traces, decimate_factor
        ).copy()  # copy per neg index err
    return torch.tensor(traces)


def load_parametric_traces(config: SimulationConfig, drop_ramp=False):
    """
    Return a torch tensor of all traces in the specified h5 file
    """
    output_dir = get_sim_data_folder_name(config)
    parameter_values_file = get_param_values_file(config)
    traces_file = output_dir + "output/v_report.h5"

    if not os.path.exists(parameter_values_file) or not os.path.exists(traces_file):
        return None, None, None

    with open(parameter_values_file, "r") as json_file:
        params_dict = json.load(json_file)
        params = params_dict["parameters"]
        parameter_values_list = params_dict["parameter_values_list"]
        amps = params_dict["amps"]

    print(f"loading large file ({traces_file})")
    traces_h5 = h5py.File(traces_file)
    traces = np.array(traces_h5["report"]["biocell"]["data"]).T
    # reorder to match parameters set
    order = list(traces_h5["report"]["biocell"]["mapping"]["node_ids"])
    traces = traces[order]

    import torch

    if (
        config["run_mode"] == "segregated"
    ):  # we should drop the first bit if we ramped up
        dt = config["simulation_parameters"]["h_dt"]
        segregation_index = get_segregation_index(
            config
        )  # could probably just get it from above, oh well
        ramp_time = config["segregation"][segregation_index].get("ramp_time", 0.0)
        ramp_splits = config["segregation"][segregation_index].get("ramp_splits", 1)
        if ramp_time > 0 and drop_ramp:
            skip_ramp_time = int(ramp_time / dt)
            traces = torch.tensor(traces)
            traces = traces[:, skip_ramp_time:]

    traces = apply_decimate_factor(config, traces)

    return traces, torch.tensor(parameter_values_list), torch.tensor(amps)


def extract_spiking_traces(
    traces_t, params_t, amps_t, threshold=-40, min_spikes=1, keep_zero_amps=True
):
    num_spikes, interspike_times, first_n_spikes_scaled, avg_spike_min, avg_spike_max = DataProcessor.extract_spike_features(traces_t)
    spiking_gids = (
        num_spikes.gt(min_spikes - 1).nonzero().flatten().cpu().detach().tolist()
    )
    if keep_zero_amps:
        zero_amps_gids = amps_t.eq(0.0).nonzero().flatten().cpu().detach().tolist()
        spiking_gids = list(set(set(zero_amps_gids) | set(spiking_gids)))

    spiking_traces = traces_t[spiking_gids]
    spiking_params = params_t[spiking_gids]
    spiking_amps = amps_t[spiking_gids]
    print(f"{len(spiking_traces)}/{len(traces_t)} spiking traces extracted.")
    return spiking_traces, spiking_params, spiking_amps, spiking_gids

def get_arima_order(config: SimulationConfig, segregation_index):
    arima_order = (10, 0, 10)
    if config.get("summary_features", {}).get("arima_order"):
        arima_order = tuple(config["summary_features"]["arima_order"])
    if config["run_mode"] == "segregated" and config["segregation"][segregation_index].get("arima_order",None):
        print(f"custom arima order for segregation set")
        arima_order = tuple(config["segregation"][segregation_index]["arima_order"])
    return arima_order

def get_fi_curve(traces, amps, ignore_negative=True, inj_dur=1000):
    """
    Returns the spike counts per amp.
    inj_dur is the duration of the current injection
    """

    spikes, interspike_times = DataProcessor.extract_summary_features(traces)

    if ignore_negative:
        non_neg_idx = (amps >= 0).nonzero().flatten()
        amps = amps[amps >= 0]
        spikes = spikes[non_neg_idx]

    spikes = (1000.0 / inj_dur) * spikes  # Convert to Hz

    return spikes


def get_fi_curve_error(
    simulated_traces,
    target_traces,
    amps,
    ignore_negative=True,
    dt=1,
    print_info=False,
    inj_dur=1000,
):
    """
    Returns the average spike count error over the entire trace.
    """
    simulated_spikes = get_fi_curve(
        simulated_traces,
        amps,
        ignore_negative=ignore_negative,
        inj_dur=inj_dur,
    )
    target_spikes = get_fi_curve(
        target_traces, amps, ignore_negative=ignore_negative, inj_dur=inj_dur
    )

    if print_info:
        print(f"target spikes: {target_spikes}")
        print(f"simulated spikes: {simulated_spikes}")
        print(f"diff: {simulated_spikes - target_spikes}")

    error = round(
        float((torch.abs(simulated_spikes - target_spikes)).sum() / len(amps)), 4
    )

    return error


def get_amplitude_frequency(traces, inj_dur, inj_start, fs=1000):
    amplitudes = []
    frequencies = []
    for idx in range(traces.shape[0]):
        x = traces[idx].cpu().numpy()[inj_start : inj_start + inj_dur]
        secs = len(x) / fs
        peaks = signal.find_peaks(x, prominence=0.1)[0].tolist()
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            amplitude = x[peaks].mean()
        frequency = int(len(peaks) / (secs))
        amplitudes.append(amplitude)
        frequencies.append(frequency)

    amplitudes = torch.tensor(amplitudes)
    frequencies = torch.tensor(frequencies)

    amplitudes[torch.isnan(amplitudes)] = 0
    frequencies[torch.isnan(frequencies)] = 0

    return amplitudes, frequencies

def get_mean_potential(traces, inj_dur, inj_start):
    mean_potential = traces[:,inj_start:inj_start+inj_dur].mean(dim=1)
    return mean_potential

def load_final_traces(trace_file):
    traces_h5 = h5py.File(trace_file)
    amps = torch.tensor(np.array(traces_h5["amps"]))
    simulated_traces = torch.tensor(np.array(traces_h5["simulated"]["voltage_trace"]))
    target_traces = torch.tensor(np.array(traces_h5["target"]["voltage_trace"]))
    return simulated_traces, target_traces, amps
