import json
import os
import shutil
import h5

import numpy as np
from bmtk.builder.networks import NetworkBuilder
from bmtk.simulator import bionet
from bmtk.utils.sim_setup import build_env_bionet
from neuron import h

from act.act_types import SimulationConfig

pc = h.ParallelContext()  # object to access MPI methods
MPI_RANK = int(pc.id())


def create_output_folder(config: SimulationConfig, overwrite=True) -> str:
    output_folder = config["output"]["folder"]
    run_output_folder_name = f"{config['run_mode']}"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_folder = os.path.join(config["output"]["folder"], run_output_folder_name)
    # delete old results
    if os.path.exists(output_folder) and overwrite:
        shutil.rmtree(output_folder, ignore_errors=True)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    return output_folder


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

def get_param_dist(config: SimulationConfig):
    # Deterimine the number of cells and their parameters to be set
    lows = [p["low"] for p in config["optimization_parameters"]["params"]]
    highs = [p["high"] for p in config["optimization_parameters"]["params"]]

    n_slices = (
        config["optimization_parameters"]
        .get("parametric_distribution", {})
        .get("n_slices", 0)
    )
    if n_slices <= 1:
        raise ValueError(
            'config["optimization_parameters"]["parametric_distribution"]["n_slices"] must be > 2 to generate a distribution.'
        )

    param_dist = np.array(
        [
            np.arange(low, high, (high - low) / (n_slices - 1))
            for low, high in zip(lows, highs)
        ]
    ).T

    param_dist = np.array(param_dist.tolist() + [highs])  # add on the highes values
    parameter_values_list = get_params(param_dist)
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

def build_parametric_network(config: SimulationConfig):

    config_file = "simulation_act_simulation_config.json"
    params = [p["channel"] for p in config["optimization_parameters"]["params"]]
    param_dist = get_param_dist(config)

    network_dir = "network"
    # remove everything from prior runs
    if os.path.exists(network_dir):
        for f in os.listdir(network_dir):
            os.remove(os.path.join(network_dir, f))

    cell_name = config["cell"]["name"]
    hoc_file = config["cell"]["hoc_file"]
    modfiles_folder = config["cell"]["modfiles_folder"]

    amps = config["optimization_parameters"]["parametric_distribution"]["amps"]
    amp_delay = config["simulation_parameters"]["h_i_delay"]
    amp_duration = config["simulation_parameters"]["h_i_dur"]

    dt = config["simulation_parameters"]["h_dt"]
    tstop = config["simulation_parameters"]["h_tstop"]
    v_init = config["simulation_parameters"]["h_v_init"]

    n_cells_per_amp = param_dist.shape[0] ** param_dist.shape[1]
    n_cells = len(amps) * n_cells_per_amp

    print(
        f"Number of cells to be generated: {n_cells}  ({len(amps)} amps * {param_dist.shape[1]} parameters ^ {param_dist.shape[0]} splits)"
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
        clamps[pop] = {
            "input_type": "current_clamp",
            "module": "IClamp",
            "node_set": pop,
            "amp": amp,
            "delay": amp_delay,
            "duration": amp_duration,
        }
    net.build()
    net.save_nodes(output_dir=network_dir)
    net.save_edges(output_dir=network_dir)

    build_env_bionet(
        base_dir="./",
        network_dir=network_dir,
        tstop=tstop,
        dt=dt,
        dL=9999999.9,
        report_vars=["v"],
        v_init=v_init,
        celsius=31.0,
        components_dir="components",
        config_file="act_simulation_config.json",
        compile_mechanisms=False,
        overwrite_config=True,
    )

    # copy the files to the correct network directories
    shutil.copy(hoc_file, "components/templates/")

    new_mods_folder = "./components/mechanisms/"
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
        json.dump(conf_dict, f)

    nodesets_file = 'node_sets.json'
    parameter_values_file = 'parameter_values.json'
    node_dict = None
    with open(nodesets_file) as json_file:
        node_dict = json.load(json_file)
        for i, amp in enumerate(amps):
            pop = f"amp{i}"
            node_dict[pop] = [i for i in range(n_cells_per_amp*i, n_cells_per_amp*i + n_cells_per_amp)]
    with open(nodesets_file, "w") as f:
        json.dump(node_dict, f, indent=2)

    parameter_values_list = get_param_dist(config)
    with open(parameter_values_file) as json_file:
        param_dict = {"params": params,
                      "parameter_values_list": parameter_values_list}
        json.dump(param_dict, f, indent=2)

def generate_parametric_traces(config: SimulationConfig):
    """
    This function utilizes BMTK and MPI to generate voltage
    traces for a large collection of cells and generates an h5
    file for injestion later.
    """
    config_file = "simulation_act_simulation_config.json"
    parameter_values_file = "parameter_values.json"
    with open(parameter_values_file) as f:
        param_dict = json.load(f)
        params = param_dict["params"]
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
        parameter_values = parameter_values_list[cell.gid % n_param_values]
        set_cell_parameters(cell.hobj, params, parameter_values)

    # Run the Simulation
    sim.run()
    bionet.nrn.quit_execution()
    # Save traces/parameters and cleanup

def load_parametric_traces(config: SimulationConfig):
    """
    Return a torch tensor of all traces in the specified h5 file
    """
    parameter_values_file = "parameter_values.json"
    traces_file = 'output/v_report.h5'

    with open(parameter_values_file, "r") as json_file:
        params_dict = json.load(json_file)
        params = params_dict["params"]
        parameter_values_list = params_dict["parameter_values_list"]

    traces = h5.File(traces_file)
    import torch

    import pdb;pdb.set_trace()