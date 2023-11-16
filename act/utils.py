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
from statsmodels.tsa.arima.model import ARIMA
import torch
import timeout_decorator
import multiprocessing as mp

from act.act_types import SimulationConfig
from act.cell_model import CellModel

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


def get_param_dist(config: SimulationConfig, preset_params={}):
    # Deterimine the number of cells and their parameters to be set
    preset_channels = [k for k,v in preset_params.items()]
    all_channels = [p["channel"] for p in config["optimization_parameters"]["params"]]
    channels = [p["channel"] for p in config["optimization_parameters"]["params"] if p["channel"] not in preset_channels]
    lows = [p["low"] for p in config["optimization_parameters"]["params"] if p["channel"] not in preset_channels]
    highs = [p["high"] for p in config["optimization_parameters"]["params"] if p["channel"] not in preset_channels]

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

    if config["run_mode"] == "segregated":
        # need to add in the preset channels to each of the generated parametric sets
        parameter_values_list_updated = []
        for parameter_values in parameter_values_list: # for each parameter set
            for channel,value in zip(channels,parameter_values): # update the preset_channels dict to include the current parameter set
                preset_params[channel] = value
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
    parameter_values_file = "parameter_values.json"

    if config["run_mode"] != "segregated":
        return -1
    if not os.path.exists(parameter_values_file):
        return -1
    with open(parameter_values_file, "r") as fp:
        parameter_values_dict = json.load(fp)
    segregation_index = parameter_values_dict["segregation_index"]

    return segregation_index

def load_preset_params(config: SimulationConfig):
    # Returns a dict of learned params from segregation
    # if segregation is not used then returns an empty dict
    parameter_values_file = "parameter_values.json"

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
    for i in range(segregation_index+1,total_segregations):
        for seg_p in config["segregation"][i]["params"]:
            print(f"Setting {seg_p} = 0 for future segregation")
            preset_params[seg_p] = 0
    
    return preset_params

def update_segregation(config: SimulationConfig, learned_params):
    # This function accepts the learned parameters for the network
    # And updates the parameter_values.json if that parameter was
    # in the current segregation index 
    # learned_params = {'channel'(str):value(float),}
    parameter_values_file = "parameter_values.json"
    if os.path.exists(parameter_values_file):
        print(f"Updating {parameter_values_file} for learned parameters")
        with open(parameter_values_file, "r") as fp:
            parameter_values_dict = json.load(fp)
        segregation_index = parameter_values_dict["segregation_index"]
        current_segregation_params = config["segregation"][segregation_index]["params"]
        for learned_param, value in learned_params.items():
            if learned_param in current_segregation_params:
                parameter_values_dict["learned_params"][learned_param] = value
                print(f"Segregation parameter {learned_param} updated to {value}")

        parameter_values_dict["segregation_index"] += 1
        print(f"Segregation stage {segregation_index+1}/{len(config['segregation'])} complete.")
        with open(parameter_values_file, "w") as fp:
            json.dump(parameter_values_dict, fp, indent=4)
    else:
        print(f"{parameter_values_file} file not found - unable to update learned params")
    

def build_parametric_network(config: SimulationConfig):
    config_file = "simulation_act_simulation_config.json"
    parameter_values_file = "parameter_values.json"

    params = [p["channel"] for p in config["optimization_parameters"]["params"]]

    if os.path.exists('components'):
        print(f"./components dir exists, removing")
        shutil.rmtree('components')

    learned_params = {} # keep track for future, not writing anything in this script
    preset_params = {} # will set some to 0 and others to what was determined before
    segregation_index = 0 # looping through each of the segregations in the config
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
                preset_params = {p:0 for p in params if p not in segregation_params}
                print(f"Segregation parameters selected: {segregation_params}")
                print(f"Setting all else to zero: {preset_params}")
        else: # this is the first run, and we've not loaded before
            print(f"Segregation stage 1/{total_segregations} started.")
            # Everything but the first set of parameters should be set to zero
            segregation_params = config["segregation"][segregation_index]["params"]
            preset_params = {p:0 for p in params if p not in segregation_params}
            print(f"Segregation parameters selected: {segregation_params}")
            print(f"Setting all else to zero: {preset_params}")
        

    param_dist = get_param_dist(config, preset_params=preset_params)

    network_dir = "network"
    # remove everything from prior runs
    if os.path.exists(network_dir):
        for f in os.listdir(network_dir):
            os.remove(os.path.join(network_dir, f))

    cell_name = config["cell"]["name"]
    hoc_file = config["cell"]["hoc_file"]
    modfiles_folder = config["cell"]["modfiles_folder"]

    amps = config["optimization_parameters"]["amps"]
    amp_delay = config["simulation_parameters"]["h_i_delay"]
    amp_duration = config["simulation_parameters"]["h_i_dur"]

    dt = config["simulation_parameters"]["h_dt"]
    tstop = config["simulation_parameters"]["h_tstop"]
    v_init = config["simulation_parameters"]["h_v_init"]
    celsius = config["simulation_parameters"].get("h_celsius")
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
        base_dir="./",
        network_dir=network_dir,
        tstop=tstop,
        dt=dt,
        dL=9999999.9,
        report_vars=["v"],
        v_init=v_init,
        celsius=celsius,
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

    nodesets_file = "node_sets.json"
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
            "learned_params": learned_params
        }
        json.dump(param_dict, json_file, indent=2)


def generate_parametric_traces(config: SimulationConfig):
    """
    This function utilizes BMTK and MPI to generate voltage
    traces for a large collection of cells and generates an h5
    file for injestion later.
    """
    passive_properties = config.get("cell", {}).get("passive_properties", None)
    config_file = "simulation_act_simulation_config.json"
    parameter_values_file = "parameter_values.json"
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
    bionet.nrn.quit_execution()
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


def load_parametric_traces(config: SimulationConfig):
    """
    Return a torch tensor of all traces in the specified h5 file
    """
    parameter_values_file = "parameter_values.json"
    traces_file = "output/v_report.h5"

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

    traces = apply_decimate_factor(config, traces)

    import torch

    return traces, torch.tensor(parameter_values_list), torch.tensor(amps)


def spike_stats(V: torch.Tensor, threshold=0, n_spikes=20):
    threshold = 0
    threshold_crossings = torch.diff(V > threshold, dim=1)

    first_n_spikes = torch.ones((V.shape[0], n_spikes)) * V.shape[1]
    avg_spike_min = torch.zeros((V.shape[0], 1))
    avg_spike_max = torch.zeros((V.shape[0], 1))
    for i in range(threshold_crossings.shape[0]):
        threshold_crossing_times = torch.arange(threshold_crossings.shape[1])[
            threshold_crossings[i, :]
        ]
        spike_times = []
        spike_mins = []
        spike_maxes = []
        for j in range(0, threshold_crossing_times.shape[0], 2):
            spike_times.append(threshold_crossing_times[j])
            ind = threshold_crossing_times[j : j + 2].cpu().tolist()
            end_ind = ind[1] if len(ind) == 2 else V.shape[1]
            spike_maxes.append(
                V[i][max(0, ind[0] - 1) : min(end_ind + 5, V.shape[1])].max()
            )
            spike_mins.append(
                V[i][max(0, ind[0] - 1) : min(end_ind + 5, V.shape[1])].min()
            )
        first_n_spikes[i][: min(n_spikes, len(spike_times))] = torch.tensor(
            spike_times
        ).flatten()[: min(n_spikes, len(spike_times))]
        avg_spike_max[i] = torch.mean(torch.tensor(spike_maxes).flatten())
        avg_spike_min[i] = torch.mean(torch.tensor(spike_mins).flatten())
    return first_n_spikes / V.shape[1], avg_spike_min, avg_spike_max


def extract_summary_features(V: torch.Tensor, threshold=-40) -> tuple:
    threshold_crossings = torch.diff(V > threshold, dim=1)
    num_spikes = torch.round(torch.sum(threshold_crossings, dim=1) * 0.5)
    interspike_times = torch.zeros((V.shape[0], 1))
    for i in range(threshold_crossings.shape[0]):
        interspike_times[i, :] = torch.mean(
            torch.diff(
                torch.arange(threshold_crossings.shape[1])[threshold_crossings[i, :]]
            ).float()
        )
    interspike_times[torch.isnan(interspike_times)] = 0

    return num_spikes, interspike_times


def extract_spiking_traces(traces_t, params_t, amps_t, threshold=-40, min_spikes=1):
    num_spikes, interspike_times = extract_summary_features(
        traces_t
    )
    spiking_gids = (
        num_spikes.gt(min_spikes - 1).nonzero().flatten().cpu().detach().tolist()
    )
    spiking_traces = traces_t[spiking_gids]
    spiking_params = params_t[spiking_gids]
    spiking_amps = amps_t[spiking_gids]
    print(f"{len(spiking_traces)}/{len(traces_t)} spiking traces extracted.")
    return spiking_traces, spiking_params, spiking_amps, spiking_gids


def get_arima_coefs(trace: np.array, order=(10, 0, 10)):
    model = ARIMA(endog=trace, order=order).fit()
    stats_df = pd.read_csv(
        StringIO(model.summary().tables[1].as_csv()),
        index_col=0,
        skiprows=1,
        names=["coef", "std_err", "z", "significance", "0.025", "0.975"],
    )
    stats_df.loc[stats_df["significance"].astype(float) > 0.05, "coef"] = 0
    coefs = stats_df.coef.tolist()
    return coefs


def arima_processor(trace_dict):
    @timeout_decorator.timeout(180, use_signals=True, timeout_exception=Exception)
    def arima_run(trace, order):
        return get_arima_coefs(trace, order)

    cell_id = trace_dict["cell_id"]
    trace = trace_dict["trace"]
    total = trace_dict["total"]
    arima_order = trace_dict["arima_order"]
    print(f"processing cell {cell_id+1}/{total}")

    try:
        coefs = arima_run(trace, arima_order)
    except Exception as e:
        print(f"problem processing cell {cell_id}: {e} | setting all values to 0.0")
        num_arima_vals = 2 + arima_order[0] + arima_order[2]
        coefs = [0.0 for _ in range(num_arima_vals)]

    trace_dict["coefs"] = coefs
    return trace_dict


def arima_coefs_proc_map(
    traces, num_procs=64, output_file="output/arima_stats.json", arima_order=(10, 0, 10)
):
    trace_list = []
    traces = traces.cpu().detach().tolist()
    num_traces = len(traces)
    for i, trace in enumerate(traces):
        trace_list.append(
            {
                "cell_id": i,
                "trace": trace,
                "total": num_traces,
                "arima_order": arima_order,
            }
        )
    with mp.Pool(num_procs) as pool:
        pool_output = pool.map_async(arima_processor, trace_list).get()
    # ensure ordering
    pool_dict = {}
    for out in pool_output:
        pool_dict[out["cell_id"]] = out["coefs"]
    coefs_list = []
    num_arima_vals = 2 + arima_order[0] + arima_order[2]
    for i in range(num_traces):
        if pool_dict.get(i):
            coefs_list.append(pool_dict[i])
        else:  # we didn't complete that task, was not found
            coefs_list.append([0 for _ in range(num_arima_vals)])

    output_dict = {}
    output_dict["arima_coefs"] = coefs_list

    with open(output_file, "w") as fp:
        json.dump(output_dict, fp, indent=4)

    return coefs_list


def load_arima_coefs(input_file="output/arima_stats.json"):
    with open(input_file) as json_file:
        arima_dict = json.load(json_file)
    return torch.tensor(arima_dict["arima_coefs"])


def extract_summary_features(V: torch.Tensor, spike_threshold=0) -> tuple:
    threshold_crossings = torch.diff(V > spike_threshold, dim=1)
    num_spikes = torch.round(torch.sum(threshold_crossings, dim=1) * 0.5)
    interspike_times = torch.zeros((V.shape[0], 1))
    for i in range(threshold_crossings.shape[0]):
        interspike_times[i, :] = torch.mean(
            torch.diff(
                torch.arange(threshold_crossings.shape[1])[threshold_crossings[i, :]]
            ).float()
        )
    interspike_times[torch.isnan(interspike_times)] = 0
    return num_spikes, interspike_times


def get_fi_curve(traces, amps, ignore_negative=True, inj_dur=1000):
    """
    Returns the spike counts per amp.
    inj_dur is the duration of the current injection
    """
    spikes, interspike_times = extract_summary_features(traces)

    if ignore_negative:
        non_neg_idx = (amps > 0).nonzero().flatten()
        amps = amps[amps > 0]
        spikes = spikes[non_neg_idx]

    spikes = (1000.0/inj_dur) * spikes # Convert to Hz

    return spikes


def get_fi_curve_error(
    simulated_traces, target_traces, amps, ignore_negative=True, dt=1, print_info=False, inj_dur=1000,
):
    """
    Returns the average spike count error over the entire trace.
    """
    simulated_spikes = get_fi_curve(
        simulated_traces, amps, ignore_negative=ignore_negative, inj_dur=inj_dur,
    )
    target_spikes = get_fi_curve(target_traces, amps, ignore_negative=ignore_negative, inj_dur=inj_dur)

    if print_info:
        print(f"target spikes: {target_spikes}")
        print(f"simulated spikes: {simulated_spikes}")
        print(f"diff: {simulated_spikes - target_spikes}")

    error = round(
        float((torch.abs(simulated_spikes - target_spikes)).sum() / len(amps)), 4
    )

    return error


def load_final_traces(trace_file):
    traces_h5 = h5py.File(trace_file)
    amps = torch.tensor(np.array(traces_h5["amps"]))
    simulated_traces = torch.tensor(np.array(traces_h5["simulated"]["voltage_trace"]))
    target_traces = torch.tensor(np.array(traces_h5["target"]["voltage_trace"]))
    return simulated_traces, target_traces, amps
