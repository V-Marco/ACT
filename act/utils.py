from act.act_types import SimulationConfig
from neuron import h
import numpy as np
import os
import shutil
from bmtk.builder.networks import NetworkBuilder
from bmtk.utils.sim_setup import build_env_bionet
from bmtk.simulator import bionet
import json

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

def cleanup_simulation():
    files = ["act_simulation_config.json", 
             "circuit_act_simulation_config.json",
             "node_sets.json",
             "run_bionet.py",
             "simulation_act_simulation_config.json",
    ]
    folders = ["components", "output"]
    # TODO Remove

def generate_parametric_traces(config: SimulationConfig):
    """
    This function utilizes BMTK and MPI to generate voltage
    traces for a large collection of cells and generates an h5
    file for injestion later.
    """

    config_file = "simulation_act_simulation_config.json"

    # Preparation on the first node
    if MPI_RANK == 0:
        # Create necessary folders
        output_dir = create_output_folder(config, overwrite=False)
        network_dir = "network"
        # remove everything from prior runs
        if os.path.exists(network_dir):
            for f in os.listdir(network_dir):
                os.remove(os.path.join(network_dir, f))

        cell_name = config["cell"]["name"]
        hoc_file = config["cell"]["hoc_file"]
        modfiles_folder = config["cell"]["modfiles_folder"]

        # Deterimine the number of cells and their parameters to be set
        params = [
            p["channel"] for p in config["optimization_parameters"]["params"]
        ]
        lows = [p["low"] for p in config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in config["optimization_parameters"]["params"]]
        n_slices = (
            config["optimization_parameters"]
            .get("parametric_distribution", {})
            .get("n_slices", 0)
        )
        if n_slices <= 1:
            raise ValueError('config["optimization_parameters"]["parametric_distribution"]["n_slices"] must be > 2 to generate a distribution.')
        
        steps = int(
            config["simulation_parameters"]["h_tstop"]
            / config["simulation_parameters"]["h_dt"]
        )
        amps = config["optimization_parameters"]["amps"]
        amp_delay = config["simulation_parameters"]["h_i_delay"]
        amp_duration = config["simulation_parameters"]["h_i_dur"]

        dt = config["simulation_parameters"]["h_dt"]
        tstop = config["simulation_parameters"]["h_tstop"]
        v_init = config["simulation_parameters"]["h_v_init"]

        param_dist = np.array(
            [
                np.arange(low, high , (high - low) / (n_slices-1))
                for low, high in zip(lows, highs)
            ]
        ).T

        param_dist = np.array(param_dist.tolist() + [highs]) # add on the highes values
        n_cells_per_amp =  param_dist.shape[1]**param_dist.shape[0]
        n_cells = len(amps) * n_cells_per_amp
        
        print(f'Number of cells to be generated: {n_cells}  ({len(amps)} amps * {param_dist.shape[1]} parameters ^ {param_dist.shape[0]} splits)')

        # Build a BMTK config to inject and record

        clamps = {}

        net = NetworkBuilder('biocell')
        # Since these need specified in the config, we split by amp delivered
        for i, amp in enumerate(amps):
            pop = f'amp{i}'
            net.add_nodes( 
                N=n_cells_per_amp,
                pop_name=pop,
                model_type='biophysical',
                model_template='hoc:' + cell_name,
                morphology=None
            )
            clamps[pop] = {
                "input_type": "current_clamp",
                "module": "IClamp",
                "node_set": {"population": pop},
                "amp": amp,
                "delay": amp_delay,
                "duration": amp_duration
            }
        net.build()
        net.save_nodes(output_dir=network_dir)
        net.save_edges(output_dir=network_dir)

        build_env_bionet(base_dir='./',
            network_dir=network_dir,
            tstop=tstop,
            dt = dt,
            dL = 9999999.9,
            report_vars = ['v'],
                v_init = v_init,
                celsius = 31.0,
            components_dir="components",
            config_file='act_simulation_config.json',
            compile_mechanisms=False,
            overwrite_config=True)
        
        # copy the files to the correct network directories
        shutil.copy(hoc_file, 'components/templates/')

        new_mods_folder = './components/mechanisms/'
        src_files = os.listdir(modfiles_folder)
        for file_name in src_files:
            full_file_name = os.path.join(modfiles_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, new_mods_folder)

        # compile mod files
        os.system(f"cd {new_mods_folder} && nrnivmodl && cd -")

        # Update the generated config to include clamps
        conf_dict = None

        with open(config_file, 'r') as json_file:
            conf_dict = json.load(json_file)
            conf_dict["inputs"] = clamps
            conf_dict["manifest"]["$NETWORK_DIR"] = "$BASE_DIR/" + network_dir
            conf_dict["network"] = "$BASE_DIR/circuit_act_simulation_config.json"

        with open(config_file, 'w') as f:
            json.dump(conf_dict, f)


    pc.barrier()

    # Build network, modify cell parameters
    conf = bionet.Config.from_dict(conf_dict, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    pop = graph._node_populations['biocell']

    cells = graph.get_local_cells()
    cell_counter = 0
    for amp_i, amp in enumerate(amps):
        parameter_values = np.zeros(param_dist.shape[1])
        # TODO DETERMINE VALUES from param dist
        cell = cells[cell_counter].hobj
        if not cell_counter%10000:
            print(f"Setting cell number {cell_counter} parameters.")
        set_cell_parameters(cell, params, parameter_values)
        cell_counter+=1

    # Run the Simulation
    sim.run()
    bionet.nrn.quit_execution()
    # Save traces/parameters and cleanup

    pc.barrier()

    if MPI_RANK == 0:
        # save to h5
        # TODO

        # cleanup
        cleanup_simulation()


def load_parametric_traces():
    """
    Return a torch tensor of all traces in the specified h5 file
    """