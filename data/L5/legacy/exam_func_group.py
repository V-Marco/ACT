import sys
sys.path.append("../")

print("Passed output folder:", sys.argv[1])


import numpy as np
import h5py, pickle, os
from multiprocessing import Process

from Modules.logger import Logger

from Modules.plotting_utils import plot_simulation_results
from cell_inference.config import params
from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces

# Output folder should store folders 'saved_at_step_xxxx'
#output_folders = ["output/2023-08-16_16-24-29_seeds_123_87L5PCtemplate[0]_196nseg_108nbranch_15842NCs_15842nsyn"]

# Set the output_folder variable based on the command line argument
output_folders = [sys.argv[1] if len(sys.argv) > 1 else "output/default_path"]

import importlib
def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module





#constants = load_constants_from_folder(output_folder)

# Check shape of each saved file
analyze = True

# Plot voltage and lfp traces
plot_voltage_lfp = True

def main(output_folder):

    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
    
    # Save folder
    save_path = os.path.join(output_folder, "Analysis Voltage and LFP")
    if os.path.exists(save_path):
      logger = Logger(output_dir = save_path, active = True)
      logger.log(f'Directory already exists: {save_path}')
    else:
      os.mkdir(save_path)
      logger = Logger(output_dir = save_path, active = True)
      logger.log(f'Creating Directory: {save_path}')
    if analyze:
        for step in steps:
            dirname = os.path.join(output_folder, f"saved_at_step_{step}")
            for filename in os.listdir(dirname):
                if filename.endswith(".h5"):
                    with h5py.File(os.path.join(dirname, filename)) as file:
                        data = np.array(file["report"]["biophysical"]["data"])
                        logger.log(f"{os.path.join(f'saved_at_step_{step}', filename)}: {data.shape}")
    
    if plot_voltage_lfp:
        t = []
        Vm = []
        lfp = []
        for step in steps:
            dirname = os.path.join(output_folder, f"saved_at_step_{step}")
            with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
                Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
            with h5py.File(os.path.join(dirname, "lfp.h5")) as file:
                lfp.append(np.array(file["report"]["biophysical"]["data"])[:step_size, :])
            with h5py.File(os.path.join(dirname, "t.h5")) as file:
                t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])

        Vm = np.hstack(Vm)
        lfp = np.vstack(lfp)
        t = np.hstack(t) # (ms)

        with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
            seg_indexes = pickle.load(file)
        
        loc_param = [0., 0., 45., 0., 1., 0.]
        elec_pos = params.ELECTRODE_POSITION
        plot_simulation_results(t, Vm, seg_indexes['soma'], seg_indexes['axon'], seg_indexes['basal'], 
                                seg_indexes['tuft'], seg_indexes['nexus'], seg_indexes['trunk'], 
                                loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, vlim = [-0.023,0.023],
                                show = False, save_dir = save_path)


if __name__ == "__main__":

    pool = []
    for folder in output_folders:
        constants = load_constants_from_folder(folder)
        if constants.parallelize:
            pool.append(Process(target = main, args = [folder]))
        else:
            p = Process(target = main, args = [folder])
            p.start()
            p.join()
            p.terminate()
    
    if constants.parallelize:
        for p in pool: p.start()
        for p in pool: p.join()
        for p in pool: p.terminate()
