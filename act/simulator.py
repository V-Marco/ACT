from neuron import h
from multiprocessing import Pool, cpu_count

from act.act_types import SimulationParameters
from act.cell_model import ACTCellModel, TargetCell, TrainCell


import numpy as np

import os

# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
def unwrap_self_run_job(args):
    return Simulator._run_job(args[0], args[1][0], args[1][1])

class Simulator:

    def __init__(self) -> None:
        self.path = "model"
        self.pool = []

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        parameters.path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))

    def run(self, path_to_modfiles: str):
        print(f"Total number of jobs: {len(self.pool)}")
        print(f"Total number of proccessors: {cpu_count()}")

        # Create the simulation parent folder if it doesn't exist
        if os.path.isdir(self.path) == False:
            os.mkdir(self.path)

        # Compile the modfiles and suppress output
        os.system(f"nrnivmodl {path_to_modfiles} > /dev/null 2>&1")
        
        pool = Pool(processes = len(self.pool))
        pool.map(unwrap_self_run_job, zip([self] * len(self.pool), self.pool))
        pool.close()
        pool.join()

        # Delete the compiled modfiles
        os.system("rm -r x86_64")

    def _run_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:

        # Create this simulation's folder
        if not os.path.exists(parameters.path):
            os.mkdir(parameters.path)

        h.nrn_load_dll(os.path.join(cell.mod_folder, "libnrnmech.so"))
        
        # Load standard run files
        h.load_file('stdrun.hoc')

        # Set parameters
        h.celsius = parameters.h_celsius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init

        # Build the cell
        cell._build_cell()

        # Set CI
        if parameters.CI["type"] == "constant":
            cell._add_constant_CI(parameters.CI["amp"], parameters.CI["dur"], parameters.CI["delay"])
        else:
            raise NotImplementedError
        
        # If this is a train cell, load gs to set
        if type(cell) == TrainCell:
            cell._set_g(cell.g_to_set_after_build[0], cell.g_to_set_after_build[1])

        # Simulate
        h.finitialize(h.v_init)
        h.run()
        V, I, g = cell.get_output()

        # Force 1 ms resolution and save
        out = np.zeros((parameters.h_tstop, 3))
        out[:, 0] = V[::int(1 / parameters.h_dt)][:parameters.h_tstop]
        out[:, 1] = I[:parameters.h_tstop]
        out[:len(g), 2] = g

        np.save(os.path.join(parameters.path, "out.npy"), out)
        
