from neuron import h
from multiprocessing import Pool, cpu_count

from act.act_types import SimulationParameters, SimParams
from act.cell_model import ACTCellModel, TargetCell, TrainCell


import numpy as np

import os
import sys
import time
import shutil

from contextlib import contextmanager

@contextmanager
def suppress_neuron_warnings():
    with open(os.devnull, 'w') as dev_null:
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = dev_null
        sys.stderr = dev_null
        try:
            yield
        finally:
            sys.stdout = temp_stdout
            sys.stderr = temp_stderr

# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
def unwrap_self_run_job(args):
    return Simulator._run_job(args[0], args[1][0], args[1][1])

def print_mechanism_conductances(sec):
    print(f"\nMechanisms in section '{sec.name()}':")
    for seg in sec:
        print(f"  Segment {seg.x}:")
        for mech in seg:
            mech_name = mech.name()
            print(f"    Mechanism '{mech_name}':")
            for var in dir(mech):
                if not var.startswith('_') and not callable(getattr(mech, var)):
                    value = getattr(mech, var)
                    print(f"      {var} = {value}")

class Simulator:

    def __init__(self, output_folder_name) -> None:
        self.path = output_folder_name
        self.pool = []

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        parameters.path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))

    def run(self, path_to_modfiles: str):
        print(f"Total number of jobs: {len(self.pool)}")
        print(f"Total number of proccessors: {cpu_count()}")

        # Create the simulation parent folder if it doesn't exist
        os.makedirs(self.path, exist_ok=True)
        try:

            # Compile the modfiles and suppress output
            os.system(f"nrnivmodl {path_to_modfiles} > /dev/null 2>&1")

            time.sleep(2)

            # Attempt to load the compiled mechanisms
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    with suppress_neuron_warnings():
                        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
                    break
                except RuntimeError as e:
                    if "hocobj_call" in str(e):
                        print("MECHANISMS already loaded.")
                        break
                    elif "is not a MECHANISM" in str(e) and attempt < max_attempts - 1:
                        print(f"Loading compiled mechanisms failed {attempt + 1} time(s). Trying again (Max tries: {max_attempts})")
                        time.sleep(2) 
                    else:
                        print(str(e))
                        raise 

            pool = Pool(processes = int(cpu_count() * 0.6))
            pool.map(unwrap_self_run_job, zip([self] * len(self.pool), self.pool))
            pool.close()
            pool.join()

        except Exception as e:
                    print(f"An error occurred during the simulation: {e}")
                    raise
        finally:
            try:
                shutil.rmtree("x86_64")
            except OSError as e:
                print(f"Error removing x86_64 directory: {e}")
        

    def _run_job(self, cell: ACTCellModel, parameters: SimParams) -> None:
        # Create this simulation's folder
        os.makedirs(parameters.path, exist_ok=True)

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
        
        # Set passive properties
        if not cell.passive_properties == None:
            cell.set_passive_properties(cell.passive_properties)

        # Set CI
        if parameters.CI["type"] == "constant":
            cell._add_constant_CI(parameters.CI["amp"], parameters.CI["dur"], parameters.CI["delay"], parameters.h_tstop, parameters.h_dt)
        elif parameters.CI["type"] == "ramp":
            #cell._add_ramp_CI(parameters.CI["start_amp"], )
            pass
        else:
            raise NotImplementedError
        
        # If this is a train cell, load gs to set
        if not parameters.set_g_to == None and not len(parameters.set_g_to) == 0:
            #print(f"Setting G: {parameters.set_g_to[parameters.sim_idx][0]} to {parameters.set_g_to [parameters.sim_idx][1]}")
            cell._set_g(parameters.set_g_to[parameters.sim_idx][0], parameters.set_g_to [parameters.sim_idx][1])   

                
        #print_mechanism_conductances(cell.soma[0])
        # Simulate
        h.finitialize(h.v_init)
        h.run()
        V, I, g = cell.get_output()

        # Force 1 ms resolution and save
        out = np.zeros((int(parameters.h_tstop / parameters.h_dt), 3))
        out[:, 0] = V[:int(parameters.h_tstop / parameters.h_dt)]
        out[:, 1] = I[:int(parameters.h_tstop / parameters.h_dt)]
        out[:len(g), 2] = g
        out[len(g):, 2] = np.nan

        np.save(os.path.join(parameters.path, f"out_{parameters.sim_idx}.npy"), out)
        
