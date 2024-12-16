from multiprocessing import Pool, cpu_count

from act.act_types import SimulationParameters
from act.cell_model import ACTCellModel, TargetCell, TrainCell

from neuron import h
import numpy as np

import os
import sys
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
    return ACTSimulator._run_job(args[0], args[1][0], args[1][1])

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

class ACTSimulator:

    def __init__(self, output_folder_name) -> None:
        self.path = output_folder_name
        self.pool = []
        print("""
        ACTSimulator (2024)
        ----------
        When submitting multiple jobs, note that the cells must share modfiles.
        """)

    def run(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:

        # Compile the modfiles and suppress output
        os.system(f"nrnivmodl {cell.path_to_mod_files} > /dev/null 2>&1")

        # Load the stdrun
        h.load_file('stdrun.hoc')

        # Mechanisms might already be loaded
        try:
            with suppress_neuron_warnings():
                h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
        except:
            pass

        # Set parameters
        h.celsius = parameters.h_celsius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init

        # Build the cell
        cell._build_cell()

        # Set CI
        if parameters.CI[0].type == "constant":
            cell._add_constant_CI(parameters.CI[0].amp, parameters.CI[0].dur, parameters.CI[0].delay, parameters.h_tstop, parameters.h_dt)

        h.finitialize(h.v_init)
        h.run()

        return cell

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        parameters._path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))

    def run_jobs(self, n_cpu: int = None) -> None:

        # Create the simulation parent folder if it doesn't exist
        os.makedirs(self.path, exist_ok = True)
        
        if n_cpu is None:
            n_cpu = cpu_count()

        # Compile the modfiles and suppress output
        os.system(f"nrnivmodl {self.pool[0][0].path_to_mod_files} > /dev/null 2>&1")
        
        pool = Pool(processes = n_cpu)
        pool.map(unwrap_self_run_job, zip([self] * len(self.pool), self.pool))
        pool.close()
        pool.join()

        # Clean
        self.pool = []
        shutil.rmtree("x86_64")

    def _run_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:

        # Create this simulation's folder
        os.makedirs(parameters._path, exist_ok = True)

        # Load the stdrun
        h.load_file('stdrun.hoc')

        # Mechanisms might already be loaded
        try:
            with suppress_neuron_warnings():
                h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
        except:
            pass

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
        if parameters.CI[0].type == "constant":
            cell._add_constant_CI(parameters.CI[0].amp, parameters.CI[0].dur, parameters.CI[0].delay, parameters.h_tstop, parameters.h_dt)
        elif parameters.CI["type"] == "ramp":
            cell._add_ramp_CI(parameters.CI["start_amp"], parameters.CI["amp_incr"],parameters.CI["num_steps"],parameters.CI["step_time"],parameters.CI["dur"], parameters.CI["delay"], parameters.h_tstop, parameters.h_dt)
            pass
        else:
            raise NotImplementedError
        
        #If this is a train cell, load gs to set
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
    
        np.save(os.path.join(parameters._path, f"out_{parameters.sim_idx}.npy"), out)
        
