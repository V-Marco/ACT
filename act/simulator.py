import os
import sys
import shutil
from act.act_types import SimulationParameters
from act.cell_model import ACTCellModel
from neuron import h
import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

# This file defines ACTSimulator (The primary class for interfacing with NEURON) with helper functions

'''
suppress_neuron_warnings
Turns of nrnivmodl compile warnings
'''

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
            
'''
unwrap_self_run_job
A function for implementing multiprocessing for NEURON
https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
'''

def unwrap_self_run_job(args):
    return ACTSimulator._run_job(args[0], args[1][0], args[1][1])


class ACTSimulator:

    def __init__(self, output_folder_name) -> None:
        self.path = output_folder_name
        self.pool = []
        print("""
        ACTSimulator (2024)
        ----------
        When submitting multiple jobs, note that the cells must share modfiles.
        """)
        
    '''
    run
    Used for single job NEURON simulations
    '''

    def run(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        os.system(f"nrnivmodl {cell.path_to_mod_files} > /dev/null 2>&1")

        h.load_file('stdrun.hoc')

        try:
            with suppress_neuron_warnings():
                h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
        except:
            pass

        h.celsius = parameters.h_celsius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init

        cell._build_cell()

        if parameters.CI[0].type == "constant":
            cell._add_constant_CI(parameters.CI[0].amp, parameters.CI[0].dur, parameters.CI[0].delay, parameters.h_tstop, parameters.h_dt)

        h.finitialize(h.v_init)
        h.run()

        return cell
    
    '''
    submit_job
    Used for setting variable simulation parameters in multi-job uses of NEURON simulations
    '''

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        parameters._path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))
        
    '''
    run_jobs
    Multiprocessing implementation of multi-job NEURON simulations. Sets up multiprocessing
    '''

    def run_jobs(self, n_cpu: int = None) -> None:
        os.makedirs(self.path, exist_ok = True)
        
        if n_cpu is None:
            n_cpu = cpu_count()

        os.system(f"nrnivmodl {self.pool[0][0].path_to_mod_files} > /dev/null 2>&1")
        
        pool = Pool(processes = n_cpu)
        pool.map(unwrap_self_run_job, zip([self] * len(self.pool), self.pool))
        pool.close()
        pool.join()

        self.pool = []
        shutil.rmtree("x86_64")
    
    '''
    _run_job
    Instructions for a single NEURON simulation that is thread safe and used in run_jobs().
    '''

    def _run_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        os.makedirs(parameters._path, exist_ok = True)

        h.load_file('stdrun.hoc')

        try:
            with suppress_neuron_warnings():
                h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
        except:
            pass

        h.celsius = parameters.h_celsius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init

        cell._build_cell()

        if parameters.CI[0].type == "constant":
            cell._add_constant_CI(parameters.CI[0].amp, parameters.CI[0].dur, parameters.CI[0].delay, parameters.h_tstop, parameters.h_dt)
        elif parameters.CI["type"] == "ramp":
            cell._add_ramp_CI(parameters.CI["start_amp"], parameters.CI["amp_incr"],parameters.CI["num_steps"],parameters.CI["step_time"],parameters.CI["dur"], parameters.CI["delay"], parameters.h_tstop, parameters.h_dt)
            pass
        else:
            raise NotImplementedError
        
        # if not parameters.set_g_to == None and not len(parameters.set_g_to) == 0:
        #     cell._set_g(parameters.set_g_to[parameters.sim_idx][0], parameters.set_g_to [parameters.sim_idx][1])   

        h.finitialize(h.v_init)
        h.run()
        out = cell.V.as_numpy()

        # V, I, g = cell.get_output()

        # out = np.zeros((int(parameters.h_tstop / parameters.h_dt), 3))
        # out[:, 0] = V[:int(parameters.h_tstop / parameters.h_dt)]
        # out[:, 1] = I[:int(parameters.h_tstop / parameters.h_dt)]
        # out[:len(g), 2] = g
        # out[len(g):, 2] = np.nan

        np.save(f"out_{parameters.sim_idx}.npy", out)
        
