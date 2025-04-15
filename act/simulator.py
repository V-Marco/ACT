import os
import sys
import shutil
from act.act_types import SimulationParameters, ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection
from act.cell_model import ACTCellModel
from neuron import h
import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

# This file defines ACTSimulator (The primary class for interfacing with NEURON) with helper functions



@contextmanager
def suppress_neuron_warnings():
    '''
    Turns of nrnivmodl compile warnings.
    Parameters:
    -----------
    None
    
    Returns:
    -----------
    None
    '''
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
            

def unwrap_self_run_job(args) -> None:
    '''
    A function for implementing multiprocessing for NEURON
    https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-class
    Parameters:
    -----------
    args: list[list]
    
    Returns:
    -----------
    None
    '''
    return ACTSimulator._run_job(args[0], args[1][0], args[1][1])


class ACTSimulator:

    def __init__(self, output_folder_name) -> None:
        self.path = output_folder_name
        self.pool = []
        print("""
        ACTSimulator (2025)
        ----------
        When submitting multiple jobs, note that the cells must share modfiles.
        """)
        

    def run(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        '''
        Used for single job NEURON simulations
        Parameters:
        -----------
        self
        
        cell: ACTCellModel
        
        parameters: SimulationParameters
        
        Returns:
        -----------
        None
        '''
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

        cell._build_cell(parameters.sim_idx)

        # Set current injection
        if len(parameters.CI) > 0:
            if isinstance(parameters.CI[0], ConstantCurrentInjection):
                cell._add_constant_CI(
                    parameters.CI[0].amp, 
                    parameters.CI[0].dur, 
                    parameters.CI[0].delay, 
                    parameters.h_tstop, 
                    parameters.h_dt, 
                    parameters.CI[0].lto_hto)
            elif isinstance(parameters.CI[0], RampCurrentInjection):
                cell._add_ramp_CI(
                    parameters.CI[0].amp_start, 
                    parameters.CI[0].amp_incr, 
                    parameters.CI[0].num_steps, 
                    parameters.CI[0].step_time, 
                    parameters.CI[0].dur, 
                    parameters.CI[0].delay, 
                    parameters.h_tstop, 
                    parameters.h_dt, 
                    parameters.CI[0].lto_hto)
            elif isinstance(parameters.CI[0], GaussianCurrentInjection):
                cell._add_gaussian_CI(
                    parameters.CI[0].amp_mean, 
                    parameters.CI[0].amp_std, 
                    parameters.CI[0].dur, 
                    parameters.CI[0].delay, 
                    parameters.random_seed, 
                    parameters.CI[0].lto_hto)
            else:
                raise NotImplementedError

        print(f"Soma area: {cell.soma[0](0.5).area()}")
        print(f"Soma diam: {cell.soma[0].diam}")
        print(f"Soma L: {cell.soma[0].L}")

        h.finitialize(h.v_init)
        h.run()

        return cell
    

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        '''
        Used for setting variable simulation parameters in multi-job uses of NEURON simulations
        Parameters:
        -----------
        self
        
        cell: ACTCellModel
        
        parameters: SimulationParameters
        
        Returns:
        -----------
        None
        '''
        parameters._path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))
        

    def run_jobs(self, n_cpu: int = None) -> None:
        '''
        Multiprocessing implementation of multi-job NEURON simulations. Sets up multiprocessing
        Parameters:
        -----------
        self
        
        n_cpu: int, default = None
            Number of cores to be used for multiprocessing
                 
        Returns:
        -----------
        None
        '''
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
    

    def _run_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        '''
        Instructions for a single NEURON simulation that is thread safe and used in run_jobs().
        Parameters:
        -----------
        self
        
        cell: ACTCellModel
        
        parameters: SimulationParameters
                 
        Returns:
        -----------
        None
        '''
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

        cell._build_cell(parameters.sim_idx)

        # Set current injection
        if isinstance(parameters.CI[0], ConstantCurrentInjection):
            cell._add_constant_CI(
                parameters.CI[0].amp, 
                parameters.CI[0].dur, 
                parameters.CI[0].delay, 
                parameters.h_tstop, 
                parameters.h_dt, 
                parameters.CI[0].lto_hto)
        elif isinstance(parameters.CI[0], RampCurrentInjection):
            cell._add_ramp_CI(
                parameters.CI[0].amp_start, 
                parameters.CI[0].amp_incr, 
                parameters.CI[0].num_steps, 
                parameters.CI[0].step_time, 
                parameters.CI[0].dur, 
                parameters.CI[0].delay, 
                parameters.h_tstop, 
                parameters.h_dt, 
                parameters.CI[0].lto_hto)
        elif isinstance(parameters.CI[0], GaussianCurrentInjection):
            cell._add_gaussian_CI(
                parameters.CI[0].amp_mean, 
                parameters.CI[0].amp_std, 
                parameters.CI[0].dur, 
                parameters.CI[0].delay, 
                parameters.random_seed, 
                parameters.CI[0].lto_hto)
        else:
            raise NotImplementedError
        
        
        if not cell.set_g_to == None and not len(cell.set_g_to) == 0:
            cell._set_g_bar(cell.active_channels, cell.set_g_to[1])   

        h.finitialize(h.v_init)
        h.run()

        V, I, g, sim_index, lto_hto = cell.get_output()

        out = np.zeros((int(parameters.h_tstop / parameters.h_dt), 4))
        out[:, 0] = V[:int(parameters.h_tstop / parameters.h_dt)]
        out[:, 1] = I[:int(parameters.h_tstop / parameters.h_dt)]
        out[:len(g), 2] = g
        out[len(g):, 2] = np.nan
        out[0, 3] = sim_index
        out[1, 3] = lto_hto
        out[2:, 3] = np.nan

        np.save(os.path.join(parameters._path, f"out_{parameters.sim_idx}.npy"), out)
        
