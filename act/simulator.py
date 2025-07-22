import os, sys, shutil
from neuron import h
import numpy as np
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

from act.types import SimulationParameters, ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection
from act.cell_model import ACTCellModel


@contextmanager
def _suppress_neuron_warnings():
    """
    Turns off nrnivmodl compile warnings.
    """
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
            

# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-class
def _unwrap_self_run_job(args) -> None:
    return ACTSimulator._run_job(args[0], args[1][0], args[1][1])


class ACTSimulator:
    """Simulate ACT-compatible cell models.
    """

    def __init__(self, output_folder_name) -> None:
        """
        Initialize the simulator.

        Parameters
        ----------
        output_folder_name: str
            Name of the output folder.
        """
        self.path = output_folder_name
        self.pool = []
        print("""
        ACTSimulator (2025)
        ----------
        When submitting multiple jobs, note that the cells must share modfiles.
        """)
        

    def run(self, cell: ACTCellModel, parameters: SimulationParameters) -> ACTCellModel:
        """
        Run a single simulation and return the cell model. Meant for interactive exploration of the cell model.

        Parameters
        ----------
        cell: ACTCellModel
            Cell model to simulate.
        
        parameters: SimulationParameters
            Paramteters for the simulation.
        
        Returns
        -------
        cell: ACTCellModel
            Cell model after the simulation.
        """
        os.system(f"nrnivmodl {cell.path_to_mod_files} > /dev/null 2>&1")

        h.load_file('stdrun.hoc')

        try:
            with _suppress_neuron_warnings():
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
        for CI in parameters.CI:
            if isinstance(CI, ConstantCurrentInjection):
                cell._add_constant_CI(
                    CI.amp, CI.dur, CI.delay, 
                    parameters.h_tstop, 
                    parameters.h_dt)
            elif isinstance(CI, RampCurrentInjection):
                cell._add_ramp_CI(
                    CI.amp_start, CI.amp_incr, CI.num_steps, CI.dur, CI.final_step_add_time, CI.delay, 
                    parameters.h_tstop, 
                    parameters.h_dt)
            elif isinstance(CI, GaussianCurrentInjection):
                cell._add_gaussian_CI(
                    CI.amp_mean, CI.amp_std, CI.dur, CI.delay, 
                    parameters.random_seed)
            else:
                raise NotImplementedError

        print(f"Soma area: {cell.soma[0](0.5).area()}")
        print(f"Soma diam: {cell.soma[0].diam}")
        print(f"Soma L: {cell.soma[0].L}")

        h.finitialize(h.v_init)
        h.run()

        return cell
    

    def submit_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        """
        Schedule a job for a delayed parallel simulation.
        
        Parameters
        ----------
        cell: ACTCellModel
            Cell model to simulate.
        
        parameters: SimulationParameters
            Parameters for the simulations.
        
        Returns
        -------
        None
        """
        parameters._path = os.path.join(self.path, parameters.sim_name)
        self.pool.append((cell, parameters))
        

    def run_jobs(self, n_cpu: int = None) -> None:
        """
        Run the scheduled jobs.
        
        Parameters
        ----------
        n_cpu: int, default = None
            Number of cores to be used for multiprocessing. If None, all available cores are used.
                 
        Returns
        -------
        None
        """
        os.makedirs(self.path, exist_ok = True)
        
        if n_cpu is None:
            n_cpu = cpu_count()

        os.system(f"nrnivmodl {self.pool[0][0].path_to_mod_files} > /dev/null 2>&1")
        
        pool = Pool(processes = n_cpu)
        pool.map(_unwrap_self_run_job, zip([self] * len(self.pool), self.pool))
        pool.close()
        pool.join()

        self.pool = []
        shutil.rmtree("x86_64")
    

    def _run_job(self, cell: ACTCellModel, parameters: SimulationParameters) -> None:
        """
        Instructions for a single NEURON simulation that is thread-safe and used in run_jobs().
        
        Parameters
        ----------
        cell: ACTCellModel
        
        parameters: SimulationParameters
                 
        Returns
        -------
        None
        """
        os.makedirs(parameters._path, exist_ok = True)

        # Load the modfiles
        h.load_file('stdrun.hoc')

        try:
            with _suppress_neuron_warnings():
                h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
        except:
            pass
        
        # Set simulation parameters
        h.celsius = parameters.h_celsius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init

        # Build the cell
        cell._build_cell(parameters.sim_idx, parameters.verbose)
        
        # Set current injection
        for CI in parameters.CI:
            if isinstance(CI, ConstantCurrentInjection):
                cell._add_constant_CI(
                    CI.amp, CI.dur, CI.delay, 
                    parameters.h_tstop, 
                    parameters.h_dt)
            elif isinstance(CI, RampCurrentInjection):
                cell._add_ramp_CI(
                    CI.amp_start, CI.amp_incr, CI.num_steps, CI.dur, CI.final_step_add_time, CI.delay, 
                    parameters.h_tstop, 
                    parameters.h_dt)
            elif isinstance(CI, GaussianCurrentInjection):
                cell._add_gaussian_CI(
                    CI.amp_mean, CI.amp_std, CI.dur, CI.delay, 
                    parameters.random_seed)
            else:
                raise NotImplementedError
        
        # Update conductances
        if not cell._set_g_to == None and not len(cell._set_g_to) == 0:
            cell._set_g_bar()   

        # Init and run the simulation
        h.finitialize(h.v_init)
        h.run()

        # Save outputs
        V, I, g = cell.get_output()

        out = np.zeros((int(parameters.h_tstop / parameters.h_dt), 3))
        out[:, 0] = V[:int(parameters.h_tstop / parameters.h_dt)]
        out[:, 1] = I[:int(parameters.h_tstop / parameters.h_dt)]
        out[:len(g), 2] = g
        out[len(g):, 2] = np.nan

        np.save(os.path.join(parameters._path, f"out_{parameters.sim_idx}.npy"), out)
        
