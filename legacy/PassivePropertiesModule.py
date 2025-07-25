import os
import time
import sys
import shutil
from contextlib import contextmanager

from neuron import h
import numpy as np
from act.cell_model import TrainCell
from act.act_types import SimulationParameters, PassiveProperties
from act.DataProcessor import DataProcessor

'''
suppress_neuron_warnings
Hides nrnivmodl compilation warnings
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
PassivePropertiesModule
The first step to the Automatic Cell Tuner process where analytical solutions to passive properties
are found.
'''
class PassivePropertiesModule():
    def __init__(self, train_cell: TrainCell, sim_params: SimulationParameters, trace_filepath, known_passive_props: PassiveProperties):
        self.train_cell = train_cell
        self.sim_params = sim_params
        self.trace_filepath = trace_filepath
        self.known_passive_props = known_passive_props
    
    '''
    set_passive_properties
    The main method for calculating the passive properties of a cell given a negative current
    injection
    '''
        
    def set_passive_properties(self):
        try:
            os.system(f"nrnivmodl {self.train_cell.path_to_mod_files} > /dev/null 2>&1")

            time.sleep(2)

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
      
            dp = DataProcessor()

            if os.path.exists(self.trace_filepath):
                dataset = np.loadtxt(self.trace_filepath, delimiter=',', skiprows=1)
                
                V = dataset[:,0]
                
                train_cell_copy = TrainCell(
                    path_to_hoc_file=self.train_cell.path_to_hoc_file,
                    path_to_mod_files=self.train_cell.path_to_mod_files,
                    cell_name=self.train_cell.cell_name,
                    active_channels=self.train_cell.active_channels,
                    passive_properties=self.train_cell.passive_properties
                )
                
                I_tend = self.sim_params.CI[0].delay + self.sim_params.CI[0].dur
                props = dp.calculate_passive_properties(V, 
                                                        train_cell_copy,
                                                        self.sim_params.h_dt,
                                                        I_tend,
                                                        self.sim_params.CI[0].delay,
                                                        self.sim_params.CI[0].amp,
                                                        self.known_passive_props)
                
                self.train_cell.passive_properties = props
            
            else:
                print("Could not find voltage trace file. Cannot set passive properties.")
                print("To manually set known passive properties:")
                print("train_cell.passive_properties = PassiveProperties(...)")
                    
                    
        except Exception as e:
            print(f"An error occurred while loading mod files: {e}")
            raise
        
        finally:
            try:
                shutil.rmtree("x86_64")
            except OSError as e:
                print(f"Error removing x86_64 directory: {e}")
                