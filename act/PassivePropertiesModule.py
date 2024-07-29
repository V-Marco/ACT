
import os
import time
import sys
import shutil
from contextlib import contextmanager

from neuron import h
import numpy as np
from act.cell_model import TrainCell
from act.act_types import SimParams
from act.DataProcessor import DataProcessor

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

class PassivePropertiesModule():
    def __init__(self, train_cell: TrainCell, sim_params: SimParams, trace_filepath, leak_conductance_variable, leak_reversal_variable):
        self.train_cell = train_cell
        self.sim_params = sim_params
        self.trace_filepath = trace_filepath
        self.leak_conductance_variable = leak_conductance_variable
        self.leak_reversal_variable = leak_reversal_variable
        
    # have user provide path to negative current trace.
        
    def set_passive_properties(self):
        try:
            # Compile the modfiles and suppress output
            os.system(f"nrnivmodl {self.train_cell.mod_folder} > /dev/null 2>&1")

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
                    
            # CALCULATE THE PASSIVE PROPERTIES        
            dp = DataProcessor()

            if os.path.exists(self.trace_filepath):
                # Load in the user provided negative current injection
                dataset = np.loadtxt(self.trace_filepath, delimiter=',', skiprows=1)
                
                V = dataset[:,0]
                
                I_tend = self.sim_params['CI_delay'] + self.sim_params['CI_dur']
                props = dp.calculate_passive_properties(V, 
                                                        self.train_cell,
                                                        self.sim_params['h_dt'],
                                                        I_tend,
                                                        self.sim_params['CI_delay'],
                                                        self.sim_params['CI_amps'][0],
                                                        self.leak_conductance_variable,
                                                        self.leak_reversal_variable)
                
                props.g_bar_leak = round(props.g_bar_leak, 15)
                #props.Cm = 1
                
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


            
        
        
