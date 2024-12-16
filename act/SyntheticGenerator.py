import os
import numpy as np

from act.act_types import SimulationParameters, OptimizationParameters, CurrentInjection
from act.cell_model import TargetCell
from act.module_parameters import ModuleParameters
from act.simulator import ACTSimulator
from act.DataProcessor import DataProcessor

# A class to generate synthetic data from a cell to test the Automatic Cell Tuner

class SyntheticGenerator:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), params.module_folder_name) + "/"
        self.target_cell: TargetCell = params.cell
        print(params.cell)
        self.sim_params: SimulationParameters = params.sim_params
        self.optim_params: OptimizationParameters = params.optim_params
        
        sim_folder = ""
        for i, current_injection in enumerate(self.sim_params.CI):
            sim_folder = sim_folder + "_" + str(current_injection.amp)
            
        self.job_name = "synthetic" + sim_folder

    '''
    generate_synthetic_target_data
    The main method for simulating a cell and saving the data to CSV format
    (simulating how a user would normally come at this pipeline with CSV data)
    '''
        
    def generate_synthetic_target_data(self, filename):

        self.simulate_target_cell()
        
        self.save_voltage_current_to_csv(filename)
    
    '''
    simulate_target_cell
    Simulates a cell given a set of simulation parameters
    '''

    def simulate_target_cell(self):
        simulator = ACTSimulator(self.output_folder_name)
        for i, current_inj in enumerate(self.sim_params.CI):
            simulator.submit_job(
                self.target_cell,
                SimulationParameters(
                    sim_name = self.job_name,
                    sim_idx=i,
                    h_v_init = self.sim_params.h_v_init,   # (mV)
                    h_tstop = self.sim_params.h_tstop,     # (ms)
                    h_dt = self.sim_params.h_dt,           # (ms)
                    h_celsius = self.sim_params.h_celsius, # (deg C)
                    CI = [CurrentInjection
                    (
                        type = current_inj.type,    
                        amp = current_inj.amp,
                        dur = current_inj.dur,
                        delay = current_inj.delay
                    )],
                    set_g_to=self.sim_params.set_g_to
                )
            )
        
        simulator.run_jobs()

        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + self.job_name)

    '''
    save_voltage_current_to_csv
    Takes the .npy file output from the ACTSimulator and converts the data to CSV.
    '''
    def save_voltage_current_to_csv(self, filename):

        dataset = np.load(self.output_folder_name + self.job_name + "/combined_out.npy")
        voltage_current = dataset[:, :, :2]
        
        num_samples, num_traces, _ = voltage_current.shape
        flat_array = voltage_current.reshape(num_samples * num_traces, 2)
        
        os.makedirs(self.output_folder_name, exist_ok=True)
        np.savetxt(self.output_folder_name + filename, flat_array, delimiter=',', header='Voltage,Current', comments='')

