import os
import numpy as np

from act.act_types import SimulationParameters, OptimizationParameters, ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection
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
            if isinstance(current_inj, ConstantCurrentInjection):
                CI = [ConstantCurrentInjection
                      (
                        amp = current_inj.amp,
                        dur = current_inj.dur,
                        delay = current_inj.delay,
                        lto_hto = current_inj.lto_hto
                      )
                ]
            elif isinstance(current_inj, RampCurrentInjection):
                CI = [RampCurrentInjection
                      (
                          amp_start = current_inj.amp_incr,
                          amp_incr = current_inj.amp_incr,
                          num_steps = current_inj.num_steps,
                          step_time = current_inj.step_time,
                          dur = current_inj.dur,
                          delay = current_inj.delay,
                          lto_hto = current_inj.lto_hto
                      ) 
                ]
            elif isinstance(current_inj, GaussianCurrentInjection):
                CI = [GaussianCurrentInjection
                      (
                          amp_mean = current_inj.amp_mean,
                          amp_std = current_inj.amp_std,
                          dur = current_inj.dur,
                          delay = current_inj.delay,
                          lto_hto = current_inj.lto_hto
                      ) 
                ]
                
            simulator.submit_job(
                self.target_cell,
                SimulationParameters(
                    sim_name = self.job_name,
                    sim_idx=i,
                    h_v_init = self.sim_params.h_v_init,   # (mV)
                    h_tstop = self.sim_params.h_tstop,     # (ms)
                    h_dt = self.sim_params.h_dt,           # (ms)
                    h_celsius = self.sim_params.h_celsius, # (deg C)
                    CI = CI,
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
        labels = dataset[:, :, 3]
        
        labels_expanded = labels[..., np.newaxis]  
        csv_data = np.concatenate((voltage_current, labels_expanded), axis=2)
        
        num_samples, num_traces, _ = csv_data.shape
        flat_array = csv_data.reshape(num_samples * num_traces, 3)
        
        os.makedirs(self.output_folder_name, exist_ok=True)
        np.savetxt(self.output_folder_name + filename, flat_array, delimiter=',', header='Voltage,Current,Labels', comments='')

