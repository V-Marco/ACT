import os
import numpy as np

from act.act_types import SimulationParameters, OptimizationParameters, CurrentInjection
from act.cell_model import TargetCell
from act.module_parameters import ModuleParameters
from act.simulator import ACTSimulator
from act.DataProcessor import DataProcessor

class SyntheticGenerator:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), params.module_folder_name) + "/"
        self.target_cell: TargetCell = params.cell
        print(params.cell)
        self.sim_params: SimulationParameters = params.sim_params
        self.optim_params: OptimizationParameters = params.optim_params
        
        sim_folder = ""
        for i, intensity in enumerate(self.sim_params.CI.amps):
            sim_folder = sim_folder + "_" + str(intensity)
            
        self.job_name = "synthetic" + sim_folder

        
    def generate_synthetic_target_data(self, filename):

        self.simulate_target_cell()
        
        self.save_voltage_current_to_csv(filename)

    def simulate_target_cell(self):
        simulator = ACTSimulator(self.output_folder_name)
        for i, intensity in enumerate(self.sim_params.CI.amps):
            simulator.submit_job(
                self.target_cell,
                SimulationParameters(
                    sim_name = self.job_name,
                    sim_idx=i,
                    h_v_init = self.sim_params.h_v_init,   # (mV)
                    h_tstop = self.sim_params.h_tstop,     # (ms)
                    h_dt = self.sim_params.h_dt,           # (ms)
                    h_celsius = self.sim_params.h_celsius, # (deg C)
                    CI = CurrentInjection
                    (
                        type = self.sim_params.CI.type,    
                        amps = [intensity],
                        dur = self.sim_params.CI.dur,
                        delay = self.sim_params.CI.delay
                    ),
                    set_g_to=self.sim_params.set_g_to
                )
            )
        

        simulator.run(self.target_cell.mod_folder)

        
        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + self.job_name)

    def save_voltage_current_to_csv(self, filename):

        dataset = np.load(self.output_folder_name + self.job_name + "/combined_out.npy")
        voltage_current = dataset[:, :, :2]
        
        num_samples, num_traces, _ = voltage_current.shape
        flat_array = voltage_current.reshape(num_samples * num_traces, 2)
        
        os.makedirs(self.output_folder_name, exist_ok=True)
        np.savetxt(self.output_folder_name + filename, flat_array, delimiter=',', header='Voltage,Current', comments='')

