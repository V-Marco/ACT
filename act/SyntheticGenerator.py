import os
import numpy as np

from act.act_types import Cell, SimulationParameters, OptimizationParameters, PassiveProperties, SimParams
from act.cell_model import TargetCell,  ModuleParameters
from act.simulator import Simulator
from act.DataProcessor import DataProcessor

class SyntheticGenerator:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), "model", params['module_folder_name']) + "/"
        self.target_cell: TargetCell = params["cell"]
        self.sim_params: SimParams = params['sim_params']
        self.optim_params: OptimizationParameters = params['optim_params']
        
        sim_folder = ""
        for i, intensity in enumerate(self.sim_params['CI_amps']):
            sim_folder = sim_folder + str(intensity) + "_"
            
        self.job_name = "synthetic_" + sim_folder

        
    def generate_synthetic_target_data(self, filename):

        self.simulate_target_cell(self.target_cell)
        
        self.save_voltage_current_to_csv(filename)

    def simulate_target_cell(self, target_cell: TargetCell):
        # Simulate voltage traces
        simulator = Simulator(self.output_folder_name)
        print(f"Blocking: {self.optim_params['blocked_channels']}")
        for i, intensity in enumerate(self.sim_params['CI_amps']):

            target_cell.block_channels(self.sim_params, self.optim_params['blocked_channels'])
            simulator.submit_job(
                target_cell,
                SimulationParameters(
                    sim_name = self.job_name,
                    sim_idx=i,
                    h_v_init = self.sim_params['h_v_init'], # (mV)
                    h_tstop = self.sim_params['h_tstop'],  # (ms)
                    h_dt = self.sim_params['h_dt'], # (ms)
                    h_celsius = self.sim_params['h_celsius'], # (deg C)
                    CI = {
                        "type": self.sim_params['CI_type'],
                        "amp": intensity,
                        "dur": self.sim_params['CI_dur'],
                        "delay": self.sim_params['CI_delay']
                    },
                    set_g_to=self.sim_params['set_g_to']
                )
            )
        

        simulator.run(target_cell.mod_folder)

        
        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + self.job_name)

    def save_voltage_current_to_csv(self, filename):

        dataset = np.load(self.output_folder_name + self.job_name + "/combined_out.npy")
        voltage_current = dataset[:, :, :2]  # Assuming the first two slices are voltage and current
        
        # Flatten array to save CSV
        num_samples, num_traces, _ = voltage_current.shape
        flat_array = voltage_current.reshape(num_samples * num_traces, 2)
        
        if not os.path.exists(self.output_folder_name + "target/"):
            os.makedirs(self.output_folder_name + "target/")
        np.savetxt(self.output_folder_name + "target/" + filename, flat_array, delimiter=',', header='Voltage,Current', comments='')

