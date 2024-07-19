import os
import numpy as np

from act.act_types import ModuleParameters, Cell, SimulationParameters, OptimizationParameters, PassiveProperties, SimParams
from act.cell_model import TargetCell
from act.simulator import Simulator
from act.DataProcessor import DataProcessor

class SyntheticGenerator:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), "model", params['module_folder_name']) + "/"
        self.cell: Cell = params["cell"]
        self.target_traces_file = params["target_traces_file"]
        passive_props: PassiveProperties = params['passive_properties']
        self.sim_params: SimParams = params['sim_params']
        self.optim_params: OptimizationParameters = params['optim_params']
        
    def generate_synthetic_target_data(self):
        target_cell = TargetCell(
                hoc_file = self.cell['hoc_file'],
                mod_folder= self.cell['modfiles_folder'],
                cell_name = self.cell['cell_name'],
                g_names = self.cell['g_names']
            )
        self.simulate_target_cell(target_cell)
        
        self.save_voltage_current_to_csv("target_data.csv")

    def simulate_target_cell(self, target_cell):
        # Simulate voltage traces
        simulator = Simulator(self.output_folder_name)
        for i, intensity in enumerate(self.sim_params['CI_amps']):
            
            simulator.submit_job(
                target_cell,
                SimulationParameters(
                    sim_name = "synthetic_user_data",
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
                    }
                )
            )
        

        simulator.run(self.cell['modfiles_folder'])

        
        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + "synthetic_user_data")

    def save_voltage_current_to_csv(self, filename):

        dataset = np.load(self.output_folder_name + "synthetic_user_data/combined_out.npy")
        voltage_current = dataset[:, :, :2]  # Assuming the first two slices are voltage and current
        
        # Flatten array to save CSV
        num_samples, num_traces, _ = voltage_current.shape
        flat_array = voltage_current.reshape(num_samples * num_traces, 2)
        
        if not os.path.exists(self.output_folder_name + "target/"):
            os.makedirs(self.output_folder_name + "target/")
        
        np.savetxt(self.output_folder_name + "target/" + filename, flat_array, delimiter=',', header='Voltage,Current', comments='')

