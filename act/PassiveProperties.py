
import os

from act.cell_model import TrainCell, TargetCell
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from act.ACTModule import ACTModule
from act.act_types import ModuleParameters, Cell, SimulationParameters, OptimizationParameters, OptimizationParam, PassiveProperties, SimParams
from act.cell_model import TargetCell, TrainCell
from act.simulator import Simulator
from act.DataProcessor import DataProcessor

from act.optimizer import RandomForestOptimizer
from act.Metrics import Metrics

class PassiveProperties(ACTModule):
    def __init__(self, act_module_params, passive_property_params):
        
        super().__init__(act_module_params)
        self.child_attr = passive_property_params
        
    def get_passive_properties(self, target_cell):

        dp = DataProcessor()

        if os.path.exists(self.target_traces_file) and self.CI_amps[0] < 0:
            # Load in the user provided negative current injection
            pass
            cell_area = dp.get_surface_area(target_cell) * 1e-8
            props = dp.calculate_passive_properties(V, self.sim_params['h_dt'],self.sim_params['h_tstop'],self.sim_params['CI_delay'],self.sim_params['CI_amps'][0],cell_area,self.leak_conductance_variable)
        else:
            # Simulating a negative current injection will only offer passive properties of the
            # default values. These are automatically set in the template.hoc.
            '''
            (
            V,
            dt, 
            h_tstop, 
            I_tstart, 
            I_intensity,
            cell_area
            ) = dp.simulate_negative_CI(self.output_folder_name, target_cell, self.leak_conductance_variable)
            props = dp.calculate_passive_properties(V, dt,h_tstop,I_tstart,I_intensity,cell_area,self.leak_conductance_variable)
            '''
