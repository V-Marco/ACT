from neuron import h
import numpy as np
import os

from act.act_types import PassiveProperties, OptimizationParameters, SimulationParameters
from typing import List, TypedDict

class ACTCellModel:

    def __init__(self, hoc_file: str, mod_folder: str, cell_name: str, g_names: list, passive_properties: PassiveProperties = None, cell_area = None, predicted_g: dict = None):

        # Hoc cell
        self.hoc_file = hoc_file
        self.mod_folder = mod_folder
        self.cell_name = cell_name
        self.passive_properties = passive_properties
        self.cell_area = cell_area
        self.predicted_g = predicted_g

        # Current injection objects
        self.CI = []

        # Conductances to consider
        self.g_names = g_names.copy()

        # Passive properties
        self.gleak_var = None
        self.g_bar_leak = None
        
    def set_surface_area(self):
        h.load_file('stdrun.hoc')

        # Initialize the cell
        h.load_file(self.hoc_file)
        init_Cell = getattr(h, self.cell_name)
        cell = init_Cell()

        # Print out all of the sections that are found
        section_list = list(h.allsec())
        print(f"Found {len(section_list)} section(s) in this cell. Calculating the total surface area of the cell.")

        cell_area = 0

        # Loop through all sections and segments and add up the areas
        for section in section_list:  
            for segment in section:
                segment_area = h.area(segment.x, sec=section)
                cell_area += segment_area
        self.cell_area = cell_area * 1e-8

    def _build_cell(self) -> None:

        # Load the .hoc file
        h.load_file(self.hoc_file)

        # Morphology
        hoc_cell = getattr(h, self.cell_name)()
        self.all = hoc_cell.all
        self.soma = hoc_cell.soma

        # Recorders
        self.t = h.Vector().record(h._ref_t)
        self.V = h.Vector().record(self.soma[0](0.5)._ref_v)
        self.I = []

    def get_output(self) -> None:
        g_values = []
        for channel in self.g_names:
            g_values.append(getattr(self.soma[0], channel))

        return self.V.as_numpy().flatten(), self.I.flatten(), np.array(g_values).flatten()
    
    def set_g(self, g_names: list, g_values: list, sim_params: SimulationParameters) -> None:
        sim_params['set_g_to'].append((g_names, g_values))

    def _set_g(self, g_names: list, g_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(g_names):
                setattr(sec, key, g_values[index])
    
    def block_channels(self, sim_params: SimulationParameters, blocked_channel_list = [], ):
        self.set_g(blocked_channel_list, [0.0 for _ in blocked_channel_list], sim_params)

    def _add_constant_CI(self, amp: float, dur: int, delay: int, sim_time) -> None:
        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp; inj.dur = dur; inj.delay = delay
        self.CI.append(inj)
        
        remainder = sim_time - delay - dur

        # Record injected current
        self.I = np.array([0.0] * delay + [amp] * dur + [0.0] * remainder)
    
    def _add_ramp_CI(self, start_amp: float, amp_incr: float, ramp_time: float, dur: int, delay: int) -> None:
        total_delay = delay
        amp = start_amp

        # Record injected current
        I = [0] * total_delay

        for _ in range(0, dur, ramp_time):
            self.add_constant_CI(amp, ramp_time, total_delay)
            I += [amp] * ramp_time
            total_delay += ramp_time
            amp += amp_incr
        
        self.I = np.array(I)


    def _add_gaussian_CI(self, amp_mean: float, amp_std: float, dur: int, delay: int, random_state: np.random.RandomState) -> None:
        total_delay = delay

        # Record injected current
        I = [0] * total_delay

        for _ in range(dur):
            amp = float(random_state.normal(amp_mean, amp_std))
            self.add_constant_CI(amp, 1, total_delay)
            I = I + [amp]
            total_delay += 1
        
        self.I = np.array(I)
    
    def set_passive_properties(self, passive_props: PassiveProperties) -> None:
        for sec in self.all:
            # Setting e_leak
            if (passive_props.V_rest) and (passive_props.leak_reversal_variable):
                setattr(sec, passive_props.leak_reversal_variable, passive_props.V_rest)
            else:
                print(
                    f"Skipping analytical setting of e_leak variable. V_rest and/or leak_reversal_variable not specified."
                )
                
            # Setting g_leak
            if (passive_props.g_bar_leak) and (passive_props.leak_conductance_variable):
                setattr(sec, passive_props.leak_conductance_variable, passive_props.g_bar_leak)
            elif (passive_props.R_in) and (passive_props.cell_area):
                g_bar_leak = (1/passive_props.R_in) / passive_props.cell_area
                setattr(sec, passive_props.leak_conductance_variable, g_bar_leak)
            else:
                print(
                    f"Skipping analytical setting of g_bar_leak variable. g_bar_leak, leak_conductance_variable, R_in, and/or cell_area not specified."
                )       
            
            if (passive_props.Cm):
                setattr(sec, "cm", passive_props.Cm)
            elif (passive_props.R_in) and (passive_props.tau):
                Cm = passive_props.tau * (1 / passive_props.R_in)
                setattr(sec, "cm", Cm)
            else:
                print(
                    f"Skipping analytical setting of cm. Cm, R_in and/or tau not specified."
                )  

class TargetCell(ACTCellModel):

    def __init__(self, hoc_file: str, mod_folder: str, cell_name: str, g_names: list = [], passive_properties: PassiveProperties = None):
        super().__init__(hoc_file, mod_folder, cell_name, g_names, passive_properties)

class TrainCell(ACTCellModel):

    def __init__(self, hoc_file: str, mod_folder: str, cell_name: str, g_names: list = [], passive_properties: PassiveProperties = None):
        super().__init__(hoc_file, mod_folder, cell_name, g_names, passive_properties)

                
class ModuleParameters(TypedDict):
    module_folder_name: str
    target_traces_file: str
    cell: ACTCellModel
    sim_params: SimulationParameters
    optim_params: OptimizationParameters