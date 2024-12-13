import numpy as np
import functools
from neuron import h

from act.act_types import PassiveProperties, SimulationParameters

class ACTCellModel:

    def __init__(
            self, 
            cell_name: str = None,
            path_to_hoc_file: str = None, 
            path_to_mod_files: str = None, 
            active_channels: list = [],
            passive_properties: PassiveProperties = None
            ):

        # Hoc cell
        self.path_to_hoc_file = path_to_hoc_file
        self.path_to_mod_files = path_to_mod_files
        self.cell_name = cell_name
        self.passive_properties = passive_properties

        # Conductances to consider
        self.active_channels = active_channels.copy()

        # Sections
        self.all = None
        self.soma = None

        # Current injection objects
        self.CI = []

        # Recorders
        self.t = None
        self.V = None
        self.I = None

        # Custom cell builder function
        self._custom_cell_builder = None
        
    def set_surface_area(self):
        # Print out all of the sections that are found
        section_list = list(self.all)
        print(f"Found {len(section_list)} section(s) in this cell. Calculating the total surface area of the cell.")

        cell_area = 0

        # Loop through all sections and segments and add up the areas
        for section in section_list:  
            for segment in section:
                segment_area = h.area(segment.x, sec=section)
                cell_area += segment_area
        self.cell_area = cell_area * 1e-8 # cm^2

    def _build_cell(self) -> None:

        # If there is a custom cell builder provided, use it
        if self._custom_cell_builder is not None:
            hoc_cell = self._custom_cell_builder()
        else: # Otherwise, use the standard build procedure
            # Load the .hoc file
            h.load_file(self.path_to_hoc_file)
            # Morphology
            hoc_cell = getattr(h, self.cell_name)()

        self.all = list(hoc_cell.all)
        self.soma = hoc_cell.soma

        # Recorders
        self.t = h.Vector().record(h._ref_t)
        self.V = h.Vector().record(self.soma[0](0.5)._ref_v)

    def set_custom_cell_builder(self, cell_builder: callable) -> None:
        self._custom_cell_builder = cell_builder
        
    def compute_surface_area(self) -> None:

        # Print out all of the sections that are found
        section_list = self.all
        print(f"Found {len(section_list)} section(s) in this cell. Calculating the total surface area of the cell.")

        # Loop through all sections and segments and add up the areas
        cell_area = 0
        for section in section_list:  
            for segment in section:
                segment_area = h.area(segment.x, sec = section)
                cell_area += segment_area
        self.passive_properties.cell_area = cell_area * 1e-8 # (cm2)
        print(f"Cell area is set to {self.passive_properties.cell_area} cm2.")

    def get_output(self) -> tuple:
        # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return functools.reduce(_getattr, [obj] + attr.split('.'))
        
        g_values = []
        for channel in self.active_channels:
            g_values.append(rgetattr(self.soma[0](0.5), channel))

        return self.V.as_numpy().flatten(), self.I.flatten(), np.array(g_values).flatten()

    ########## Current injection ##########

    def _add_constant_CI(self, amp: float, dur: int, delay: int, sim_time: int, dt: float) -> None:

        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp; inj.dur = dur; inj.delay = delay
        self.CI.append(inj)
        
        delay_steps = int(delay / dt)
        dur_steps = int(dur / dt)
        remainder_steps = int((sim_time - delay - dur) / dt)

        # Record injected current
        self.I = np.array([0.0] * delay_steps + [amp] * dur_steps + [0.0] * remainder_steps)
    
    def _add_ramp_CI(self, start_amp: float, amp_incr: float, num_steps: int, step_time: float, dur: int, delay: int, sim_time: int, dt: float) -> None:
        total_delay = delay
        amp = start_amp

        # Record injected current
        I = [0] * total_delay

        for _ in range(num_steps):
            self._add_constant_CI(amp, step_time, total_delay, sim_time, dt)
            I += [amp] * (step_time / dt)
            total_delay += step_time
            amp += amp_incr

        # Current injection plateau after ramp
        amp -= amp_incr # undo the last increase in for-loop
        remainder_inj_time = (dur + delay) - total_delay
        self._add_constant_CI(amp, remainder_inj_time, total_delay, sim_time, dt)
        I += [amp] * (remainder_inj_time / dt)
        
        # no current injection after plateau
        remainder_no_injection = sim_time - dur - delay
        I += [amp - amp_incr] * (remainder_no_injection / dt)
        
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

    ########## TODO -- potentially adjust to the new outline

    def set_g(self, g_names: list, g_values: list, sim_params: SimulationParameters) -> None:
        print(f"Setting {g_names} to {g_values}")
        sim_params.set_g_to.append((g_names, g_values))

    def _set_g(self, g_names: list, g_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(g_names):
                setattr(sec, key, g_values[index])
    
    def block_channels(self, sim_params: SimulationParameters, blocked_channel_list = [], ):
        if not blocked_channel_list == None and not len(blocked_channel_list) == 0:
            self.set_g(blocked_channel_list, [0.0 for _ in blocked_channel_list], sim_params)
    
    def set_passive_properties(self, passive_props: PassiveProperties) -> None:
        for sec in self.all:
            # Setting e_leak
            if (passive_props.V_rest) and (passive_props.leak_reversal_variable):
                #print(f"Setting {passive_props.leak_reversal_variable} = {passive_props.V_rest}")
                setattr(sec, passive_props.leak_reversal_variable, passive_props.V_rest)
            else:
                print(f"Skipping analytical setting of e_leak variable. V_rest and/or leak_reversal_variable not specified.")
                
            # Setting g_leak
            if (passive_props.g_bar_leak) and (passive_props.leak_conductance_variable):
                #print(f"Setting {passive_props.leak_conductance_variable} = {passive_props.g_bar_leak}")
                setattr(sec, passive_props.leak_conductance_variable, passive_props.g_bar_leak)
            elif (passive_props.R_in) and (passive_props.cell_area):
                g_bar_leak = (1/passive_props.R_in) / passive_props.cell_area
                #print(f"Setting {passive_props.leak_conductance_variable} = {g_bar_leak}")
                setattr(sec, passive_props.leak_conductance_variable, g_bar_leak)
            else:
                print(f"Skipping analytical setting of g_bar_leak variable. g_bar_leak, leak_conductance_variable, R_in, and/or cell_area not specified.")       
            
            if (passive_props.Cm):
                #print(f"Setting cm = {passive_props.Cm}")
                setattr(sec, "cm", passive_props.Cm)
            elif (passive_props.R_in) and (passive_props.tau):
                Cm = passive_props.tau * (1 / passive_props.R_in)
                #print(f"Setting cm = {Cm}")
                setattr(sec, "cm", Cm)
            else:
                print(f"Skipping analytical setting of cm. Cm, R_in and/or tau not specified.")  

class TargetCell(ACTCellModel):

    def __init__(self, path_to_hoc_file: str, path_to_mod_files: str, cell_name: str, active_channels: list = [], passive_properties: PassiveProperties = None):
        super().__init__(path_to_hoc_file=path_to_hoc_file, path_to_mod_files=path_to_mod_files, cell_name=cell_name, active_channels=active_channels, passive_properties=passive_properties)

class TrainCell(ACTCellModel):

    def __init__(self, path_to_hoc_file: str, path_to_mod_files: str, cell_name: str, active_channels: list = [], passive_properties: PassiveProperties = None):
        super().__init__(path_to_hoc_file=path_to_hoc_file, path_to_mod_files=path_to_mod_files, cell_name=cell_name, active_channels=active_channels, passive_properties=passive_properties)