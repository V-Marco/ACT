import numpy as np
import functools
from neuron import h
from act.act_types import PassiveProperties, SimulationParameters

'''
Defines ACTCellModel which is a parent class to TargetCell and TrainCell (also defined)
These classes hold vital information about a cell and the current injections applied to them.
'''

class ACTCellModel:

    def __init__(
            self, 
            cell_name: str = None,
            path_to_hoc_file: str = None, 
            path_to_mod_files: str = None, 
            active_channels: dict = {},
            passive_properties: PassiveProperties = None
            ):

        self.path_to_hoc_file = path_to_hoc_file
        self.path_to_mod_files = path_to_mod_files
        self.cell_name = cell_name
        self.passive_properties = passive_properties
        self.active_channels = active_channels.copy()

        self.all = None
        self.soma = None

        self.CI = []

        self.t = None
        self.V = None
        self.I = None

        self._custom_cell_builder = None
        
    '''
    set_surface_area
    Finds the surface area of a cell and stores the value.
    '''
        
    def set_surface_area(self):
        section_list = list(self.all)
        print(f"Found {len(section_list)} section(s) in this cell. Calculating the total surface area of the cell.")

        cell_area = 0

        for section in section_list:  
            for segment in section:
                segment_area = h.area(segment.x, sec=section)
                cell_area += segment_area
        self.cell_area = cell_area * 1e-8 # cm^2
        
    '''
    _build_cell
    Builds the NEURON cell and sets up recorders
    '''

    def _build_cell(self) -> None:
        if self._custom_cell_builder is not None:
            hoc_cell = self._custom_cell_builder()
        else:
            h.load_file(self.path_to_hoc_file)
            hoc_cell = getattr(h, self.cell_name)()

        self.all = list(hoc_cell.all)
        self.soma = hoc_cell.soma

        self.t = h.Vector().record(h._ref_t)
        self.V = h.Vector().record(self.soma[0](0.5)._ref_v)
    
    '''
    set_custom_cell_builder
    Allows users to change the builder
    '''

    def set_custom_cell_builder(self, cell_builder: callable) -> None:
        self._custom_cell_builder = cell_builder
        
    '''
    get_output
    Gets the Voltage trace, current injection trace, and conductance set label from the simulation
    '''

    def get_output(self) -> tuple:
        # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return functools.reduce(_getattr, [obj] + attr.split('.'))
        
        g_values = []
        for channel in self.active_channels.values():
            g_values.append(rgetattr(self.soma[0](0.5), channel))

        return self.V.as_numpy().flatten(), self.I.flatten(), np.array(g_values).flatten()
    
    '''
    _add_constant_CI
    Creates a constant current injection
    '''

    def _add_constant_CI(self, amp: float, dur: int, delay: int, sim_time: int, dt: float) -> None:
        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp; inj.dur = dur; inj.delay = delay
        self.CI.append(inj)
        
        delay_steps = int(delay / dt)
        dur_steps = int(dur / dt)
        remainder_steps = int((sim_time - delay - dur) / dt)

        self.I = np.array([0.0] * delay_steps + [amp] * dur_steps + [0.0] * remainder_steps)
    
    '''
    _add_ram_CI
    Creates a ramp current injection
    '''
    
    def _add_ramp_CI(self, start_amp: float, amp_incr: float, num_steps: int, step_time: float, dur: int, delay: int, sim_time: int, dt: float) -> None:
        total_delay = delay
        amp = start_amp

        I = [0] * total_delay

        for _ in range(num_steps):
            self._add_constant_CI(amp, step_time, total_delay, sim_time, dt)
            I += [amp] * (step_time / dt)
            total_delay += step_time
            amp += amp_incr

        amp -= amp_incr
        remainder_inj_time = (dur + delay) - total_delay
        self._add_constant_CI(amp, remainder_inj_time, total_delay, sim_time, dt)
        I += [amp] * (remainder_inj_time / dt)
        
        remainder_no_injection = sim_time - dur - delay
        I += [amp - amp_incr] * (remainder_no_injection / dt)
        
        self.I = np.array(I)

    '''
    _add_gaussian_CI
    Creates a random current injection input
    '''

    def _add_gaussian_CI(self, amp_mean: float, amp_std: float, dur: int, delay: int, random_state: np.random.RandomState) -> None:
        total_delay = delay
        I = [0] * total_delay

        for _ in range(dur):
            amp = float(random_state.normal(amp_mean, amp_std))
            self.add_constant_CI(amp, 1, total_delay)
            I = I + [amp]
            total_delay += 1
        
        self.I = np.array(I)
    
    '''
    set_g
    Records a list of maximal conductance densities for ion channels (to be referenced later)
    '''

    def set_g(self, g_names: list, g_values: list, sim_params: SimulationParameters) -> None:
        sim_params.set_g_to.append((g_names, g_values))
    
    '''
    _set_g
    Interacts with the NEURON simulator to actually set the maximal conductance density for an
    ion channel.
    '''

    def _set_g(self, g_names: list, g_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(g_names):
                setattr(sec, key, g_values[index])
    
    '''
    block_channels
    An ease of use function to set the maximal conductance density of a selected channel
    to 0.0.
    '''
    
    def block_channels(self, sim_params: SimulationParameters, blocked_channel_list = [], ):
        if not blocked_channel_list == None and not len(blocked_channel_list) == 0:
            self.set_g(blocked_channel_list, [0.0 for _ in blocked_channel_list], sim_params)
    
    '''
    set_passive_properties
    Interacts with NEURON to actually set passive property values for a cell.
    '''
    
    
    def set_passive_properties(self, passive_props: PassiveProperties) -> None:
        for sec in self.all:
            if passive_props.V_rest and passive_props.leak_reversal_variable:
                setattr(sec, passive_props.leak_reversal_variable, passive_props.V_rest)
            else:
                print(f"Skipping analytical setting of e_leak variable. V_rest and/or leak_reversal_variable not specified.")
                
            if passive_props.g_bar_leak and passive_props.leak_conductance_variable:
                setattr(sec, passive_props.leak_conductance_variable, passive_props.g_bar_leak)
            elif passive_props.R_in and passive_props.cell_area:
                g_bar_leak = (1/passive_props.R_in) / passive_props.cell_area
                setattr(sec, passive_props.leak_conductance_variable, g_bar_leak)
            else:
                print(f"Skipping analytical setting of g_bar_leak variable. g_bar_leak, leak_conductance_variable, R_in, and/or cell_area not specified.")       
            
            if passive_props.Cm:
                setattr(sec, "cm", passive_props.Cm)
            elif passive_props.R_in and passive_props.tau:
                Cm = passive_props.tau * (1 / passive_props.R_in)
                setattr(sec, "cm", Cm)
            else:
                print(f"Skipping analytical setting of cm. Cm, R_in and/or tau not specified.") 
            
            if passive_props.h_conductance_variable and passive_props.g_bar_h:
                setattr(sec, passive_props.h_conductance_variable, passive_props.g_bar_h)
                
                
'''
TargetCell
A class to differentiate the Target Cell and the Train Cell (though functionally identical)
''' 

class TargetCell(ACTCellModel):

    def __init__(self, path_to_hoc_file: str, path_to_mod_files: str, cell_name: str, active_channels: list = [], passive_properties: PassiveProperties = None):
        super().__init__(path_to_hoc_file=path_to_hoc_file, path_to_mod_files=path_to_mod_files, cell_name=cell_name, active_channels=active_channels, passive_properties=passive_properties)

'''
TrainCell
A class to differentiate the Target Cell and the Train Cell (though functionally identical)
''' 

class TrainCell(ACTCellModel):

    def __init__(self, path_to_hoc_file: str, path_to_mod_files: str, cell_name: str, active_channels: list = [], passive_properties: PassiveProperties = None):
        super().__init__(path_to_hoc_file=path_to_hoc_file, path_to_mod_files=path_to_mod_files, cell_name=cell_name, active_channels=active_channels, passive_properties=passive_properties)