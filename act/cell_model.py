import numpy as np
import functools
from neuron import h
from act.act_types import SimulationParameters, SettablePassiveProperties

# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

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
            passive: list = None,
            active_channels: list = None,
            prediction: dict = None
            ):
        '''
        Parameters:
        ----------
        cell_name: str
            Cell name
        
        path_to_hoc_file: str
            Path to hoc template
        
        path_to_mod_files: str
            Path to mod files directory
            
        passive: list
            Names of the (1) leak channel g_bar, (2) leak channel reversal potential and (3) h channel g_bar (in THIS order) as stated in the hoc file.

        active_channels: list
            Names of the active channel variables (in ANY order) as stated in the hoc file.
        
        prediction: dict
            A dictionary of the channel conductance name (used in mod files) and respective predicted values
        
        Returns:
        ----------
        None
        '''

        # Build cell info
        self.path_to_hoc_file = path_to_hoc_file
        self.path_to_mod_files = path_to_mod_files
        self.cell_name = cell_name
        self.all = None
        self.prediction = prediction
        

        # Channels
        self.passive = passive.copy()
        self.active_channels = active_channels.copy()
        self.set_g_to = ()
        self._overridden_channels = {}
        self.spp = None

        # Minimal morphology
        self.soma = None

        # Current injections
        self.CI = []

        # Recorders
        self.t = None
        self.V = None
        self.I = None

        self._custom_cell_builder = None
        self.lto_hto = 0
        self.sim_index = None
    
    # ----------
    # Channels
    # ----------

    def set_g_bar(self, g_names: list, g_values: list) -> None:
        """
        Stores conductance values to a list in this cell's instance
        Parameters:
        -----------
        self
        
        g_names: list
            Conductance variable name (found in mod files)
        
        g_values: list
            Conductance values
        
        Returns:
        ----------
        None
        """
        #print(f"gnames: {g_names}, gvalues: {g_values}")
        self.set_g_to = (g_names, g_values)
    
    def _set_g_bar(self, g_names: list, g_values: list) -> None:
        """
        A function that actually sets the g_bar values to the NEURON cell during sim runtime.
        Parameters:
        -----------
        self
        
        g_names: list
            Conductance variable name (found in mod files)
        
        g_values: list
            Conductance values
        
        Returns:
        ----------
        None
        """
        for sec in self.all:
            for index, key in enumerate(g_names):
                if g_values[index]:
                    rsetattr(sec(0.5), key, g_values[index])
    
    def set_passive_properties(self, spp: SettablePassiveProperties) -> None:
        """
        Setter method for passive properties on this cell
        Parameters:
        -----------
        self
        
        spp: SettablePassiveProperties
        
        Returns:
        ----------
        None
        """
        self.spp = spp

    def block_channels(self, channels: list) -> None:
        """
        Setter method for passive properties on this cell
        Parameters:
        -----------
        self
        
        spp: SettablePassiveProperties
        
        Returns:
        ----------
        None
        """
        for channel in channels:
            self._overridden_channels[channel] = 0.0

    # ----------
    # Build cell
    # ----------

    def _get_soma_area(self) -> float:
        """
        Getter method for the surface area of the soma
        Parameters:
        -----------
        self
        
        Returns:
        ----------
        soma_area: float
        """
        soma_area = 0
        for segment in self.soma[0]:
            segment_area = h.area(segment.x, sec = self.soma[0])
            soma_area += segment_area
        return soma_area * 1e-8 # (cm2)

    def _build_cell(self, sim_index: int) -> None:
        """
        Builds the NEURON cell 
        Parameters:
        -----------
        self
        
        sim_index: int
            Simulation index
        
        Returns:
        ----------
        None
        """
        self.sim_index = sim_index
        if self._custom_cell_builder is not None:
            hoc_cell = self._custom_cell_builder()
        else:
            h.load_file(self.path_to_hoc_file)
            hoc_cell = getattr(h, self.cell_name)()

        # Soma must exist in any cell
        self.all = list(hoc_cell.all)
        self.soma = hoc_cell.soma

        # Report soma area
        #print(f"Soma area (cm2): {self._get_soma_area()}")
        
        # Update passive properties if needed
        if self.spp is not None:
            self.soma[0].cm = self.spp.Cm
            setattr(self.soma[0], self.passive[0], self.spp.g_bar_leak)
            setattr(self.soma[0], self.passive[1], self.spp.e_rev_leak)
            setattr(self.soma[0], self.passive[2], self.spp.g_bar_h)

        # Set recorders
        self.t = h.Vector().record(h._ref_t)
        self.V = h.Vector().record(self.soma[0](0.5)._ref_v)

    def set_custom_cell_builder(self, cell_builder: callable) -> None:
        """
        Sets a custom cell builder 
        Parameters:
        -----------
        self
        
        cell_builder: callable
        
        Returns:
        ----------
        None
        """
        self._custom_cell_builder = cell_builder

    def get_output(self) -> tuple:
        """
        Gets the output of the simulator
        Parameters:
        -----------
        self
        
        Returns:
        ----------
        V: np.ndarray
            Voltage trace
        
        I: np.ndarray
            Current injection trace
        
        g_values: np.ndarray
            Conductance values (Nan padded)
        
        sim_index: int
            Simulation indext
        
        lto_hto: int
            0 for not low/high threshold oscillation, 1 for either low or high threshold oscillations
        """
        g_values = []
        for channel in self.active_channels:
            g_values.append(rgetattr(self.soma[0](0.5), channel))

        return self.V.as_numpy().flatten(), self.I.flatten(), np.array(g_values).flatten(), self.sim_index, self.lto_hto
    
    # ----------
    # Current injection
    # ----------

    def _add_constant_CI(self, amp: float, dur: int, delay: int, sim_time: int, dt: float, lto_hto: int) -> None:
        """
        Sets the cell's constant current injection
        Parameters:
        -----------
        self
        
        amp: float
            Amps (nA)
        
        dur: int
            Duration of current injection (ms)
        
        delay: int
            Delay (ms)
        
        sim_time: int
            Total simulation time (ms)
        
        dt: float
            Timestep (ms)
        
        lto_hto: int
            0 for not low/high threshold oscillation, 1 for either low or high threshold oscillations
        
        Returns:
        ----------
        None
        """
        self.lto_hto = lto_hto
        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp; inj.dur = dur; inj.delay = delay
        self.CI.append(inj)
        
        delay_steps = int(delay / dt)
        dur_steps = int(dur / dt)
        remainder_steps = int((sim_time - delay - dur) / dt)

        self.I = np.array([0.0] * delay_steps + [amp] * dur_steps + [0.0] * remainder_steps)
        
    
    def _add_ramp_CI(self, start_amp: float, amp_incr: float, num_steps: int, step_time: float, dur: int, delay: int, sim_time: int, dt: float, lto_hto) -> None:
        """
        Sets the cell's ramp current injection
        Parameters:
        -----------
        self
        
        start_amp: float
            Starting Amps (nA)
            
        amp_incr: float
            How much the current injection increases each step
        
        num_steps: int
            Number of step increases in current injection trace
        
        step_time: float
            Amount of time for each current injection step (ms)
        
        dur: int
            Duration of current injection (ms)
        
        delay: int
            Delay (ms)
        
        sim_time: int
            Total simulation time (ms)
        
        dt: float
            Timestep (ms)
        
        lto_hto: int
            0 for not low/high threshold oscillation, 1 for either low or high threshold oscillations
        
        Returns:
        ----------
        None
        """
        
        self.lto_hto = lto_hto
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


    def _add_gaussian_CI(self, amp_mean: float, amp_std: float, dur: int, delay: int, sim_time: int, dt: float, random_state: np.random.RandomState, lto_hto) -> None:
        """
        Sets the cell's ramp current injection
        Parameters:
        -----------
        self
        
        amp_mean: float
            Mean Amps (nA)
            
        amp_std: float
            Mean Amps (nA)
        
        dur: int
            Duration of current injection (ms)
        
        delay: int
            Delay (ms)
        
        sim_time: int
            Total simulation time (ms)
            
        dt: float
            Timestep (ms)
            
        random_state: np.random.RandomState
            Random seed
        
        lto_hto: int
            0 for not low/high threshold oscillation, 1 for either low or high threshold oscillations
        
        Returns:
        ----------
        None
        """
        self.lto_hto = lto_hto
        total_delay = delay
        I = [0] * total_delay

        for _ in range(dur):
            amp = float(random_state.normal(amp_mean, amp_std))
            self.add_constant_CI(amp, 1, total_delay)
            I = I + [amp]
            total_delay += 1
            
        remainder_no_injection = sim_time - dur - delay
        I += [0.0] * (remainder_no_injection / dt)
        
        self.I = np.array(I)