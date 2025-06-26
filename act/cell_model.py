import numpy as np
import functools
from neuron import h

from act.types import SettablePassiveProperties

# Utility functions to get and set conductances with suffix.names
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)

class ACTCellModel:

    def __init__(
            self, 
            cell_name: str = None,
            path_to_hoc_file: str = None, 
            path_to_mod_files: str = None,
            passive: list = None,
            active_channels: list = None
            ):
        """
        Initialize a cell model for simulation or optimization.

        Parameters:
        ----------
        cell_name: str, default = None
            Cell name.
        
        path_to_hoc_file: str, default = None
            Path to the hoc template.
        
        path_to_mod_files: str, default = None
            Path to the mod files directory.
            
        passive: list, default = None
            Names of the (1) leak channel g_bar, (2) leak channel reversal potential and (3) h channel g_bar (in THIS order) as stated in the hoc file.

        active_channels: list, default = None
            Names of the active channel variables (in ANY order) as stated in the hoc file.
        """

        # Build cell info
        self.path_to_hoc_file = path_to_hoc_file
        self.path_to_mod_files = path_to_mod_files
        self.cell_name = cell_name

        # Channels
        self.passive = passive.copy()
        self.active_channels = active_channels.copy()

        self._set_g_to = {}
        self.spp = None

        # Minimal morphology
        self.all = None
        self.soma = None

        # Current injections
        self.CI = []

        # Recorders
        self.t = None
        self.V = None
        self.I = None

        self._custom_cell_builder = None
        self.sim_index = None

        # Predictions
        self.prediction = {ch : None for ch in self.active_channels}
    
    # ----------
    # Channels
    # ----------

    def set_g_bar(self, g_names: list, g_values: list) -> None:
        """
        Set each conductance in g_names to a corresponding value in g_values at simulation time.

        Parameters:
        -----------
        g_names: list
            Conductance variable name as found in the modfiles.
        
        g_values: list
            Conductance values to set.
        
        Returns:
        ----------
        None
        """
        for g_name, g_value in zip(g_names, g_values):
            self._set_g_to[g_name] = g_value
    
    def _set_g_bar(self) -> None:
        """
        Set conductances during NEURON runtime.

        Parameters:
        -----------
        g_names: list
            Conductance variable name as found in the modfiles.
        
        g_values: list
            Conductance values to set.
        
        Returns:
        ----------
        None
        """
        for sec in self.all:
            for g_name, g_value in self._set_g_to.items():
                _rsetattr(sec(0.5), g_name, g_value)
                    
    
    def set_passive_properties(self, spp: SettablePassiveProperties) -> None:
        """
        Set passive properties at simulation time.

        Parameters:
        -----------
        spp: SettablePassiveProperties
            Passive properties to set.
        """
        self.spp = spp

    # ----------
    # Build cell
    # ----------

    def _get_soma_area(self) -> float:
        """
        Compute the area of the soma in cm2.
        
        Returns:
        ----------
        soma_area: float
            Computed area in cm2.
        """
        soma_area = 0 # (um2)
        for segment in self.soma[0]:
            segment_area = h.area(segment.x, sec = self.soma[0])
            soma_area += segment_area
        return soma_area * 1e-8 # (um2 -> cm2)
    
    def _get_total_area(self) -> float:
        """
        Compute the total cell area in cm2.
        
        Returns:
        ----------
        total_area: float
            Computed area in cm2.
        """
        total_area = 0 # (um2)
        for sec in self.all:
            for segment in sec:
                segment_area = h.area(segment.x, sec = sec)
                total_area += segment_area
        return total_area * 1e-8 # (um2 -> cm2)

    def _build_cell(self, sim_index: int, print_soma_area = False) -> None:
        """
        Build the NEURON cell.

        Parameters:
        -----------
        sim_index: int
            Simulation index

        print_soma_area: bool, default = False
            If true, prints out the soma area.
        
        Returns:
        ----------
        None
        """
        self.sim_index = sim_index

        # Get the cell hoc object
        if self._custom_cell_builder is not None:
            hoc_cell = self._custom_cell_builder()
        else:
            h.load_file(self.path_to_hoc_file)
            hoc_cell = getattr(h, self.cell_name)()

        # Soma and all must exist in any cell
        self.soma = hoc_cell.soma
        try:
            self.all = list(hoc_cell.all)
        except:
            self.all = []
            for sec in hoc_cell.allsec():
                self.all.append(sec)

        # Report soma area
        if print_soma_area:
            print(f"Soma diam (um): {self.soma[0].diam}")
            print(f"Soma L (um): {self.soma[0].L}")
            print(f"Soma area (cm2): {self._get_soma_area()}")
            print(f"Total area (cm2): {self._get_total_area()}")
        
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
        Set a custom cell builder at simulaiton time.

        Parameters:
        -----------
        cell_builder: callable
            Callable that returns a hoc cell object.

        Returns:
        ----------
        None
        """
        self._custom_cell_builder = cell_builder
    
    # ----------
    # Current injection
    # ----------

    def _add_constant_CI(self, amp: float, dur: int, delay: int, t_stop: int, dt: float) -> None:
        """
        Set a constant current injection. Note that t_stop and dt parameters are required only for internal self.I construction.
        
        Parameters:
        -----------
        amp: float
            Injection amplitude (nA).
        
        dur: int
            Injection duration (ms).
        
        delay: int
            Injection delay (ms).
        
        t_stop: int
            Total simulation time (ms).
        
        dt: float
            Timestep (ms).
        
        Returns:
        ----------
        None
        """
        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp
        inj.dur = dur
        inj.delay = delay
        self.CI.append(inj)
        
        delay_steps = int(delay / dt)
        dur_steps = int(dur / dt)
        remainder_steps = int((t_stop - delay - dur) / dt)

        self.I = np.array([0.0] * delay_steps + [amp] * dur_steps + [0.0] * remainder_steps)
        
    
    def _add_ramp_CI(
            self, 
            start_amp: float, 
            amp_incr: float, 
            num_steps: int, 
            dur: int, 
            final_step_add_time: int,
            delay: int, 
            t_stop: int, 
            dt: float) -> None:
        """
        Set a ramp current injection. Note that t_stop and dt parameters are required only for internal self.I construction.
        
        Parameters:
        -----------
        start_amp: float
            Initial injection amplitude (nA).
            
        amp_incr: float
            Amplitude increase per step (nA).
        
        num_steps: int
            Number of steps.
        
        dur: int
            Total duration of current injection (ms).

        final_step_add_time: int
            How long to prolong the final step for.
        
        delay: int
            Injection delay (ms).
        
        t_stop: int
            Total simulation time (ms).
        
        dt: float
            Timestep (ms).
        
        Returns:
        ----------
        None
        """

        # No injection in the beginning
        I = [0] * delay
        
        # Accumulate delay for each step
        total_delay = delay

        amp = start_amp
        step_time = dur / num_steps

        for _ in range(num_steps):
            self._add_constant_CI(amp, step_time, total_delay, -1, -1)
            I += [amp] * int(step_time / dt)
            total_delay += step_time
            amp += amp_incr

        # Prolong the final step
        amp = amp - amp_incr
        self._add_constant_CI(amp, final_step_add_time, total_delay, -1, -1)
        total_delay += final_step_add_time
        I += [amp] * int(final_step_add_time / dt)

        # Record the no-injection part
        remainder_no_injection = int((t_stop - total_delay) / dt)
        I += [0] * int(remainder_no_injection / dt)
        
        self.I = np.array(I)


    def _add_gaussian_CI(self, amp_mean: float, amp_std: float, dur: int, delay: int, sim_time: int, dt: float, random_state: np.random.RandomState) -> None:
        """
        Sets the cell's gaussian current injection
        
        Parameters:
        -----------
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
        
        Returns:
        ----------
        None
        """
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

    # ----------
    # Outputs
    # ----------

    def get_output(self) -> tuple:
        """
        Gets the output of the simulator.
        
        Returns:
        ----------
        V: np.ndarray of shape (T, 1) #TODO: check shapes of all returns here; might be (T,) or (1, T)
            Voltage trace.
        
        I: np.ndarray of shape  (T, 1)
            Current injection trace.
        
        g_values: np.ndarray of shape (len_active_channels, 1)
            Conductance values (NaN padded).
        """
        g_values = []
        for channel in self.active_channels:
            g_values.append(_rgetattr(self.soma[0](0.5), channel))

        return self.V.as_numpy().flatten(), self.I.flatten(), np.array(g_values).flatten()