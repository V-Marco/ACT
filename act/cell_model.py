from neuron import h
import numpy as np
import os

from act import act_types

class ACTCellModel:

    def __init__(self, hoc_file: str, cell_name: str, g_names: list):

        # Hoc cell
        self.hoc_file = hoc_file
        self.cell_name = cell_name

        # Current injection objects
        self.CI = []

        # Conductances to consider
        self.g_names = g_names.copy()

        # Passive properties
        self.gleak_var = None
        self.g_bar_leak = None

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

    def _add_constant_CI(self, amp: float, dur: int, delay: int) -> None:
        inj = h.IClamp(self.soma[0](0.5))
        inj.amp = amp; inj.dur = dur; inj.delay = delay
        self.CI.append(inj)

        # Record injected current
        self.I = np.array([0.0] * delay + [amp] * dur)
    
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

class TargetCell(ACTCellModel):

    def __init__(self, hoc_file: str, cell_name: str, g_names: list):
        super().__init__(hoc_file, cell_name, g_names)

class TrainCell(ACTCellModel):

    def __init__(self, hoc_file: str, cell_name: str, g_names: list):
        super().__init__(hoc_file, cell_name, g_names)
        self.g_to_set_after_build = None

    def set_g(self, g_names: list, g_values: list) -> None:
        self.g_to_set_after_build = (g_names, g_values)

    def _set_g(self, g_names: list, g_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(g_names):
                setattr(sec, key, g_values[index])

    def set_passive_properties(self, passive_properties: act_types.PassiveProperties) -> None:
            
        V_rest = passive_properties.V_rest  # (mV)
        R_in = passive_properties.R_in  # (10^6 Ohm)
        tau = passive_properties.tau # (ms)

        gleak_var = passive_properties.leak_conductance_variable
        eleak_var = passive_properties.leak_reversal_variable

        # Assuming Vrest is within the range for ELeak
        # ELeak = Vrest
        if (V_rest) and (eleak_var):
            print(f"Setting {eleak_var} = {V_rest}")
            self.set_parameters(self.all, [eleak_var], [V_rest])
        else:
            print(
                f"Skipping analytical setting of e_leak variable. Cell v_rest and/or leak_reversal_variable not specified in config."
            )

        for sec in self.all:
            # Rin = 1 / (Area * g_bar_leak)
            # g_bar_leak = 1 / (Rin * Area)

            # Have to specify the location to get access to area
            area = sec(0.5).area() # (um2)
            
            if not R_in:
                print(f"Skipping analytical setting of gleak and cm variables. Cell r_in not specified in config.")
                return
            
            # [S / cm2] = 1 / (10^6 Ohm * um2) = 10^6 S / 10^{-8} cm2
            g_bar_leak = 1 / (R_in * area) * 1e2
            print(f"Setting {sec}.{gleak_var} = {g_bar_leak:.8f}")
            setattr(sec, gleak_var, g_bar_leak)

            if sec == self.soma: g_bar_leak = g_bar_leak

            if not tau:
                print(f"Skipping analytical setting of cm variable. Cell tau not specified in config.")
                return
            
            # tau = (R * C) = R * Area * cm
            # cm = tau * g_bar_leak * Area
            # cm/cm2 = (tau*g_bar_leak*Area)/Area = tau * g_bar_leak
            cm = tau * g_bar_leak * 1e3  # tau ms->s
            print(f"Setting {sec}.cm = {cm:.8f}")
            setattr(sec, "cm", cm)
                
