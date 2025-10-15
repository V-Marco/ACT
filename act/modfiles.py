from dataclasses import dataclass
import numpy as np
import os
from scipy.optimize import curve_fit

@dataclass
class StateVariable:
    name: str
    power: int
    vhalf: float
    k: float
    tau: float

    def activation_function(self, v: np.ndarray):
        return self._activation_function(self.vhalf, self.k, v)

    def _activation_function(self, vhalf: float, k: float, v: np.ndarray):
        return 1.0 / (1.0 + np.exp(-(v - vhalf) / k))


@dataclass
class SegmentableStateVariable(StateVariable):
    segmentation_vhalf_bounds: tuple
    segmentation_k_bounds: tuple

    _segmented_v_threshold: float = None
    _segmented_vhalf: float = None
    _segmented_k: float = None
    _segmented_linear_slope: float = None
    _segmented_linear_intercept: float = None

    def segmented_activation_function(self, v: float | np.ndarray):
        # Check if segmented
        if self._segmented_v_threshold is None:
            raise ValueError(f"The state variable {self.name} has not been segmented yet. Use ACTSegmenter to perform segmentation.")
        
        if not isinstance(v, np.ndarray):
            v = np.array([v]).flatten()

        # Compute the full activation function
        saf = self._activation_function(self._segmented_vhalf, self._segmented_k, v)
        
        # Cut at threshold
        saf[v < self._segmented_v_threshold] = 0

        # Add the linear segment
        linear_segment_conditon = (v >= self._segmented_v_threshold) & (v < self._segmented_v_threshold + 3)
        saf[linear_segment_conditon] = self._segmented_linear_intercept + self._segmented_linear_slope * v[linear_segment_conditon]

        return saf
    

class ACTModfile:

    def __init__(
            self,
            channel_name: str,
            ion: str | None,
            gbar: float,
            e: float,
            state_variables: list[StateVariable | SegmentableStateVariable]):
        
        self.channel_name = channel_name
        self.ion = ion
        self.gbar = gbar
        self.e = e
        self.state_variables = state_variables
    
    def __repr__(self):
        return f"ACTModfile(channel_name = {self.channel_name}, e = {self.e}, gbar = {self.gbar})"
    
    def __getitem__(self, svar_name):
        for svar in self.state_variables:
            if svar.name == svar_name:
                return svar
            
    def to_mod(self, path_to_folder, segmented = False):
        # Build line-by-line
        modfile = []

        # Write the NEURON section
        modfile.append("NEURON {")
        modfile.append(f"\tSUFFIX {self.channel_name}")
        if self.ion is None:
            modfile.append(f"\tNONSPECIFIC_CURRENT i")
            modfile.append(f"\tRANGE i")
        else:
            modfile.append(f"\tUSEION {self.ion} WRITE i{self.ion}")
            modfile.append(f"\tRANGE i{self.ion}")
        modfile.append(f"\tRANGE gbar")
        modfile.append("}\n")

        # Write the UNITS section
        modfile.append("UNITS {")
        modfile.append("\t(mA) = (milliamp)")
        modfile.append("\t(mV) = (millivolt)")
        modfile.append("}\n")

        # Write PARAMETER section
        modfile.append("PARAMETER {")
        modfile.append(f"\tgbar = {self.gbar} (S/cm2)")
        if self.ion is None:
            modfile.append(f"\te{self.channel_name} = {self.e} (mV)")
        else:
            modfile.append(f"\te{self.ion} = {self.e} (mV)")
        modfile.append("}\n")

        # Write STATE section
        if len(self.state_variables) > 0:
            modfile.append("STATE {")
            for svar in self.state_variables:
                modfile.append(f"\t{svar.name}")
            modfile.append("}\n")

        # Write ASSIGNED section
        modfile.append("ASSIGNED {")
        modfile.append("\tv (mV)")
        if self.ion is None:
            modfile.append(f"\ti (mA/cm2)")
        else:
            modfile.append(f"\ti{self.ion} (mA/cm2)")
        for svar in self.state_variables:
             modfile.append(f"\t{svar.name}inf")
        modfile.append("}\n")

        # Write BREAKPOINT section
        modfile.append("BREAKPOINT {")
        if len(self.state_variables) > 0:
            modfile.append("\tSOLVE states METHOD cnexp")

        if self.ion is None:
            current_str = f"\ti = gbar * "
        else:
            current_str = f"\ti{self.ion} = gbar * "

        for svar in self.state_variables:
            for _ in range(svar.power):
                current_str += f"{svar.name} * "

        if self.ion is None:
            current_str += f"(v - e{self.channel_name})"
        else:
            current_str += f"(v - e{self.ion})"
            
        modfile.append(current_str)
        modfile.append("}\n")

        # Write DERIVATIVE section
        if len(self.state_variables) > 0:
            modfile.append("DERIVATIVE states {")
            for svar in self.state_variables:
                if (segmented) and (isinstance(svar, SegmentableStateVariable)):
                    modfile.append(self._compose_segmented_activation_function(svar))
                else:
                    modfile.append(f"\t{svar.name}inf = 1.0 / (1.0 + exp(-(v - {svar.vhalf}) / {svar.k}))")
                modfile.append(f"\t{svar.name}\' = ({svar.name}inf - {svar.name}) / {svar.tau}")
            modfile.append("}\n")

        # Save
        modfile = str.join("\n", modfile)

        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)

        with open(os.path.join(path_to_folder, f"{self.channel_name}.mod"), "w") as f:
            f.write(modfile)
    
    def _compose_segmented_activation_function(self, svar: SegmentableStateVariable):

        # Check if segmented
        if svar._segmented_v_threshold is None:
            raise ValueError(f"The state variable {self.name} has not been segmented yet. Use ACTSegmenter to perform segmentation.")
        
        af = []
        af.append("\n\t:Segmentation-start")

        # Main segment
        af.append(f"\t{svar.name}inf = 1.0 / (1.0 + exp(-(v - {svar._segmented_vhalf}) / {svar._segmented_k}))")

        # Linear segment
        af.append(f"\tif (v < {svar._segmented_v_threshold + 3}) " + "{")
        af.append(f"\t\t{svar.name}inf = {round(svar._segmented_linear_intercept, 5)} + v * {round(svar._segmented_linear_slope, 5)}")
        af.append("\t}")

        # 0 segment
        af.append(f"\tif (v < {svar._segmented_v_threshold}) " + "{")
        af.append(f"\t\t{svar.name}inf = 0")
        af.append("\t}")

        af.append("\t:Segmentation-end\n")

        return str.join("\n", af)
    
# -----
# ALLEN
# -----

def vtrap(x, y):
    if np.any(np.abs(x / y) < 1e-6):
        vtrap = y * (1 - x / y / 2)
    else:
        vtrap = x / (np.exp(x / y) - 1)
    return vtrap

def fit_boltzmann(v, p_v):
    def boltzmann(v, vhalf, k):
        return 1 / (1 + np.exp((vhalf - v) / k))
    params, _ = curve_fit(f = boltzmann, xdata = v, ydata = p_v)
    return params # vhalf, k
    
class AllenIh(ACTModfile):

     def __init__(self):
        
        self.channel_name = "Ih"
        self.ion = None
        self.gbar = 0.00001
        self.e = -45.0
        self.state_variables = state_variables