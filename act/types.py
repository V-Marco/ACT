from dataclasses import dataclass

@dataclass
class SettablePassiveProperties:
    Cm: float = None
    g_bar_leak: float = None
    e_rev_leak: float = None
    g_bar_h: float = None

@dataclass
class GettablePassiveProperties:
    R_in: float = None
    tau1: float = None
    tau2: float = None
    sag_ratio: float = None
    V_rest: float = None

@dataclass
class ConstantCurrentInjection:
    amp: float = 0.1                          # (nA)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)
    lto_hto: float = 0                        # "lto", "hto"

@dataclass
class RampCurrentInjection:
    amp_start: float = 0                      # (nA)
    amp_incr: float = 0.1                     # (nA)
    num_steps: int = 10
    step_time: float = 20                     # (ms)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)
    lto_hto: float = 0                       # "lto", "hto"

@dataclass
class GaussianCurrentInjection:
    amp_mean: float = 0.1                     # (nA)
    amp_std: float = 0.02                     # (nA)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)
    lto_hto: float = 0                        # "lto", "hto"
    
@dataclass  
class SimulationParameters:
    sim_name: str = ""
    sim_idx: int = 0 
    h_v_init: float = -50                     # (mV)
    h_tstop: int = 500                        # (ms)
    h_dt: float = 0.1                         # (ms)
    h_celsius: float = 37                     # (deg C)
    CI: list = None
    random_seed: int = 42
    _path: str = None

@dataclass
class ConductanceOptions:
    variable_name: str = None
    blocked: bool = False
    low: float = None
    high: float = None
    prediction: float = None
    bounds_variation: float = None
    n_slices: int = 1
    
@dataclass
class FilterParameters:
    filtered_out_features: list = None
    window_of_inspection: tuple = None
    saturation_threshold: float = -50

@dataclass
class OptimizationParameters:
    # Set the search space
    conductance_options: list = None
    CI_options: list = None

    # Set the RF model
    random_state: int = 42
    n_estimators: int = 1000
    max_depth: int = None

    # Set the features
    train_features: list = None
    filter_parameters: FilterParameters = None
    spike_threshold: float = 0
    max_n_spikes: int = 20

    # Set segregation #TODO
    previous_modules: list = None