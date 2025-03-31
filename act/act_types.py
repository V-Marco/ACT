from typing import List, Union
from dataclasses import dataclass, field
from dataclasses import dataclass
from act.optimizer import RandomForestOptimizer

# A collection of dataclass types that should be used to define the user supplied configuration.

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
    lto_hto: float = 0                       # "lto", "hto"
    
    
@dataclass  
class SimulationParameters:
    sim_name: str = ""
    sim_idx: int = 0
    set_g_to: List[float] = None
    h_v_init: float = -50                     # (mV)
    h_tstop: int = 500                        # (ms)
    h_dt: float = 0.1                         # (ms)
    h_celsius: float = 37                     # (deg C)
    CI: List[Union[ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection]] = None
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
    filtered_out_features: List[str] = None
    window_of_inspection: tuple = None
    saturation_threshold: float = -50


@dataclass
class OptimizationParameters:
    conductance_options: List[ConductanceOptions] = None
    amps: List[float] = None
    random_state: int = 42
    n_estimators: int = 1000
    max_depth: int = None
    eval_n_repeats: int = 3
    sample_rate_decimate_factor: int = None
    train_features: List[str] = None
    filter_parameters: FilterParameters = None
    spike_threshold: float = 0
    first_n_spikes: int = 20
    prediction_eval_method: str = 'fi_curve'
    evaluate_random_forest: bool = False
    rf_model: RandomForestOptimizer = None
    previous_modules: List[str] = None
    save_file: str = None