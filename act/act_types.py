"""
These types should be used to define the user supplied simulation config/config.

"""

from typing import List, TypedDict
from dataclasses import dataclass
from act.optimizer import RandomForestOptimizer

@dataclass
class PassiveProperties:
    cell_area: float = None
    V_rest: float = None
    R_in: float = None
    tau: float = None
    Cm: float = None
    g_leak: str = None
    g_bar_leak: float = None
    leak_conductance_variable: str = None # eg: glbar_leak
    leak_reversal_variable: str = None # eg: el_leak


@dataclass
class CurrentInjection:
    type: str = "constant"
    amps: List[float] = None # (nA)
    dur: float = 400 # (ms)
    delay: float = 50 # (ms)
    
@dataclass  
class SimulationParameters:
    sim_name: str = None
    sim_idx: int = None
    set_g_to: List[float] = None
    h_v_init: float = -50 # (mV)
    h_tstop: int = 500  # (ms)
    h_dt: float = 0.1 # (ms)
    h_celsius: float = None # (deg C)
    CI: CurrentInjection = None
    _path: str = None

@dataclass
class ConductanceOptions:
    variable_name: str
    blocked: bool
    low: float
    high: float
    prediction: float
    bounds_variation: float
    n_slices: int


@dataclass
class OptimizationParameters:
    conductance_options: List[ConductanceOptions] = None
    random_state: int = None
    n_estimators: int = None
    max_depth: int = None
    eval_n_repeats: int = None
    sample_rate_decimate_factor: int = None
    train_features: List[str] = None
    spike_threshold: float = None
    filtered_out_features: List[str] = None
    window_of_inspection: tuple = None
    saturation_threshold: float = None
    first_n_spikes: int = None
    prediction_eval_method: str = None
    rf_model: RandomForestOptimizer = None
    previous_modules: List[str] = None
    save_file: str = None


# class ParametricDistribution(TypedDict):
#     n_slices: int  # slice each variable min to max, into n equal slices
#     simulations_per_amp: int  # each amp will be split equally

# class SummaryFeatures:
#     spike_threshold: int  # (mV)
#     arima_order: list
#     num_first_spikes: int
#     # Target-sim match conditions (max abs diff between sim and target)
#     mc_num_spikes: int
#     mc_interspike_time: int  # (ms)
#     mc_min_v: int  # (mV)
#     mc_mean_v: int  # (mV)
#     mc_max_v: int  # (mV)