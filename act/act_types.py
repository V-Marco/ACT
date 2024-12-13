
# These types should be used to define the user supplied simulation config/config.

from typing import List, Tuple
from dataclasses import dataclass
from act.optimizer import RandomForestOptimizer

@dataclass
class PassiveProperties:
    V_rest: float
    R_in: float
    tau: float
    Cm: float
    g_bar_leak: float
    cell_area: float
    leak_conductance_variable: str  # eg: glbar_leak
    leak_reversal_variable: str  # eg: el_leak


@dataclass
class CurrentInjection:
    type: str = "constant"
    amp: float = 0.1 # (nA)
    dur: float = 400 # (ms)
    delay: float = 50 # (ms)
    
    
@dataclass  
class SimulationParameters:
    sim_name: str = ""
    sim_idx: int = 0
    set_g_to: List[Tuple[float,float]] = None
    h_v_init: float = -50 # (mV)
    h_tstop: int = 500  # (ms)
    h_dt: float = 0.1 # (ms)
    h_celsius: float = None # (deg C)
    CI: List[CurrentInjection] = None
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