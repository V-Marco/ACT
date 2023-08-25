"""
These types should be used to define the user supplied simulation config/constants.

"""

from typing import List, TypedDict


class Cell(TypedDict):
    cell_hoc_file: str
    cell_name: str


class SimulationParameters(TypedDict):
    h_v_init: float
    h_tstop: int  # (ms)
    h_i_delay: int  # (ms)
    h_i_dur: int  # (ms)
    h_dt: float


class OptimizationParam(TypedDict):
    param: str
    low: float
    high: float


class OptimizationParameters(TypedDict):
    amps: List[float]
    params: List[OptimizationParam]


class SummaryFeatures:
    spike_threshold: int  # (mV)

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes: int
    mc_interspike_time: int  # (ms)
    mc_min_v: int  # (mV)
    mc_mean_v: int  # (mV)
    mc_max_v: int  # (mV)


class Segregation(TypedDict):
    segr_param_inds: List[List[int]]
    segr_voltage_bounds: List[List[int]]
    segr_time_bounds: List[List[int]]

    # Target voltage
    target_V: int
    target_params: List[float]


class Output(TypedDict):
    output_folder: str
    produce_plots: bool


class SimulationConstants(TypedDict):
    cell: Cell
    simulation_parameters: SimulationParameters
    optimization_parameters: OptimizationParameters
    summary_features: SummaryFeatures
    segregation: Segregation
    output: Output

    run_mode: str  # "original", "segregated"
    modfiles_mode: str  # Used only for the output folder name
    modfiles_folder: str
    num_repeats: int
    num_amps_to_match: int
    num_epochs: int
