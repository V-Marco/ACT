"""
These types should be used to define the user supplied simulation config/config.

"""

from typing import List, TypedDict


class PassiveProperties(TypedDict):
    v_rest: float
    r_in: float
    tau: float
    leak_conductance_variable: str  # eg: g_leak
    leak_reversal_variable: str  # eg: e_leak


class Cell(TypedDict):
    hoc_file: str
    modfiles_folder: str
    name: str
    passive_properties: PassiveProperties  # optional if you want an analytical passive approach


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
    target_V: List[List[float]]  # Target voltage
    target_params: List[List[float]]
    num_repeats: int
    num_amps_to_match: int
    num_epochs: int


class SummaryFeatures:
    spike_threshold: int  # (mV)

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes: int
    mc_interspike_time: int  # (ms)
    mc_min_v: int  # (mV)
    mc_mean_v: int  # (mV)
    mc_max_v: int  # (mV)


class SegregationModule(TypedDict):
    params: List[str]
    voltage: List[int]
    time: List[int]


class Output(TypedDict):
    folder: str
    produce_plots: bool


class SimulationConfig(TypedDict):
    cell: Cell
    simulation_parameters: SimulationParameters
    optimization_parameters: OptimizationParameters
    summary_features: SummaryFeatures
    segregation: List[SegregationModule]
    output: Output

    run_mode: str  # "original", "segregated"
