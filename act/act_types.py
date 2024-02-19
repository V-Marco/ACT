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
    h_celsius: float


class OptimizationParam(TypedDict):
    param: str
    low: float
    high: float


class ParametricDistribution(TypedDict):
    n_slices: int  # slice each variable min to max, into n equal slices
    simulations_per_amp: int  # each amp will be split equally


class OptimizationParameters(TypedDict):
    skip_match_voltage: bool
    amps: List[float]
    lto_amps: List[float]
    hto_amps: List[float]
    lto_block_channels: List[str]
    hto_block_channels: List[str]
    params: List[OptimizationParam]
    target_cell: Cell
    target_cell_params: List[OptimizationParam]
    target_cell_target_params: List[List[float]]
    target_V_file: str  # location of voltage traces, stored as a json {"traces":[[],]}
    target_V: List[List[float]]  # Target voltage
    target_params: List[List[float]]
    num_repeats: int
    num_amps_to_match: int
    num_epochs: int
    random_seed: int
    parametric_distribution: ParametricDistribution
    decimate_factor: int  # reduce voltage traces after simulation
    use_random_forest: bool


class SummaryFeatures:
    spike_threshold: int  # (mV)
    arima_order: list
    num_first_spikes: int
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
    model_class: str  # what kind of neural network should we use (optional) in act.models, just specify class name as string
    learning_rate: float
    weight_decay: float
    selection_metric: str  # fi_error or mse should be enum
    num_epochs: int  # change the number of epochs that we can train for overrides global num_epochs
    train_spiking_only: bool  # only train on spiking, true by default
    nonsaturated_only: bool  # only train on nonsaturating traces, true by default
    train_amplitude_frequency: bool  # train using freqency/amplitude - only useful for lto/hto, finds peaks and counts, with mean value for peaks
    train_mean_potential: bool # get the mean of each and use in training data
    adjustment_percent: float  # a percentage that future seg modules will be allowed to modify the suggested param
    adjustment_n_slices: int  # n_splits for the adjustment, will use default parametric distribution n_splits if this doesn't exist
    use_lto_amps: bool  # use lto amps instead of amps
    use_hto_amps: bool
    use_spike_summary_stats: bool  # if set to false, then don't train on spike interval, spike times, etc...
    arima_order: List[int]  # use a custom arima order for this segregation index
    learned_variability: float  # allow the previously learned  parameters to vary by the specified percentage
    learned_variability_params: List[
        str
    ]  # select the parameters you want to vary, otherwise it's the last segregation module
    n_splits: int  # custom number of splits
    ramp_splits: int  # how many current injections should be made and split
    ramp_time: float  # how long should a ramp be before you deliver current inj
    h_tstop: int  # (ms) Override the simulation config params
    h_i_delay: int  # (ms) ""
    h_i_dur: int  # (ms) ""


class Output(TypedDict):
    folder: str
    auto_structure: bool
    produce_plots: bool
    target_label: str
    simulated_label: str


class SimulationConfig(TypedDict):
    cell: Cell
    simulation_parameters: SimulationParameters
    optimization_parameters: OptimizationParameters
    summary_features: SummaryFeatures
    segregation: List[SegregationModule]
    output: Output

    run_mode: str  # "original", "segregated"
