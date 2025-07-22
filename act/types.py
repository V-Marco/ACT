from dataclasses import dataclass

@dataclass
class SettablePassiveProperties:
    """
    Passive properties that can be directly set for a cell model.

    Attributes
    ----------
    Cm: float, default = None
        Membrane capacitance (uf / cm2).

    g_bar_leak: float, default = None
        Maximum conductance of the leak channel (S / cm2).
    
    e_rev_leak: float, default = None
        Reversal potential of the leak channel (mV).
    
    g_bar_h: float, default = None
        Maximum conductance of the H channel (S / cm2).
    """
    Cm: float = None
    g_bar_leak: float = None
    e_rev_leak: float = None
    g_bar_h: float = None

@dataclass
class GettablePassiveProperties:
    """
    Passive properties that can be retrieved from traces generated from a cell model.

    Attributes
    ----------
    R_in: float, default = None
        Input resistance (MOhm).

    tau1: float, default = None
        Lower bound on the membrane time constant (ms).
    
    tau2: float, default = None
        Upper bound on the membrance time constant (ms).
    
    sag_ratio: float, default = None
        Computed as (v_final - v_min) / (v_rest - v_min).
    
    V_rest: float, default = None
        Resting potential (mV).
    """
    R_in: float = None
    tau1: float = None
    tau2: float = None
    sag_ratio: float = None
    V_rest: float = None

@dataclass
class ConstantCurrentInjection:
    """
    Set parameters for a constant (pulse) current injection.

    Attributes
    ----------
    amp: float, default = 0.1
        Current injection amplitude (nA).
    
    dur: float, default = 400
        Current injection duration (ms).
    
    delay: float, default = 50
        Current injection delay (ms).
    """
    amp: float = 0.1                          # (nA)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)

@dataclass
class RampCurrentInjection:
    """
    Set parameters for a ramp (step) current injection.

    Attributes
    ----------
    amp_start: float, default = 0
        Current injection initial amplitude (nA).
    
    amp_incr: float, default = 0.1
        Current injection amplitude increase per step (nA).
    
    num_steps: int, default = 10
        Number of steps (periods of constant current injection).

    final_step_add_time: int, default = 0
        Additional time to extend the final current injection step for (ms).

    dur: float, default = 400
        Total time length of current injection (ms).
    
    delay: float, default = 50
        Current injection delay (ms).
    """
    amp_start: float = 0                      # (nA)
    amp_incr: float = 0.1                     # (nA)
    num_steps: int = 10
    final_step_add_time: int = 0              # (ms)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)

@dataclass
class GaussianCurrentInjection:
    """
    Set parameters for a Gaussian current injection.

    Attributes
    ----------
    amp_mean: float, default = 0.1
        Mean of current injection amplitude (nA).
    
    amp_std: float, default = 0.02
        Standard deviation of current injection amplitude (nA).

    dur: float, default = 400
        Current injection duration (ms).
    
    delay: float, default = 50
        Current injection delay (ms).
    """
    amp_mean: float = 0.1                     # (nA)
    amp_std: float = 0.02                     # (nA)
    dur: float = 400                          # (ms)
    delay: float = 50                         # (ms)
    
@dataclass  
class SimulationParameters:
    """
    Set parameters for a simulation with the ACTSimulator.

    Attributes
    ----------
    sim_name: str, default = ""
        Simulation name. Saves to a folder output_folder/sim_name. Creates this folder if it does not exist.
    
    sim_idx: int, default = 0
        Simulation index. Simulation results will be saved as output_folder/sim_name/sim_idx.npy.
    
    h_v_init: float, default = -50
        Initial voltage (mV).
    
    h_tstop: int, default = 500
        Simulation time (ms).
    
    h_dt: float, default = 0.1
        Simulation step (ms).

    h_celsius: float, default = 37
        Simulation temperature (deg C).

    CI: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        List of current injections to be applied during the simulation. Supports overlapping injections.
    
    random_seed: int, default = 42
        Global random seed used throughout the simulation.
    
    verbose: bool, default = False
        If True, additional information is printed (e.g., soma and total area).
    """
    sim_name: str = ""
    sim_idx: int = 0 
    h_v_init: float = -50                     # (mV)
    h_tstop: int = 500                        # (ms)
    h_dt: float = 0.1                         # (ms)
    h_celsius: float = 37                     # (deg C)
    CI: list = None
    random_seed: int = 42
    verbose: bool = False
    _path: str = None

# ----------
# Optimization
# ----------

@dataclass
class ConductanceOptions:
    """
    Set a conductance for optimization. 
    During training, the conductance specified by `variable_name` is uniformly sampled from the (`low`, `high`) range `n_slices` times.
    Alternatively, if `bounds_variation` is specified, the sampling space is defined by the `ACTCellModel.prediction` ± `bounds_variation`.

    Attributes
    ----------
    variable_name: str, default = None
        Conductance name matching that in the ACTCellModel.
    
    blocked: bool, default = False
        If True, the channel is blocked during the simulation (its maximum conductance is set to 0).
    
    low: float, default = None
        Lower bound of the conductance sampling space. Ignored if `bounds_variation` is specified.

    high: float, default = None
        Upper bound of the conductance sampling space. Ignored if `bounds_variation` is specified.

    bounds_variation: float, default = None
        If specified, the conductance sampling space is defined by the (`ACTCellModel.prediction` - `bounds_variation`, `ACTCellModel.prediction` + `bounds_variation`).
    
    n_slices: int, default = 1
        Number of conductance samples.
    """
    variable_name: str = None
    blocked: bool = False
    low: float = None
    high: float = None
    bounds_variation: float = None
    n_slices: int = 1
    
@dataclass
class FilterParameters:
    """
    Set criteria for filtering generated voltage traces.

    Attributes
    ----------
    filtered_out_features: list[str], default = None
        Which features to filter out. May contain the following:
        - "saturated" (removes saturated traces, i.e., traces with sustained elevated voltage, see `data_processing.remove_saturated_traces`),
        - "no_spikes" (removes traces without spikes, see `data_processing.get_traces_with_spikes`).
    
    window_of_inspection: tuple(int, int), default = None
        Time window to consider for filtering (ms).
    
    saturation_threshold: float, default = -50
        If "saturated" is in `filtered_out_features`, defines the voltage elevation threshold (mV).
    """
    filtered_out_features: list = None
    window_of_inspection: tuple = None
    saturation_threshold: float = -50

@dataclass
class OptimizationParameters:
    """
    Set optimization parameters for an `ACTModule`. Optimization space is defined as a multidimensional grid specified by `conductance_options` and `CI_options`.

    Attributes
    ----------
    n_cpus: int, default = None
        Number of CPU cores to use during optimization. If `None`, all available cores are used.
    
    conductance_options: list[ConductanceOptions], default = None
        Conductances to optimize in this module.
    
    CI_options: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        Current injections to consider for optimization. Overwrites `SimulationParameters.CI`.

    random_state: int, default = 42
        Random state for the `RandomForestRegressor` model.

    n_estimators: int, default = 1000
        Number of estimators (decision trees) for the `RandomForestRegressor` model.

    max_depth: int, default = None
        Maximum tree depth for the `RandomForestRegressor` model.

    train_features: list[str], default = None
        If specified, only use these summary features for training.

    filter_parameters: FilterParameters, default = None
        Parameters to filter out sampled voltage traces.
    
    spike_threshold: float, default = 0
        Threshold for spike detection.

    max_n_spikes: int, default = 20
        Maximum number of spikes to consider for summary features computation.
    """
    # Set hardware
    n_cpus: int = None

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