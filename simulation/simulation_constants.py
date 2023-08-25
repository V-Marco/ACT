import numpy as np

from act.act_types import SimulationConstants


class PospischilsPY:
    # Cell
    cell_hoc_file = "../data/Pospischil/sPY/template.hoc"
    cell_name = "sPY"

    # Simulation parameters
    h_v_init = -67.0  # (mV)
    h_tstop = 2000  # (ms)
    h_i_delay = 500  # (ms)
    h_i_dur = 1500  # (ms)
    h_dt = 0.025

    # Optimization parameters
    amps = list(np.arange(-2, 10, 0.1))
    params = ["g_pas", "gnabar_hh2", "gkbar_hh2", "gkbar_im"]
    lows = [5.0e-05, 2.5e-02, 2.5e-03, 3.5e-05]
    highs = [0.0002, 0.1, 0.01, 0.00014]

    # Summary features
    spike_threshold = 20  # (mV)

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes = 1
    mc_interspike_time = 200  # (ms)
    mc_min_v = 1  # (mV)
    mc_mean_v = 2  # (mV)
    mc_max_v = 1  # (mV)

    # Segregation
    segr_param_inds = [[0], [1, 2, 3]]
    segr_voltage_bounds = [[-100, -65], [-65, 100]]
    segr_time_bounds = [[0, 500], [0, 2000]]

    # Target voltage
    target_V = None
    target_params = [0.0001, 0.05, 0.005, 7e-5]

    # Runtime
    run_mode = "segregated"  # "original", "segregated"
    modfiles_mode = "segregated"  # Used only for the output folder name
    modfiles_folder = "../data/Pospischil/sPY/seg_modfiles"
    num_repeats = 3
    num_amps_to_match = 12
    num_epochs = 5000

    # Output
    output_folder = "output_Pospischil_sPY"
    produce_plots = True


class PospischilsPYr:
    # Cell
    cell_hoc_file = "../data/Pospischil/sPYr/template.hoc"
    cell_name = "sPYr"

    # Simulation parameters
    h_v_init = -67.0  # (mV)
    h_tstop = 2000  # (ms)
    h_i_delay = 500  # (ms)
    h_i_dur = 1500  # (ms)
    h_dt = 0.025

    # Optimization parameters
    amps = list(np.arange(-2, 10, 0.1))
    params = ["g_pas", "gnabar_hh2", "gkbar_hh2", "gkbar_im", "gcabar_it"]
    lows = [5.0e-06, 2.5e-02, 2.5e-03, 1.5e-05, 5.0e-04]
    highs = [2.0e-05, 1.0e-01, 1.0e-02, 6.0e-05, 2.0e-03]

    # Summary features
    spike_threshold = 20  # (mV)

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes = 1
    mc_interspike_time = 200  # (ms)
    mc_min_v = 1  # (mV)
    mc_mean_v = 2  # (mV)
    mc_max_v = 1  # (mV)

    # Segregation
    segr_param_inds = [[0], [1, 2, 3, 4]]
    segr_voltage_bounds = [[-100, -65], [-65, 100]]
    segr_time_bounds = [[0, 500], [0, 2000]]

    # Target voltage
    target_V = None
    target_params = [1e-5, 0.05, 0.005, 3e-5, 0.001]

    # Runtime
    run_mode = "segregated"  # "original", "segregated"
    modfiles_mode = "segregated"  # Used only for the output folder name
    modfiles_folder = "../data/Pospischil/sPYr/seg_modfiles"
    num_repeats = 3
    num_amps_to_match = 12
    num_epochs = 5000

    # Output
    output_folder = "output_Pospischil_sPYr"
    produce_plots = True


pospischilsPYr: SimulationConstants = {
    "cell": {
        "cell_hoc_file": "../data/Pospischil/sPYr/template.hoc",
        "cell_name": "sPYr",
    },
    "simulation_parameters": {
        "h_v_init": -67.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.025,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-2, 10, 0.1)),
        "params": [
            {"channel": "g_pas", "low": 5.0e-06, "high": 2.0e-05},
            {"channel": "gnabar_hh2", "low": 2.5e-02, "high": 1.0e-01},
            {"channel": "gkbar_hh2", "low": 2.5e-03, "high": 1.0e-02},
            {"channel": "gkbar_im", "low": 1.5e-05, "high": 6.0e-05},
            {"channel": "gcabar_it", "low": 5.0e-04, "high": 2.0e-03},
        ],
    },
    "summary_features": {
        "spike_threshold": 20,  # (mV)
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": {
        "segr_param_inds": [[0], [1, 2, 3, 4]],
        "segr_voltage_bounds": [[-100, -65], [-65, 100]],
        "segr_time_bounds": [[0, 500], [0, 2000]],
        # Target voltage
        "target_V": None,
        "target_params": [1e-5, 0.05, 0.005, 3e-5, 0.001],
    },
    "output": {"output_folder": "output_Pospischil_sPYr", "produce_plots": True},
    "run_mode": "segregated",  # "original", "segregated"
    "modfiles_mode": "segregated",  # Used only for the output folder name
    "modfiles_folder": "../data/Pospischil/sPYr/seg_modfiles",
    "num_repeats": 3,
    "num_amps_to_match": 12,
    "num_epochs": 5000,
}
