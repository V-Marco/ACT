import numpy as np

class PospischilsPY:
    # Cell
    cell_hoc_file = "../data/Pospischil/sPY/template.hoc"
    cell_name_in_hoc_file = "sPY"

    # Simulation parameters
    h_v_init = -67.0 # (mV)
    h_tstop = 1500 # (ms)
    h_i_delay = 100 # (ms)
    h_i_dur = 1200 # (ms)
    h_dt = 0.025

    # Optimization parameters
    amps = list(np.arange(-2, 10, 0.1))
    params = ["g_pas", "gnabar_hh2", "gkbar_hh2", "gkbar_im"]
    lows = [0.000001, 0.01, 0.001, 0.00005]
    highs = [0.000015, 0.09, 0.009, 0.00020]

    # Summary features
    spike_threshold = 20 # (mV) 

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes = 1
    mc_interspike_time = 200 # (ms)
    mc_min_v = 1 # (mV)
    mc_mean_v = 2 # (mV)
    mc_max_v = 1 # (mV)

    # Segregation
    segr_param_inds = [[0], [3], [1, 2]]
    segr_voltage_bounds = [[-100, -75], [-75, -65], [-65, 100]]
    segr_time_bounds = [[None], [None], [None]]

    # Target voltage
    target_V = None
    target_params = [1.4998139e-05, 0.08155758, 0.0056799036, 0.0002]

    # Runtime
    run_mode = "original" # "original", "segregated"
    modfiles_mode = "segregated" # Used only for the output folder name
    modfiles_folder = "../data/Pospischil/sPY/seg_modfiles"
    num_repeats = 3
    num_amps_to_match = 30
    num_epochs = 5000

    # Output
    output_folder = "output_Pospischil_sPY"
    produce_plots = True

class PospischilsPYr:
    # Cell
    cell_hoc_file = "../data/Pospischil/sPYr/template.hoc"
    cell_name_in_hoc_file = "sPYr"

    # Simulation parameters
    h_v_init = -67.0 # (mV)
    h_tstop = 1500 # (ms)
    h_i_delay = 100 # (ms)
    h_i_dur = 1200 # (ms)
    h_dt = 0.025

    # Optimization parameters
    amps = list(np.arange(-2, 10, 0.1))
    params = ["g_pas", "gnabar_hh2", "gkbar_hh2", "gkbar_im", "gcabar_it"]
    lows = [0.000001, 0.01, 0.001, 0.00005, 0.0001]
    highs = [0.000015, 0.09, 0.009, 0.00020, 0.01]

    # Summary features
    spike_threshold = 20 # (mV) 

    # Target-sim match conditions (max abs diff between sim and target)
    mc_num_spikes = 1
    mc_interspike_time = 200 # (ms)
    mc_min_v = 1 # (mV)
    mc_mean_v = 2 # (mV)
    mc_max_v = 1 # (mV)

    # Segregation
    segr_param_inds = [[0], [3], [1, 2]]
    segr_voltage_bounds = [[-100, -75], [-75, -65], [-65, 100]]
    segr_time_bounds = [[None], [None], [None]]

    # Target voltage
    target_V = None
    target_params = [1.4998139e-05, 0.08155758, 0.0056799036, 0.0002, 0.001]

    # Runtime
    run_mode = "original" # "original", "segregated"
    modfiles_mode = "segregated" # Used only for the output folder name
    modfiles_folder = "../data/Pospischil/sPYr/seg_modfiles"
    num_repeats = 3
    num_amps_to_match = 30
    num_epochs = 5000

    # Output
    output_folder = "output_Pospischil_sPYr"
    produce_plots = True