import numpy as np

from act.act_types import SimulationConfig

# ===================================================================================================================
# ===================================================================================================================
#                                                   LA A
# ===================================================================================================================
# ===================================================================================================================
LA_A_seg = {
    "cell": {
        "hoc_file": "../data/LA/A/seg_modfiles_modeldb/template.hoc",
        "modfiles_folder": "../data/LA/A/seg_modfiles_modeldb",
        "name": "Cell_A_seg",
        "passive_properties": {
            "v_rest": -71.486,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.1, 0.2, 0.3, 0.4],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        "lto_block_channels": [], # in the alterki paper, no channels were blocked for lto, but above config change worked
        "hto_block_channels": ["gbar_na3", "gbar_nap", "gsAHPbar_sAHP"],
        "params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {"channel": "gbar_nap", "high": 0.000284, "low": 0.0}, #"high": 0.000213, "low": 0.000071}, #"high": 0.000426, "low": 4.736e-05},
            {"channel": "gbar_im", "high": 0.0012, "low": 0.0}, #"high": 0.003000, "low": 0.001000}, #"high": 0.006, "low": 0.000666},
            {"channel": "gbar_na3", "high": 0.054, "low": 0.0},#{"channel": "gbar_na3", "high": 0.054, "low": 0.0}, #"high": 0.060000, "low": 0.000000}, #"high": 0.045000, "low": 0.015000}, #"high": 0.09, "low": 0.01},
            {"channel": "gbar_kdr", "high": 0.006, "low": 0.0},#{"channel": "gbar_kdr", "high": 0.003, "low": 0.0}, #"high": 0.002250, "low": 0.000750}, #"high": 0.0045, "low": 0.0005},
            {"channel": "gcabar_cadyn", "high": 0.0032, "low": 0.0}, #{"channel": "gcabar_cadyn", "high": 0.0016, "low": 0.0}, #"high": 0.000090, "low": 0.000030}, #"high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.0006, "low": 0.0}, #"high": 0.013500, "low": 0.004500}, #"high": 0.026996, "low": 0.0029996},
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        "target_cell": {
            "hoc_file": "../data/LA/A/orig_modfiles/template.hoc",
            "modfiles_folder": "../data/LA/A/orig_modfiles",
            "name": "Cell_A",
        },
        "target_cell_params": [
           {"channel": "gbar_nap"},
            {"channel": "gmbar_im"},
            {"channel": "gbar_na3"},
            {"channel": "gkdrbar_kdr"},
            {"channel": "gcabar_cadyn"},
            {"channel": "gsAHPbar_sAHP"},
        ],
        "target_passive_properties": {
            "v_rest": -70,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
        "target_cell_target_params": [0.0003, 0.002, 0.03, 0.03, 6e-5, 0.009],
        # ======================================================
        "target_V": None,  # Target voltages
        "target_params": [ # will not be used, except for analyzing error
            0.000142, #0.000142,
            0.0006, #0.002,
            0.027, #0.03,
            0.0015, #0.0015,
            0.00055, #0.00055, #6e-5, # 0.0008 -> Walt found that this was too low, typo from paper
            0.0003, #0.009,
        ],  # [2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 6,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        { # LTO
            "params": ["gbar_nap", "gbar_im"],
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "train_amplitude_frequency": True,
            "train_mean_potential": True,
            "use_lto_amps": True,
            "ramp_time": 1000, #ms to ramp up the amp input
            "ramp_splits": 20, # the amps should be steped up n times
            "use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            "h_tstop": 1500,  # (ms)
            "h_i_delay": 250,  # (ms)
            "h_i_dur": 1000,  # (ms)
        },
        { # Spiking rough estimate
            "params": ["gbar_na3", "gbar_kdr"],
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # Spiking Adaptation - allow variation the learned spiking parameters by 20%
            "params": ["gcabar_cadyn", "gsAHPbar_sAHP"],
            "learned_variability_params": ["gbar_na3", "gbar_kdr"],
            "learned_variability": 0.25,
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # HTO - na3 and nap are blocked
            "params": [],
            "learned_variability_params": ["gbar_im", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "learned_variability": 0.25,
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "nonsaturated_only": False,
            "train_amplitude_frequency": True,
            "train_mean_potential": True,
            "use_hto_amps": True,
            "ramp_time": 1000, #ms to ramp up the amp input
            "ramp_splits": 20, # the amps should be steped up n times
            "use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            "h_tstop": 1500,  # (ms)
            "h_i_delay": 250,  # (ms)
            "h_i_dur": 1000,  # (ms)
        }
    ],
    "output": {
        "folder": "output_LA_A_seg",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Segregated",
    },
    "run_mode": "segregated",  # "original", "segregated"
}

LA_A_orig = {
    "cell": {
        "hoc_file": "../data/LA/A/orig_modfiles/template.hoc",
        "modfiles_folder": "../data/LA/A/orig_modfiles",
        "name": "Cell_A",
         "passive_properties": {
            "v_rest": -70,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.1, 0.2, 0.3, 0.4],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        "lto_block_channels": [],
        "hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_nap", "high": 0.0009, "low": 0.0001},
            {"channel": "gmbar_im", "high": 0.006, "low": 0.00066667},
            {"channel": "gbar_na3", "high": 0.09, "low": 0.01},
            {"channel": "gkdrbar_kdr", "high": 0.09, "low": 0.01},
            {"channel": "gcabar_cadyn", "high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.027, "low": 0.003},
        ],
        "target_V_file": "./target_v.json",
        "target_V": None,  # Target voltages
        # "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "target_params": [0.0003, 0.002, 0.03, 0.03, 6e-5, 0.009],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "use_random_forest": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        {  # passive
            "params": ["glbar_leak", "ghdbar_hd"],
            "voltage": [-80, -67.5],
        },
        {  # lto
            "params": ["gbar_nap", "gmbar_im"],
            "voltage": [-67.5, -57.5],
        },
        {  # spking / adaptation
            "params": ["gbar_na3", "gkdrbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "voltage": [-57.5, 0],
        },
        {  # hto
            "params": ["gbar_nap", "gmbar_im"],
            "voltage": [-40, -30],
        },
    ],
    "output": {
        "folder": "output_LA_A_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}

# ===================================================================================================================
# ===================================================================================================================
#                                                   LA C
# ===================================================================================================================
# ===================================================================================================================

LA_C_seg = {
    "cell": {
        "hoc_file": "../data/LA/C/seg_modfiles/template.hoc",
        "modfiles_folder": "../data/LA/C/seg_modfiles",
        "name": "Cell_C",
        "passive_properties": {
            "v_rest": -71.486,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.1, 0.2, 0.3, 0.4],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        "lto_block_channels": [], # in the alterki paper, no channels were blocked for lto, but above config change worked
        "hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {"channel": "gbar_nap", "high": 0.000284, "low": 0.0}, #"high": 0.000213, "low": 0.000071}, #"high": 0.000426, "low": 4.736e-05},
            {"channel": "gbar_im", "high": 0.0012, "low": 0.0}, #"high": 0.003000, "low": 0.001000}, #"high": 0.006, "low": 0.000666},
            {"channel": "gbar_na3", "high": 0.054, "low": 0.0},#{"channel": "gbar_na3", "high": 0.054, "low": 0.0}, #"high": 0.060000, "low": 0.000000}, #"high": 0.045000, "low": 0.015000}, #"high": 0.09, "low": 0.01},
            {"channel": "gbar_kdr", "high": 0.003, "low": 0.0},#{"channel": "gbar_kdr", "high": 0.003, "low": 0.0}, #"high": 0.002250, "low": 0.000750}, #"high": 0.0045, "low": 0.0005},
            {"channel": "gcabar_cadyn", "high": 0.0011, "low": 0.0}, #{"channel": "gcabar_cadyn", "high": 0.0016, "low": 0.0}, #"high": 0.000090, "low": 0.000030}, #"high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.0001, "low": 0.0}, #"high": 0.013500, "low": 0.004500}, #"high": 0.026996, "low": 0.0029996},
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        "target_cell": {
            "hoc_file": "../data/LA/C/orig_modfiles/template.hoc",
            "modfiles_folder": "../data/LA/C/orig_modfiles",
            "name": "Cell_C",
        },
        "target_cell_params": [
           {"channel": "gbar_nap"},
            {"channel": "gmbar_im"},
            {"channel": "gbar_na3"},
            {"channel": "gkdrbar_kdr"},
            {"channel": "gcabar_cadyn"},
            {"channel": "gsAHPbar_sAHP"},
        ],
        "target_passive_properties": {
            "v_rest": -70,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
        "target_cell_target_params": [0.00014, 0.001, 0.03, 0.009, 7e-5, 0.00025],
        # ======================================================
        "target_V": None,  # Target voltages
        "target_params": [ # will not be used, except for analyzing error
            0.000142, #0.000142,
            0.0006, #0.002,
            0.027, #0.03,
            0.0015, #0.0015,
            0.00055, #0.00055, #6e-5,
            0.00005, #0.009,
        ],  # [2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 6,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        { # LTO
            "params": ["gbar_nap", "gbar_im"],
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "train_amplitude_frequency": True,
            "train_mean_potential": True,
            "use_lto_amps": True,
            "ramp_time": 1000, #ms to ramp up the amp input
            "ramp_splits": 20, # the amps should be steped up n times
            "use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            "h_tstop": 1500,  # (ms)
            "h_i_delay": 250,  # (ms)
            "h_i_dur": 1000,  # (ms)
        },
        { # Spiking rough estimate
            "params": ["gbar_na3", "gbar_kdr"],
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # Spiking Adaptation - allow variation the learned spiking parameters by 20%
            "params": ["gcabar_cadyn", "gsAHPbar_sAHP"],
            "learned_variability_params": ["gbar_na3", "gbar_kdr"],
            "learned_variability": 0.25,
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # HTO - na3 and nap are blocked
            "params": [],
            "learned_variability_params": ["gbar_im", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "learned_variability": 0.25,
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "nonsaturated_only": False,
            "train_amplitude_frequency": True,
            "train_mean_potential": True,
            "use_hto_amps": True,
            "ramp_time": 1000, #ms to ramp up the amp input
            "ramp_splits": 20, # the amps should be steped up n times
            "use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            "h_tstop": 1500,  # (ms)
            "h_i_delay": 250,  # (ms)
            "h_i_dur": 1000,  # (ms)
        }
    ],
    "output": {
        "folder": "output_LA_C_seg",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Segregated",
    },
    "run_mode": "segregated",  # "original", "segregated"
}

LA_C_orig = {
    "cell": {
        "hoc_file": "../data/LA/C/orig_modfiles/template.hoc",
        "modfiles_folder": "../data/LA/C/orig_modfiles",
        "name": "Cell_C",
         "passive_properties": {
            "v_rest": -70,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.1, 0.2, 0.3, 0.4],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        "lto_block_channels": [],
        "hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_nap", "high": 0.0009, "low": 0.0001},
            {"channel": "gmbar_im", "high": 0.006, "low": 0.00066667},
            {"channel": "gbar_na3", "high": 0.09, "low": 0.01},
            {"channel": "gkdrbar_kdr", "high": 0.09, "low": 0.01},
            {"channel": "gcabar_cadyn", "high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.027, "low": 0.003},
        ],
        "target_V_file": "./target_v.json",
        "target_V": None,  # Target voltages
        # "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "target_params": [0.00014, 0.001, 0.03, 0.009, 7e-5, 0.00025],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "use_random_forest": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        {  # passive
            "params": ["glbar_leak", "ghdbar_hd"],
            "voltage": [-80, -67.5],
        },
        {  # lto
            "params": ["gbar_nap", "gmbar_im"],
            "voltage": [-67.5, -57.5],
        },
        {  # spking / adaptation
            "params": ["gbar_na3", "gkdrbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "voltage": [-57.5, 0],
        },
        {  # hto
            "params": ["gbar_nap", "gmbar_im"],
            "voltage": [-40, -30],
        },
    ],
    "output": {
        "folder": "output_LA_C_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}

# ===================================================================================================================
# ===================================================================================================================
#                                                   Burster Izh
# ===================================================================================================================
# ===================================================================================================================

Burster_Izh_seg = {
    "cell": {
        "hoc_file": "../data/Burster/Izhikevich_p307_seg/template.hoc",
        "modfiles_folder": "../data/Burster/Izhikevich_p307_seg/",
        "name": "Burster_Izh",
        "passive_properties": {
            "v_rest": -75,
            "r_in": 260,
            "tau": 102,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -75.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.25, 0.5, 0.75, 1.0],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        "lto_block_channels": [], # in the alterki paper, no channels were blocked for lto, but above config change worked
        "hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_nap", "high": 0.0008, "low": 0.0}, #"high": 0.000213, "low": 0.000071}, #"high": 0.000426, "low": 4.736e-05},
            {"channel": "gmbar_im", "high": 0.0076, "low": 0.0}, #"high": 0.003000, "low": 0.001000}, #"high": 0.006, "low": 0.000666},
            {"channel": "gbar_na3", "high": 0.1, "low": 0.0},#{"channel": "gbar_na3", "high": 0.054, "low": 0.0}, #"high": 0.060000, "low": 0.000000}, #"high": 0.045000, "low": 0.015000}, #"high": 0.09, "low": 0.01},
            {"channel": "gkdrbar_kdr", "high": 0.06, "low": 0.0},#{"channel": "gbar_kdr", "high": 0.003, "low": 0.0}, #"high": 0.002250, "low": 0.000750}, #"high": 0.0045, "low": 0.0005},
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        "target_cell": {
            "hoc_file": "../data/Burster/Izhikevich_p307_orig/template.hoc",
            "modfiles_folder": "../data/Burster/Izhikevich_p307_orig/",
            "name": "Burster_Izh",
        },
        "target_cell_params": [
           {"channel": "gbar_nap"},
            {"channel": "gmbar_im"},
            {"channel": "gbar_na3"},
            {"channel": "gkdrbar_kdr"},
        ],
        "target_passive_properties": {
            "v_rest": -75,
            "r_in": 260,
            "tau": 102,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
        "target_cell_target_params": [0.0003, 0.0033, 0.03, 0.028],
        # ======================================================
        "target_V": None,  # Target voltages
        "target_params": [ # will not be used, except for analyzing error
            0.0004,
            0.0038,
            0.05,
            0.03,
        ],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 6,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        { # Spiking rough estimate
            "params": ["gbar_na3", "gkdrbar_kdr"],
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # Bursting dynamics
            "params": ["gbar_nap", "gmbar_im"],
            "learned_variability_params": ["gbar_na3", "gbar_kdr"],
            "learned_variability": 0.2,
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "fi_error", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "train_amplitude_frequency": False,
            "train_mean_potential": True,
            "use_lto_amps": False,
            #"ramp_time": 1000, #ms to ramp up the amp input
            #"ramp_splits": 20, # the amps should be steped up n times
            #"use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            #"h_tstop": 1500,  # (ms)
            #"h_i_delay": 250,  # (ms)
            #"h_i_dur": 1000,  # (ms)
        },
    ],
    "output": {
        "folder": "output_Burster_Izh_seg",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Segregated",
    },
    "run_mode": "segregated",  # "original", "segregated"
}

Burster_Izh_orig = {
    "cell": {
        "hoc_file": "../data/Burster/Izhikevich_p307_orig/template.hoc",
        "modfiles_folder": "../data/Burster/Izhikevich_p307_orig/",
        "name": "Burster_Izh",
         "passive_properties": {
            "v_rest": -75,
            "r_in": 260,
            "tau": 102,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
         },
    },
    "simulation_parameters": {
        "h_v_init": -75.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.25, 0.5, 0.75, 1.0],
        #"lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        #"hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": [],
        #"hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_nap", "high": 0.0006, "low": 0.0001},
            {"channel": "gmbar_im", "high": 0.0033, "low": 0.00066667},
            {"channel": "gbar_na3", "high": 0.06, "low": 0.01},
            {"channel": "gkdrbar_kdr", "high": 0.056, "low": 0.01},
        ],
        "target_V_file": "./target_v.json",
        "target_V": None,  # Target voltages
        "target_params": [0.0003, 0.0033, 0.03, 0.028],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "use_random_forest": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 10,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
    ],
    "output": {
        "folder": "output_Burster_Izh_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}

# ===================================================================================================================
# ===================================================================================================================
#                                                   Burster S3
# ===================================================================================================================
# ===================================================================================================================


Burster_S3_seg = {
    "cell": {
        "hoc_file": "../data/BursterS3/seg/template.hoc",
        "modfiles_folder": "../data/BursterS3/seg/seg_modfiles",
        "name": "Burster",
        "passive_properties": {
            "v_rest": -49,
            "r_in": 90,
            "tau": 43.175,
            "leak_conductance_variable": "gbar_leak",  # eg: g_leak
            "leak_reversal_variable": "eleak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -49.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.5, 1.0, 1.5, 2],
        #"lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        #"lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        #"hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #"lto_block_channels": [], # in the alterki paper, no channels were blocked for lto, but above config change worked
        #"hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_na", "high": 0.26, "low": 0.0},
            {"channel": "gbar_kdr", "high": 0.2, "low": 0.0},
            {"channel": "gbar_cas", "high": 0.02, "low": 0.0},
            {"channel": "gbar_cat", "high": 0.01, "low": 0.0},
            {"channel": "gbar_ka", "high": 0.34, "low": 0.0},
            {"channel": "gbar_kca", "high": 0.04, "low": 0.0},
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        "target_cell": {
            "hoc_file": "../data/BursterS3/orig/template.hoc",
            "modfiles_folder": "../data/BursterS3/orig/orig_modfiles",
            "name": "Burster",
        },
        "target_cell_params": [
           {"channel": "gbar_na"},
            {"channel": "gbar_kdr"},
            {"channel": "gbar_cas"},
            {"channel": "gbar_cat"},
            {"channel": "gbar_ka"},
            {"channel": "gbar_kca"},
        ],
        "target_passive_properties": {
            "v_rest": -49,
            "r_in": 90,
            "tau": 43.175,
            "leak_conductance_variable": "gbar_leak",  # eg: g_leak
            "leak_reversal_variable": "eleak",  # eg: e_leak
         },
        "target_cell_target_params": [0.13, 0.1, 0.01, 0.005, 0.17, 0.02],
        # ======================================================
        "target_V": None,  # Target voltages
        "target_params": [0.13, 0.1, 0.01, 0.005, 0.17, 0.02],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 10,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        { # Spiking rough estimate
            "params": ["gbar_na", "gbar_kdr"],
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # Bursting dynamics
            "params": ["gbar_cas", "gbar_cat", "gbar_ka", "gbar_kca"],
            "learned_variability_params": ["gbar_na", "gbar_kdr"],
            "learned_variability": 0.2,
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "fi_error", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "train_amplitude_frequency": False,
            "train_mean_potential": True,
            "use_lto_amps": False,
            #"ramp_time": 1000, #ms to ramp up the amp input
            #"ramp_splits": 20, # the amps should be steped up n times
            #"use_spike_summary_stats": False, # don't use spike summary stats for training
            "arima_order": [4, 0, 4], # custom arima settings
            #"h_tstop": 1500,  # (ms)
            #"h_i_delay": 250,  # (ms)
            #"h_i_dur": 1000,  # (ms)
        },
    ],
    "output": {
        "folder": "output_Burster_S3_seg",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Segregated",
    },
    "run_mode": "segregated",  # "original", "segregated"
}

Burster_S3_orig = {
    "cell": {
        "hoc_file": "../data/BursterS3/orig/template.hoc",
        "modfiles_folder": "../data/BursterS3/orig/orig_modfiles",
        "name": "Burster",
         "passive_properties": {
            "v_rest": -49,
            "r_in": 90,
            "tau": 43.175,
            "leak_conductance_variable": "gbar_leak",  # eg: g_leak
            "leak_reversal_variable": "eleak",  # eg: e_leak
         },
    },
    "simulation_parameters": {
        "h_v_init": -49.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": [0.0, 0.5, 1.0, 1.5, 2],
        #"lto_amps": [0.060, 0.065, 0.070, 0.075, 0.08],
        #"hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        #"lto_block_channels": [],
        #"hto_block_channels": ["gbar_na3", "gbar_nap"],
        "params": [
            {"channel": "gbar_na", "high": 0.26, "low": 0.0},
            {"channel": "gbar_kdr", "high": 0.2, "low": 0.0},
            {"channel": "gbar_cas", "high": 0.02, "low": 0.0},
            {"channel": "gbar_cat", "high": 0.01, "low": 0.0},
            {"channel": "gbar_ka", "high": 0.34, "low": 0.0},
            {"channel": "gbar_kca", "high": 0.04, "low": 0.0},
        ],
        "target_V_file": "./target_v.json",
        "target_V": None,  # Target voltages
        "target_params": [0.13, 0.1, 0.01, 0.005, 0.17, 0.02],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "use_random_forest": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -20,  # (mV)
        "arima_order": [4, 0, 4],
        "num_first_spikes": 10,
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
    ],
    "output": {
        "folder": "output_Burster_S3_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}


# ===================================================================================================================
# ===================================================================================================================
#                                                 SELECTED_CONFIG
# ===================================================================================================================
# ===================================================================================================================
selected_config = Burster_S3_seg
