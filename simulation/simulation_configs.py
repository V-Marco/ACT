import numpy as np

from act.act_types import SimulationConfig

pospischilsPY: SimulationConfig = {
    "cell": {
        "hoc_file": "../data/Pospischil/sPy/template.hoc",
        "modfiles_folder": "../data/Pospischil/sPy/seg_modfiles",
        "name": "sPY",
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
        ],
        "target_V": None,  # Target voltages
        "target_params": [0.0001, 0.05, 0.005, 7e-5],
        "num_repeats": 3,
        "num_amps_to_match": 12,
        "num_epochs": 5000,
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
    "segregation": [
        {
            "params": ["g_pas"],
            "voltage": [-100, -65],
            "time": [0, 500],
        },
        {
            "params": ["gnabar_hh2", "gkbar_hh2", "gkbar_im"],
            "voltage": [-65, 100],
            "time": [0, 2000],
        },
    ],
    "output": {"folder": "output_Pospischil_sPY", "produce_plots": True},
    "run_mode": "original",  # "original", "segregated"
}


pospischilsPYr_orig: SimulationConfig = {
    "cell": {
        "hoc_file": "../data/Pospischil/sPyr/template.hoc",
        "modfiles_folder": "../data/Pospischil/sPyr/orig_modfiles",
        "name": "sPYr",
        "passive_properties": {
            "v_rest": -80,
            "r_in": 393.45,
            "tau": 117.425,
            "leak_conductance_variable": "g_pas",  # eg: g_leak
            "leak_reversal_variable": "e_pas",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -80.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-0.1, 4.1, 0.5)),
        "params": [
            {"channel": "gnabar_hh2", "low": 2.5e-02, "high": 1.0e-01},
            {"channel": "gkbar_hh2", "low": 2.5e-03, "high": 1.0e-02},
            {"channel": "gkbar_im", "low": 1.5e-05, "high": 6.0e-05},
            {"channel": "gcabar_it", "low": 5.0e-04, "high": 2.0e-03},
        ],
        "target_V": None,  # Target voltages
        "target_params": [0.05, 0.005, 3e-5, 0.001],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 5000,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
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
    "segregation": [
        {
            "params": ["g_pas"],
            "voltage": [-100, -65],
            "time": [0, 500],
        },
        {
            "params": ["gnabar_hh2", "gkbar_hh2", "gkbar_im", "gcabar_it"],
            "voltage": [-65, 100],
            "time": [0, 2000],
        },
    ],
    "output": {"folder": "output_Pospischil_sPYr", "produce_plots": True},
    "run_mode": "original",  # "original", "segregated"
}

pospischilsPYr_seg: SimulationConfig = {
    "cell": {
        "hoc_file": "../data/Pospischil/sPyr/template.hoc",
        "modfiles_folder": "../data/Pospischil/sPyr/seg_modfiles",
        "name": "sPYr",
        "passive_properties": {
            "v_rest": -80,
            "r_in": 393.45,
            "tau": 117.425,
            "leak_conductance_variable": "g_pas",  # eg: g_leak
            "leak_reversal_variable": "e_pas",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -80.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-0.1, 4.1, 0.5)),
        "params": [
            {"channel": "gnabar_hh2", "low": 2.5e-02, "high": 1.0e-01},
            {"channel": "gkbar_hh2", "low": 2.5e-03, "high": 1.0e-02},
            {"channel": "gkbar_im", "low": 1.5e-05, "high": 6.0e-05},
            {"channel": "gcabar_it", "low": 5.0e-04, "high": 2.0e-03},
        ],
        "target_V": None,  # Target voltages
        "target_params": [0.05, 0.005, 3e-5, 0.001],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 5000,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
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
    "segregation": [
        {
            "params": ["gnabar_hh2", "gkbar_hh2", "gkbar_im", "gcabar_it"],
            "voltage": [-65, 100],
            "time": [0, 2000],
        },
    ],
    "output": {"folder": "output_Pospischil_sPYr_p", "produce_plots": True},
    "run_mode": "original",  # "original", "segregated"
}

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
        "hto_block_channels": ["gbar_na3", "gbar_nap"],
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
            0.0008, #0.00055, #6e-5, # 0.0008 -> Walt found that this was too low, typo from paper
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
        # { # passive
        #    "params": ["ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        # },
        #{  # lto
        #    "params": ["gbar_nap", "gbar_im"],
        #    "voltage": [-100, 100],  # [-67.5, 100],  # [-67.5, -57.5],
        #},
        #{  # spking / adaptation
        #    "params": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "voltage": [-100, 100],  # [-57.5, 100],  # [-57.5, 0],
        #},
        #{  # hto
        #    "params": ["gbar_nap", "gbar_im"],
        #    "voltage": [-100, 100],  # [-40, 100],  # [-40, -30],
        #},
        ######### TAKE 2 ##########
        #{
        #    "params": ["gbar_na3", "gbar_kdr"]
        #},
        #{
        #    "params": ["gbar_nap", "gbar_im", "gcabar_cadyn", "gsAHPbar_sAHP"]
        #},
        ######### TAKE 3 ########## GOOD RESULTS
        #{
        #    "params": ["gbar_nap", "gbar_im"],
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse", 
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_lto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [10, 0, 10], # custom arima settings
        #},
        #{
        #    "params": ["gbar_na3", "gbar_kdr"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #},
        #{
        #    "params": ["gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 100,
        #    #"learned_variability": 0.5,
        #},
        ######## TAKE 4 ######### Combine 2 and 3 seg modules
        #{
        #    "params": ["gbar_nap", "gbar_im"],
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "use_lto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [10, 0, 10], # custom arima settings
        #},
        #{
        #    "params": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #},
        ######### TAKE 5 ########## MEH RESULTS
        #{
        #    "params": ["gbar_nap", "gbar_im"],
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_lto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [10, 0, 10], # custom arima settings
        #},
        #{
        #    "params": ["gbar_na3", "gbar_kdr"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #},
        #{ # HTO addition
        #    "params": ["gcabar_cadyn", "gsAHPbar_sAHP"], # taken care of with learned variability, na will be blocked
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "nonsaturated_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_hto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [10, 0, 10], # custom arima settings
        #    "learned_variability": 0.2 # 20% of previous bounds
        #}
        ######### TAKE 6 ########## GOOD RESULTS
        #{
        #    "params": ["gbar_nap", "gbar_im"],
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_lto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [10, 0, 10], # custom arima settings
        #},
        #{
        #    "params": ["gbar_na3", "gbar_kdr"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #},
        #{ # Spiking Adaptation - allow variation the learned spiking parameters by 20%
        #    "params": ["gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "learned_variability_params": ["gbar_na3", "gbar_kdr"],
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #    "learned_variability": 0.2,
        #},
        #{ # HTO addition - allow variation of learned im, kdr, cadyn by 20%
        #    "params": [], # taken care of with learned variability, na will be blocked
        #    "learned_variability_params": ["gbar_im", "gbar_kdr", "gcabar_cadyn"],
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "nonsaturated_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_hto_amps": True,
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [4, 0, 4], # custom arima settings
        #    "learned_variability": 0.2 # 20% of previous bounds
        #}
        ######### TAKE 7 ############# Dr. Nair suggested HTO first
        #{ # HTO - na3 and nap are blocked
        #    "params": ["gbar_im", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],  
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "nonsaturated_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_hto_amps": True,
        #    "ramp_time": 1000, #ms to ramp up the amp input
        #    "ramp_splits": 20, # the amps should be steped up n times
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [4, 0, 4], # custom arima settings
        #    "h_tstop": 1500,  # (ms)
        #    "h_i_delay": 250,  # (ms)
        #    "h_i_dur": 1000,  # (ms)
        #},
        #{ # Spiking Adaptation - allow variation the learned spiking parameters by 20%
        #    "params": ["gbar_na3"],
        #    "learned_variability_params": ["gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "learned_variability": 0.5,
        #    "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
        #    "selection_metric": "fi_error",
        #    "num_epochs": 200,
        #},
        #{ # LTO
        #    "params": ["gbar_nap"], # Everything has already been "learned" only vary
        #    "learned_variability_params": ["gbar_im"],
        #    "learned_variability": 0.2,
        #    "model_class": "RandomForest", #"SimpleSummaryNet",
        #    "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
        #    "num_epochs": 1000,
        #    "train_spiking_only": False,
        #    "train_amplitude_frequency": True,
        #    "use_lto_amps": True,
        #    "ramp_time": 1000, #ms to ramp up the amp input
        #    "ramp_splits": 20, # the amps should be steped up n times
        #    "use_spike_summary_stats": False, # don't use spike summary stats for training
        #    "arima_order": [4, 0, 4], # custom arima settings
        #    "h_tstop": 1500,  # (ms)
        #    "h_i_delay": 250,  # (ms)
        #    "h_i_dur": 1000,  # (ms)
        #},
        ############ TAKE 8 ###############
        { # LTO
            "params": ["gbar_nap", "gbar_im"],
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "train_amplitude_frequency": True,
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
            "learned_variability": 0.2,
            "model_class": "RandomForest", #"ConvolutionEmbeddingNet",
            "selection_metric": "fi_error",
            "num_epochs": 200,
        },
        { # HTO - na3 and nap are blocked
            "params": [],
            "learned_variability_params": ["gbar_im", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "learned_variability": 0.2,
            "model_class": "RandomForest", #"SimpleSummaryNet",
            "selection_metric": "mse", #"amplitude_frequency_error", #"mse",
            "num_epochs": 1000,
            "train_spiking_only": False,
            "nonsaturated_only": False,
            "train_amplitude_frequency": True,
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
        #"amps": [0.1, 0.25, 0.5, 0.75, 1.0],
        "amps": [0.0, 0.1, 0.2, 0.3, 0.4],
        "lto_amps": [0.0, 0.025, 0.05, 0.075, 0.1],
        "hto_amps": [2.5, 3.0, 3.5, 4.0, 4.5],
        "lto_block_channels": ["gbar_na3", "gkdrbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
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

LA_A_orig_old = {
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
        "amps": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "params": [
            # {"channel": "glbar_leak", "low": 2.75e-5, "high": 1e-4},  # leak, passive
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05},  # hd, passive
            # {
            #    "channel": "gbar_nap",
            #    "low": 0.000071,
            #    "high": 0.000284,
            # },  # nap, lto and hto
            # {"channel": "gmbar_im", "low": 0.001, "high": 0.004},  # im, lto and hto
            # {
            #   "channel": "gbar_na3",
            #    "low": 0.015,
            #    "high": 0.06,
            # },  # na3, spiking/adaptation
            # {
            #    "channel": "gkdrbar_kdr",
            #    "low": 0.00075,
            #    "high": 0.003,
            # },  # kdr, spiking/adaptation
            # {
            #    "channel": "gcabar_cadyn",
            #    "low": 3e-5,
            #    "high": 1.2e-4,
            # },  # cadyn, spiking/adaptation
            # {
            #    "channel": "gsAHPbar_sAHP",
            #    "low": 0.0045,
            #    "high": 0.018,
            # },  # sahp, spiking/adaptation
            {"channel": "gbar_nap", "high": 0.0006, "low": 0.00015},
            {"channel": "gmbar_im", "high": 0.004, "low": 0.001},
            {"channel": "gbar_na3", "high": 0.06, "low": 0.015},
            {"channel": "gkdrbar_kdr", "high": 0.06, "low": 0.015},
            {"channel": "gcabar_cadyn", "high": 0.00012, "low": 0.00003},
            {"channel": "gsAHPbar_sAHP", "high": 0.018, "low": 0.0045},
        ],
        "target_V_file": "./target_v.json",
        "target_V": None,  # Target voltages
        # "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "target_params": [0.0003, 0.002, 0.03, 0.03, 6e-5, 0.009],
        "num_repeats": 1,
        "num_amps_to_match": 12,
        "num_epochs": 10,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "use_random_forest": True,
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
        #{  # passive
        #    "params": ["glbar_leak", "ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        #},
        #{  # lto
        #    "params": ["gbar_nap", "gmbar_im"],
        #    "voltage": [-67.5, -57.5],
        #},
        #{  # spking / adaptation
        #    "params": ["gbar_na3", "gkdrbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "voltage": [-57.5, 0],
        #},
        #{  # hto
        #    "params": ["gbar_nap", "gmbar_im"],
        #    "voltage": [-40, -30],
        #},
        {
            "params": ["gbar_na3", "gkdrbar_kdr"]
        },
        {
            "params": ["gbar_nap", "gmbar_im", "gcabar_cadyn", "gsAHPbar_sAHP"]
        }
    ],
    "output": {
        "folder": "output_LA_A_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}

LA_C_seg = {
    "cell": {
        "hoc_file": "../data/LA/C/seg_modfiles/template.hoc",
        "modfiles_folder": "../data/LA/C/seg_modfiles",
        "name": "Cell_C",
        "passive_properties": {
            "v_rest": -70.52, #-67,#-69.17387,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -67.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
        "h_celsius": 31.0,
    },
    "optimization_parameters": {
        "amps": [0.1, 0.2, 0.3, 0.4, 0.5],
        "params": [
            {"channel": "gbar_nap", "high": 0.000426, "low": 4.733e-05},
            {"channel": "gbar_im", "high": 0.0018, "low": 0.0002},
            {"channel": "gbar_na3", "high": 0.081, "low": 0.009},
            {"channel": "gbar_kdr", "high": 0.0045, "low": 0.0005},
            {"channel": "gcabar_cadyn", "high": 0.00165, "low": 0.00018333},
            {"channel": "gsAHPbar_sAHP", "high": 0.00015, "low": 1.667e-05},
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
        "target_cell_target_params": [0.00014, 0.001, 0.03, 0.009, 7e-5, 0.00025],
        # ======================================================
        "target_V": None,  # Target voltages
        "target_params": [
            0, #0.000142,
            0, #0.0006,
            0, #0.027,
            0, #0.0015,
            0, #0.00055,
            0, #0.00005,
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
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        # { # passive
        #    "params": ["ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        
        #{  # lto
        #    "params": ["gbar_nap", "gbar_im"],
        #    "voltage": [-100, 100],  # [-67.5, 100],  # [-67.5, -57.5],
        #},
        #{  # spking / adaptation
        #    "params": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
        #    "voltage": [-100, 100],  # [-57.5, 100],  # [-57.5, 0],
        #},
        #{  # hto
        #    "params": ["gbar_nap", "gbar_im"],
        #    "voltage": [-100, 100],  # [-40, 100],  # [-40, -30],
        #},
        {
            "params": ["gbar_na3", "gbar_kdr"],
            "num_epochs": 1000
        },
        {
            "params": ["gbar_nap", "gbar_im", "gcabar_cadyn", "gsAHPbar_sAHP"]
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
           "v_rest": -67,
           "r_in": 141,
           "tau": 30.88,
           "leak_conductance_variable": "glbar_leak",  # eg: g_leak
           "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -67.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
        "h_celsius": 31.0,
    },
    "optimization_parameters": {
        "amps": [0.1, 0.2, 0.3, 0.4, 0.5],  # list(np.arange(-2, 10, 0.1)),
        "params": [
            # {"channel": "glbar_leak", "low": 2.75e-5, "high": 1e-4},  # leak, passive
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05},  # hd, passive
            # {
            #    "channel": "gbar_nap",
            #    "low": 0.000071,
            #    "high": 0.000284,
            # },  # nap, lto and hto
            # {"channel": "gmbar_im", "low": 0.001, "high": 0.004},  # im, lto and hto
            # {
            #   "channel": "gbar_na3",
            #    "low": 0.015,
            #    "high": 0.06,
            # },  # na3, spiking/adaptation
            # {
            #    "channel": "gkdrbar_kdr",
            #    "low": 0.00075,
            #    "high": 0.003,
            # },  # kdr, spiking/adaptation
            # {
            #    "channel": "gcabar_cadyn",
            #    "low": 3e-5,
            #    "high": 1.2e-4,
            # },  # cadyn, spiking/adaptation
            # {
            #    "channel": "gsAHPbar_sAHP",
            #    "low": 0.0045,
            #    "high": 0.018,
            # },  # sahp, spiking/adaptation
            {"channel": "gbar_nap", "high": 0.00042, "low": 4.667e-05},
            {"channel": "gmbar_im", "high": 0.003, "low": 0.00033333},
            {"channel": "gbar_na3", "high": 0.09, "low": 0.01},
            {"channel": "gkdrbar_kdr", "high": 0.027, "low": 0.003},
            {"channel": "gcabar_cadyn", "high": 0.00021, "low": 2.333e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.00075, "low": 8.333e-05},
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        # ======================================================
        "target_V": None,  # Target voltages
        # "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "target_params": [0.00014, 0.001, 0.03, 0.009, 7e-5, 0.00025],
        "num_repeats": 1,
        "num_amps_to_match": 12,
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


Simple_Spiker_seg = {
    "cell": {
        "hoc_file": "../data/Spiker/seg/template.hoc",
        "modfiles_folder": "../data/Spiker/seg",
        "name": "Simple_Spiker_seg",
        "passive_properties": {
            "v_rest": -65,
            "r_in": 192,
            "tau": 2.575,
            "leak_conductance_variable": "gl_hh_seg",  # eg: g_leak
            "leak_reversal_variable": "el_hh_seg",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -65.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
        "h_celsius": 6.3,
    },
    "optimization_parameters": {
        "amps": [0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
        "params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {
                "channel": "gnabar_hh_seg",
                "high": 0.36,
                "low": 0.04,
            },
            {
                "channel": "gkbar_hh_seg",
                "high": 0.108,
                "low": 0.012,
            },
        ],
        "target_V": None,  # Target voltages
        "target_params": [
            0.0,
            0.0,
        ],
        # ======================================================
        "target_V_file": "./target_v.json",
        "target_cell": {
            "hoc_file": "../data/Spiker/orig/template.hoc",
            "modfiles_folder": "../data/Spiker/orig",
            "name": "Simple_Spiker_orig",
        },
        "target_cell_params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {
                "channel": "gnabar_hh_orig",
            },
            {
                "channel": "gkbar_hh_orig",
            },
        ],
        "target_cell_target_params": [
            0.12,
            0.036,
        ],
        # ======================================================
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 2500,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -30,  # (mV)
        "arima_order": [4, 0, 4],
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        # { # passive
        #    "params": ["ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        # },
        {  # lto
            "params": ["gnabar_hh_seg", "gkbar_hh_seg"],
            "voltage": [-100, 100],  # [-67.5, 100],  # [-67.5, -57.5],
        },
    ],
    "output": {
        "folder": "output_Simple_Spiker_seg",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Segregated",
    },
    "run_mode": "original",  # "original", "segregated"
}

Simple_Spiker_orig = {
    "cell": {
        "hoc_file": "../data/Spiker/orig/template.hoc",
        "modfiles_folder": "../data/Spiker/orig",
        "name": "Simple_Spiker_orig",
        "passive_properties": {
            "v_rest": -65,
            "r_in": 192,
            "tau": 2.575,
            "leak_conductance_variable": "gl_hh_orig",  # eg: g_leak
            "leak_reversal_variable": "el_hh_orig",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -65.0,  # (mV)
        "h_tstop": 1000,  # (ms)
        "h_i_delay": 250,  # (ms)
        "h_i_dur": 500,  # (ms)
        "h_dt": 0.1,
        "h_celsius": 6.3,
    },
    "optimization_parameters": {
        "amps": [0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
        "params": [
            # {"channel": "gl_hh_orig", "low": 0.0001, "high": 0.009},  # hd, passive
            {
                "channel": "gnabar_hh_orig",
                "high": 0.36,
                "low": 0.04,
            },
            {
                "channel": "gkbar_hh_orig",
                "high": 0.108,
                "low": 0.012,
            },
        ],
        "target_V": None,  # Target voltages
        "target_params": [
            # 0.0003,
            0.12,
            0.036,
        ],
        "target_V_file": "./target_v.json",
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 2500,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
        },
        "decimate_factor": 10,
    },
    "summary_features": {
        "spike_threshold": -30,  # (mV)
        "arima_order": [4, 0, 4],
        # Target-sim match conditions (max abs diff between sim and target)
        "mc_num_spikes": 1,
        "mc_interspike_time": 200,  # (ms)
        "mc_min_v": 1,  # (mV)
        "mc_mean_v": 2,  # (mV)
        "mc_max_v": 1,  # (mV)
    },
    "segregation": [
        # { # passive
        #    "params": ["ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        # },
        {  # lto
            "params": ["gnabar_hh_orig", "gkbar_hh_orig"],
            "voltage": [-100, 100],  # [-67.5, 100],  # [-67.5, -57.5],
        },
    ],
    "output": {
        "folder": "output_Simple_Spiker_orig",
        "produce_plots": True,
        "target_label": "User Trace",
        "simulated_label": "Model ACT-Original",
    },
    "run_mode": "original",  # "original", "segregated"
}


selected_config = LA_A_seg
