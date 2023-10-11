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


pospischilsPYr: SimulationConfig = {
    "cell": {
        "hoc_file": "../data/Pospischil/sPyr/template.hoc",
        "modfiles_folder": "../data/Pospischil/sPyr/seg_modfiles",
        "name": "sPYr",
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
        "target_V": None,  # Target voltages
        "target_params": [1e-5, 0.05, 0.005, 3e-5, 0.001],
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
            "params": ["gnabar_hh2", "gkbar_hh2", "gkbar_im", "gcabar_it"],
            "voltage": [-65, 100],
            "time": [0, 2000],
        },
    ],
    "output": {"folder": "output_Pospischil_sPYr", "produce_plots": True},
    "run_mode": "original",  # "original", "segregated"
}

pospischilsPYr_passive: SimulationConfig = {
    "cell": {
        "hoc_file": "../data/Pospischil/sPyr/template.hoc",
        "modfiles_folder": "../data/Pospischil/sPyr/seg_modfiles",
        "name": "sPYr",
        "passive_properties": {
            "v_rest": -60,
            "r_in": 393.45,
            "tau": 117.425,
            "leak_conductance_variable": "g_pas",  # eg: g_leak
            "leak_reversal_variable": "e_pas",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -60.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.025,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-2, 10, 0.1)),
        "params": [
            {"channel": "gnabar_hh2", "low": 2.5e-02, "high": 1.0e-01},
            {"channel": "gkbar_hh2", "low": 2.5e-03, "high": 1.0e-02},
            {"channel": "gkbar_im", "low": 1.5e-05, "high": 6.0e-05},
            {"channel": "gcabar_it", "low": 5.0e-04, "high": 2.0e-03},
        ],
        "target_V": None,  # Target voltages
        "target_params": [0.05, 0.005, 3e-5, 0.001],
        "num_repeats": 3,
        "num_amps_to_match": 12,
        "num_epochs": 5000,
        "skip_match_voltage": True,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
            "amps": list(np.arange(0.05, 3.5, 0.75)),  # list(np.arange(0.0, 3.0, 1.0))
        },
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
        "hoc_file": "../data/LA/A/template.hoc",
        "modfiles_folder": "../data/LA/A/seg_modfiles_modeldb",
        "name": "Cell_A",
        "passive_properties": {
            "v_rest": -69.17387,
            "r_in": 141,
            "tau": 30.88,
            "leak_conductance_variable": "glbar_leak",  # eg: g_leak
            "leak_reversal_variable": "el_leak",  # eg: e_leak
        },
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.1,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-0.1, 3, 0.05)),
        "params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {"channel": "gbar_nap", "high": 0.000426, "low": 4.736e-05},
            {"channel": "gbar_im", "high": 0.006, "low": 0.000666},
            {"channel": "gbar_na3", "high": 0.09, "low": 0.01},
            {"channel": "gbar_kdr", "high": 0.0045, "low": 0.0005},
            {"channel": "gcabar_cadyn", "high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.026996, "low": 0.0029996},
        ],
        "skip_match_voltage": True,
        "decimate_factor": 10,
        "target_V": None,  # Target voltages
        "target_params": [
            0.000142,
            0.002,
            0.03,
            0.0015,
            6e-5,
            0.009,
        ],  # [2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "num_repeats": 1,
        "num_amps_to_match": 1,
        "num_epochs": 5000,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
            "amps": list(np.arange(0.05, 3.5, 0.75)),  # list(np.arange(0.0, 3.0, 1.0))
        },
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
        # { # passive
        #    "params": ["ghdbar_hd"],
        #    "voltage": [-80, -67.5],
        # },
        {  # lto
            "params": ["gbar_nap", "gbar_im"],
            "voltage": [-100, 100],  # [-67.5, 100],  # [-67.5, -57.5],
        },
        {  # spking / adaptation
            "params": ["gbar_na3", "gbar_kdr", "gcabar_cadyn", "gsAHPbar_sAHP"],
            "voltage": [-100, 100],  # [-57.5, 100],  # [-57.5, 0],
        },
        {  # hto
            "params": ["gbar_nap", "gbar_im"],
            "voltage": [-100, 100],  # [-40, 100],  # [-40, -30],
        },
    ],
    "output": {
        "folder": "output_LA_A",
        "produce_plots": True,
        "target_label": "ModelDB Segregated",
        "simulated_label": "Model ACT",
    },
    "run_mode": "original",  # "original", "segregated"
}

LA_A_orig = {
    "cell": {
        "hoc_file": "../data/LA/A/template.hoc",
        "modfiles_folder": "../data/LA/A/orig_modfiles",
        "name": "Cell_A",
    },
    "simulation_parameters": {
        "h_v_init": -70.0,  # (mV)
        "h_tstop": 2000,  # (ms)
        "h_i_delay": 500,  # (ms)
        "h_i_dur": 1500,  # (ms)
        "h_dt": 0.025,
    },
    "optimization_parameters": {
        "amps": list(np.arange(-2, 3, 0.025)),  # list(np.arange(-2, 10, 0.1)),
        "params": [
            {"channel": "glbar_leak", "low": 2.75e-5, "high": 1e-4},  # leak, passive
            {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05},  # hd, passive
            {
                "channel": "gbar_nap",
                "low": 0.000071,
                "high": 0.000284,
            },  # nap, lto and hto
            {"channel": "gmbar_im", "low": 0.001, "high": 0.004},  # im, lto and hto
            {
                "channel": "gbar_na3",
                "low": 0.015,
                "high": 0.06,
            },  # na3, spiking/adaptation
            {
                "channel": "gkdrbar_kdr",
                "low": 0.00075,
                "high": 0.003,
            },  # kdr, spiking/adaptation
            {
                "channel": "gcabar_cadyn",
                "low": 3e-5,
                "high": 1.2e-4,
            },  # cadyn, spiking/adaptation
            {
                "channel": "gsAHPbar_sAHP",
                "low": 0.0045,
                "high": 0.018,
            },  # sahp, spiking/adaptation
        ],
        "target_V": None,  # Target voltages
        "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "num_repeats": 3,
        "num_amps_to_match": 12,
        "num_epochs": 5000,
        "parametric_distribution": {  # sample the parameter space for training if n_slices is > 1
            "n_slices": 5,
            "amps": list(np.arange(0.0, 3.0, 1.0)),
        },
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
        "folder": "output_LA_A",
        "produce_plots": True,
        "target_label": "ModelDB Original",
        "simulated_label": "Model ACT",
    },
    "run_mode": "original",  # "original", "segregated"
}
