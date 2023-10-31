import pprint
import sys

params = {
    "params": [
            # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
            {"channel": "gbar_nap", "high": 0.000426, "low": 4.736e-05},
            {"channel": "gbar_im", "high": 0.006, "low": 0.000666},
            {"channel": "gbar_na3", "high": 0.09, "low": 0.01},
            {"channel": "gbar_kdr", "high": 0.0045, "low": 0.0005},
            {"channel": "gcabar_cadyn", "high": 0.00018, "low": 2e-05},
            {"channel": "gsAHPbar_sAHP", "high": 0.026996, "low": 0.0029996},
        ],
        "target_V": None,  # Target voltages
        # "target_params": [5.5e-5, 2.3e-05, 0.000142, 0.002, 0.03, 0.0015, 6e-5, 0.009],
        "target_params": [0.00014, 0.001, 0.03, 0.009, 7e-5, 0.00025]
}

for i, (target, param) in enumerate(zip(params["target_params"], params["params"])):
    params["params"][i]["low"] = round(target / float(sys.argv[-1]),8)
    params["params"][i]["high"] = round(target * float(sys.argv[-1]),8)

pprint.pprint(params)
