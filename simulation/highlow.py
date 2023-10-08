import pprint
import sys

params = {
    "params": [
        # {"channel": "ghdbar_hd", "low": 1.15e-05, "high": 4.6e-05}, # hd, passive
        {
            "channel": "gbar_nap",
            "low": 4.73333333e-05,  # 0.000071,
            "high": 4.26e-04,  # 0.000284,
        },  # nap, lto and hto
        {
            "channel": "gbar_im",
            "low": 6.66666667e-04,
            "high": 6.00e-03,
        },  # im, lto and hto
        {
            "channel": "gbar_na3",
            "low": 1.00000000e-02,  # 0.015,
            "high": 9.00e-02,
        },  # na3, spiking/adaptation
        {
            "channel": "gbar_kdr",
            "low": 5.00000000e-04,
            "high": 4.50e-03,
        },  # kdr, spiking/adaptation
        {
            "channel": "gcabar_cadyn",
            "low": 2.00000000e-05,
            "high": 1.80e-04,
        },  # cadyn, spiking/adaptation
        {
            "channel": "gsAHPbar_sAHP",
            "low": 3.00000000e-03,
            "high": 2.70e-02,
        },  # sahp, spiking/adaptation
    ],
    "target_V": None,  # Target voltages
    "target_params": [
        0.000142,
        0.002,
        0.03,
        0.0015,
        6e-5,
        0.009,
    ],
}

for i, (target, param) in enumerate(zip(params["target_params"], params["params"])):
    params["params"][i]["low"] = target / float(sys.argv[-1])
    params["params"][i]["high"] = target * float(sys.argv[-1])

pprint.pprint(params)
