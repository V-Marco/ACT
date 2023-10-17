# Execute using `NOCUDA=1 python generate_arma_stats.py`

from act import utils

import warnings

warnings.filterwarnings("ignore")

config = pospischilsPYr_passive

if __name__ == "__main__":
    traces, params, amps = utils.load_parametric_traces(config)
    utils.arima_coefs_proc_map(traces)
    print("done")
