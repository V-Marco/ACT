# Execute using `NOCUDA=1 python generate_arma_stats.py`

from act import utils
from simulation_configs import selected_config
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    traces, params, amps = utils.load_parametric_traces(selected_config)
    utils.arima_coefs_proc_map(traces)
