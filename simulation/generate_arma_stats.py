# Execute using `NOCUDA=1 python generate_arma_stats.py`
import sys
sys.path.append("../")
from act import utils
from proj.ACT.act.DataProcessor_old import DataProcessor
from simulation_configs import selected_config
import warnings
import os.path
import meta_sweep


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()

    dp = DataProcessor()
    dp.generate_arima_coefficients(selected_config)

    #traces, params, amps = utils.load_parametric_traces(selected_config)
    #spikes, interspike_times = DataProcessor.extract_summary_features(traces)
    #print(spikes, interspike_times)
        
