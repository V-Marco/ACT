# Execute using `NOCUDA=1 python generate_arma_stats.py`
import sys
sys.path.append("../")
from act import utils
from simulation_configs import selected_config
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    traces, params, amps = utils.load_parametric_traces(selected_config)
    segregation_index = utils.get_segregation_index(selected_config)

    arima_order = (10, 0, 10)
    if selected_config.get("summary_features", {}).get("arima_order"):
        arima_order = tuple(selected_config["summary_features"]["arima_order"])
    if selected_config["run_mode"] == "segregated" and selected_config["segregation"][segregation_index].get("arima_order",None):
        print(f"custom arima order for segregation set")
        arima_order = tuple(selected_config["segregation"][segregation_index]["arima_order"])
    print(f"ARIMA order set to {arima_order}")

    output_dir = utils.get_output_folder_name(selected_config) + "sim_data/output/arima_stats.json"
    utils.arima_coefs_proc_map(traces, output_file=output_dir, arima_order=arima_order)
