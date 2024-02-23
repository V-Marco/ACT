# Execute using `NOCUDA=1 python generate_arma_stats.py`
import sys
sys.path.append("../")
from act import utils
from simulation_configs import selected_config
import warnings
import os.path
import meta_sweep


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()

    output_dir = utils.get_sim_output_folder_name(selected_config)
    arma_stats_file = output_dir + "arima_stats.json"
    arma_stats_exists = os.path.exists(arma_stats_file)
    generate_arma = selected_config["optimization_parameters"]["generate_arma"]

    if (arma_stats_exists):
        print("--------------------------------------------------------------------")
        print(f"ARMA STATS ALREADY GENERATED - Using stats from: {arma_stats_file}")
        print("--------------------------------------------------------------------")
    elif (not generate_arma):
        print("-------------------------------------------------")
        print("ARMA STATS TURNED OFF IN SIMULATION CONFIGURATION")
        print("-------------------------------------------------")
    else:
        traces, params, amps = utils.load_parametric_traces(selected_config)
        segregation_index = utils.get_segregation_index(selected_config)

        arima_order = (10, 0, 10)
        if selected_config.get("summary_features", {}).get("arima_order"):
            arima_order = tuple(selected_config["summary_features"]["arima_order"])
        if selected_config["run_mode"] == "segregated" and selected_config["segregation"][segregation_index].get("arima_order",None):
            print(f"custom arima order for segregation set")
            arima_order = tuple(selected_config["segregation"][segregation_index]["arima_order"])
        print(f"ARIMA order set to {arima_order}")

        print(output_dir)

        utils.arima_coefs_proc_map(traces, output_file=arma_stats_file, arima_order=arima_order)
        
