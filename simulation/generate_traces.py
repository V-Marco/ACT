from simulation_configs import selected_config

import os

import sys
sys.path.append("../")
import meta_sweep

from act.utils import build_parametric_network, generate_parametric_traces, get_sim_output_folder_name

if __name__ == "__main__":
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()
    if "build" in sys.argv:
        print("Building Network")
        build_parametric_network(selected_config)
    else:
        # Check if data already exists: spikes.h5 and v_report.h5
        sim_data_dir = get_sim_output_folder_name(selected_config)

        if (os.path.exists(sim_data_dir + "spikes.h5")) and (os.path.exists(sim_data_dir + "v_report.h5")):
            print("------------------------------------------------------------")
            print(f"DATA ALREADY GENERATED FOR THIS MODULE AT: {sim_data_dir}")
            print("------------------------------------------------------------")
        else:
            print("Generating Traces")
            generate_parametric_traces(selected_config)
