import h5py
import torch
import os

from legacy import utils

from simulation_configs import selected_config

if __name__ == "__main__":
    traces_file = os.path.join(
        selected_config["output"]["folder"], selected_config["run_mode"], "traces.h5"
    )
    simulated_traces, target_traces, amps = utils.load_final_traces(traces_file)
    error = utils.get_fi_curve_error(simulated_traces, target_traces, amps)
    print(
        f"Simulated and target FI curve error [SUM((target-actual)/actual)/n]: {error}"
    )
