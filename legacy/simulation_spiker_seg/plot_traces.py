import h5py
from matplotlib import pyplot as plt
import os
import sys
import json
from legacy.legacy_from_act import utils
import torch
import numpy as np
import math

from simulation_configs import selected_config

config = selected_config


def plot_trace(
    amp: float,
    target_V=None,
    label="Simulated",
    dt=0.1,
    ax=None
):
    decimate_factor = config["optimization_parameters"].get("decimate_factor")
    if decimate_factor:
        dt = dt * decimate_factor
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    title = f"I = {(amp * 1000):.0f} nA"
    if target_V is not None:
        times = np.arange(0, int(len(target_V.flatten()) * dt), dt)
        ax.plot(times, target_V.flatten(), label=label)
    #ax.set_title(title)
    #ax.set_xlabel("Timestamp (ms)")
    #ax.set_ylabel("V (mV)")
    ax.legend()
    ax.grid()

    #plt.show()


def stats(traces, params_dict):
    traces_t, params_t, amps_t = utils.load_parametric_traces(config)
    traces, params, amps = (
        traces_t.cpu().detach().numpy(),
        params_t.cpu().detach().numpy(),
        amps_t.cpu().detach().numpy(),
    )

    amp_list = list(set(amps))
    amp_list.sort()

    print(f"{len(traces)} total traces")
    print(f"{amp_list} total amps supplied")
    print(f"{int(len(traces)/len(amp_list))} total unique parameter sets")

    #spiking_traces, spiking_params, spiking_amps = utils.extract_spiking_traces(
    #    traces_t, params_t, amps_t
    #)
    cell_id = 0
    #plot_trace(
    #    spiking_amps[cell_id].cpu(), spiking_traces[cell_id].cpu(), f"Cell {cell_id}"
    #)
    if True: # only really works for simple spiker case since there are only 2 variables
        total_sims, n_params = params.shape
        sims_per_amp = int(total_sims/len(amp_list))
        n_splits = int(math.sqrt(sims_per_amp))
        for i_amp, amp in enumerate(amp_list):
            figure, axis = plt.subplots(n_splits, n_splits)
            figure.suptitle(f'{amp*1000:.0f}nA')
            for split_na in range(n_splits):
                for split_k in range(n_splits):
                    pos = i_amp * sims_per_amp + (split_na * n_splits + split_k)
                    print(pos)
                    plot_trace(amp, traces[pos], label=f"{params[pos]}", ax=axis[split_na,split_k])
            #figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    traces_path = "output/v_report.h5"
    params_path = "parameter_values.json"

    if not os.path.exists(traces_path) or not os.path.exists(params_path):
        print(
            f"{traces_path} or {params_path} not found. Generate traces before running."
        )
        exit()

    traces_h5 = h5py.File(traces_path)
    with open(params_path, "r") as json_file:
        params_dict = json.load(json_file)


    stats(traces_h5, params_dict)
