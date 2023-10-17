import h5py
from matplotlib import pyplot as plt
import os
import sys
import json
from act import utils
import torch
import numpy as np

from simulation_configs import LA_A_seg, pospischilsPYr_passive

config = pospischilsPYr_passive


def plot_trace(
    amp: float,
    target_V=None,
    label="Simulated",
    dt=0.1,
):
    decimate_factor = config["optimization_parameters"].get("decimate_factor")
    if decimate_factor:
        dt = dt * decimate_factor
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    title = f"I = {(amp * 1000):.0f} nA"
    if target_V is not None:
        times = np.arange(0, int(len(target_V.flatten()) * dt), dt)
        ax.plot(times, target_V.flatten(), label=label)
    ax.set_title(title)
    ax.set_xlabel("Timestamp (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.grid()

    plt.show()


def stats(traces, params_dict):
    traces_t, params_t, amps_t = utils.load_parametric_traces(config)
    traces, params, amps = (
        traces_t.cpu().detach().numpy(),
        params_t.cpu().detach().numpy(),
        amps_t.cpu().detach().numpy(),
    )

    amp_list = list(set(amps))

    print(f"{len(traces)} total traces")
    print(f"{amp_list} total amps supplied")
    print(f"{int(len(traces)/len(amp_list))} total unique parameter sets")

    spiking_traces, spiking_params, spiking_amps = utils.extract_spiking_traces(
        traces_t, params_t, amps_t
    )
    cell_id = 0
    plot_trace(
        spiking_amps[cell_id].cpu(), spiking_traces[cell_id].cpu(), f"Cell {cell_id}"
    )

    import pdb

    pdb.set_trace()


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
