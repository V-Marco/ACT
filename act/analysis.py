import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from act.act_types import SimulationConfig
from act.metrics import correlation_score, mse_score, torch_correlation_score
from act.optim import ACTOptimizer
from act import utils


def save_plot(
    amp: float,
    output_folder: str,
    simulated_data=None,
    target_V=None,
    output_file=None,
    simulated_label="Simulated",
    target_label="Target",
    dt=0.025,
    fontsize='xx-large',
    tick_fontsize='x-large',
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    title = f"I = {(amp * 1000):.0f} pA"
    if simulated_data is not None:
        times = np.arange(0, int(len(simulated_data.flatten()) * dt), dt)
        ax.plot(times, simulated_data.flatten(), label=simulated_label, alpha=0.7)
    if target_V is not None:
        times = np.arange(0, int(len(target_V.flatten()) * dt), dt)
        ax.plot(times, target_V.flatten(), label=target_label, alpha=0.7)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Timestamp (ms)", fontsize=fontsize)
    ax.set_ylabel("V (mV)", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels() +
              ax.get_legend().get_texts()):
     item.set_fontsize(tick_fontsize)
    ax.grid()
    fig.tight_layout()
    if not output_file:
        output_file = os.path.join(output_folder, f"{(amp * 1000):.0f}pA.png")
    else:
        output_file = os.path.join(output_folder, output_file)
    plt.savefig(output_file)
    plt.close()


def save_prediction_plots(
    target_V: torch.Tensor,
    amp: list,
    simulation_config: SimulationConfig,
    predicted_params_values: torch.Tensor,
    output_folder: str,
    output_file: str = None,
) -> np.ndarray:
    optim = ACTOptimizer(simulation_config=simulation_config)
    params = [
        p["channel"] for p in simulation_config["optimization_parameters"]["params"]
    ]
    simulated_data = optim.simulate(amp, params, predicted_params_values, cut_ramp=False)
    decimate_factor = simulation_config["optimization_parameters"].get(
        "decimate_factor"
    )
    if decimate_factor:
        print(f"decimate_factor set - reducing sims voltage by {decimate_factor}x")
        from scipy import signal

        simulated_data = torch.tensor(
            signal.decimate(simulated_data.cpu(), decimate_factor).copy()
        )
    if simulated_data.reshape((1, -1)).shape[1] != target_V.shape[1]:
        simulated_data = optim.resample_voltage(
            V=simulated_data.reshape((1, -1)), num_obs=target_V.shape[1]
        )
    else:
        simulated_data = simulated_data.reshape((1, -1))
    dt = simulation_config["simulation_parameters"]["h_dt"]
    if decimate_factor:
        dt = dt * decimate_factor
    simulated_label = simulation_config["output"].get("simulated_label", "Simulated")
    target_label = simulation_config["output"].get("target_label", "Target")
    save_plot(
        amp,
        output_folder,
        simulated_data.cpu().detach().numpy(),
        target_V,
        dt=dt,
        simulated_label=simulated_label,
        target_label=target_label,
        output_file=output_file,
    )

    return simulated_data


def save_mse_corr(
    target_V: torch.Tensor,
    simulation_config: SimulationConfig,
    predicted_params_values: list,
    output_folder: str,
    amps=None,
) -> None:
    with open(os.path.join(output_folder, "metrics.csv"), "w") as file:
        file.write(f"amp,mse,corr\n")
    if not amps:
        amps = simulation_config["optimization_parameters"]["amps"]
    optim = ACTOptimizer(simulation_config=simulation_config)
    for ind, amp in enumerate(amps):
        params = [
            p["channel"] for p in simulation_config["optimization_parameters"]["params"]
        ]
        sim_data = optim.simulate(amp, params, predicted_params_values, cut_ramp=False)
        decimate_factor = simulation_config["optimization_parameters"].get(
            "decimate_factor"
        )
        if decimate_factor:
            print(f"decimate_factor set - reducing sims voltage by {decimate_factor}x")
            from scipy import signal

            sim_data = torch.tensor(
                signal.decimate(sim_data.cpu(), decimate_factor).copy()
            )

        if sim_data.reshape((1, -1)).shape[1] != target_V.shape[1]:
            simulated_data = optim.resample_voltage(
                V=sim_data.reshape((1, -1)), num_obs=target_V.shape[1]
            )
        else:
            simulated_data = sim_data.reshape((1, -1))

        mse = mse_score(target_V[ind].reshape(-1, 1), simulated_data.reshape(-1, 1))

        corr = torch_correlation_score(
            target_V[ind].reshape(1, -1), simulated_data.reshape(1, -1)
        )

        with open(os.path.join(output_folder, "metrics.csv"), "a") as file:
            file.write(f"{amp},{mse},{corr}\n")


def print_run_stats(config: SimulationConfig):
    output_folder = utils.get_output_folder_name(config)
    target_params = config["optimization_parameters"].get("target_params")
    if(config["run_mode"] == "segregated"):
        segregation_index = utils.get_segregation_index(config)
        segregation_dir = f"seg_module_{segregation_index+1}/"
        model_data_dir = os.path.join(output_folder, segregation_dir)
    else:
        model_data_dir = output_folder + "model_data/"
    pred_passive_json_path = model_data_dir + "pred_passive_properties.json"
    metrics = pd.read_csv(model_data_dir + "metrics.csv")
    preds_df = pd.read_csv(model_data_dir + "pred.csv", index_col=0)
    pred_passive_json = None
    if os.path.isfile(pred_passive_json_path):
        with open(pred_passive_json_path, "r") as fp:
            pred_passive_json = json.load(fp)

    target_passive_json = None
    if config["cell"].get("passive_properties"):
        target_passive_json = config["cell"]["passive_properties"]
        if config["optimization_parameters"].get("target_passive_properties"):
            target_passive_json = config["optimization_parameters"][
                "target_passive_properties"
            ]

    preds = np.array(preds_df)
    print(output_folder)
    print(f"Med MSE: {metrics['mse'].median():.4f} ({metrics['mse'].std():.4f})")
    print(f"Med Corr: {metrics['corr'].median():.4f} ({metrics['corr'].std():.4f})")
    print()
    print("Predicted values:")
    print(preds_df)
    if target_params:
        print("Target values:")
        print(pd.DataFrame([target_params], columns=preds_df.columns))
        print("Error:")
        print(preds_df - target_params)
        print()
        print(f"Pred MAE: {np.mean(np.abs(target_params - preds)):.4f}")
    if pred_passive_json:
        print()
        simulated_label = config["output"].get("simulated_label", "Simulated")
        print(f"{simulated_label} Passive properties:")
        print(json.dumps(pred_passive_json, indent=2))
        print("----------\n")
    if target_passive_json:
        print()
        target_label = config["output"].get("target_label", "Target")
        print(f"{target_label} Passive properties:")
        print(json.dumps(target_passive_json, indent=2))
        print("----------\n")
    if pred_passive_json and target_passive_json:
        print()
        print("Passive properties errror")
        print(
            f"v_rest: {(pred_passive_json['v_rest']-target_passive_json['v_rest']):.2f}"
        )
        print(f"r_in: {(pred_passive_json['r_in']-target_passive_json['r_in']):.2f}")
        print(f"tau: {(pred_passive_json['tau']-target_passive_json['tau']):.2f}")
        print("----------\n")

    traces_file = model_data_dir + "traces.h5"
    
    simulated_traces, target_traces, amps = utils.load_final_traces(traces_file)
    error = utils.get_fi_curve_error(
        simulated_traces, target_traces, amps, print_info=True
    )
    print(
        f"Simulated and target FI curve error [SUM((simulated-target)/target)/n]: {error}"
    )


def plot_fi_curves(
    spike_counts_list,
    amps,
    labels,
    title="FI Curves",
    ignore_negative=True,
    output_file=None,
    fontsize='xx-large',
    tick_fontsize='x-large',
):
    if ignore_negative:
        amps = amps[amps >= 0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for spike_counts, label in zip(spike_counts_list, labels):
        print(f"{label}: {amps*1e3} pA : {spike_counts} Hz")
        ax.plot(amps * 1e3, spike_counts, label=label, alpha=0.75)

    # plt.ylim((0,np.max(np.array(spike_counts))))
    ax.legend(fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel("Hz", fontsize=fontsize)
    ax.set_xlabel("pA", fontsize=fontsize)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels() +
              ax.get_legend().get_texts()):
        item.set_fontsize(tick_fontsize)
    ax.grid()
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
