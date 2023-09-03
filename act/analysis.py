import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from act.act_types import SimulationConfig
from act.metrics import correlation_score, mse_score
from act.optim import ACTOptimizer


def save_plot(
    amp: float, output_folder: str, simulated_data=None, target_V=None, output_file=None
):
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    title = f"I = {(amp * 1000):.0f} nA"
    if simulated_data is not None:
        ax.plot(simulated_data.flatten(), label="Simulated")
    if target_V is not None:
        ax.plot(target_V.flatten(), label="Target")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.grid()

    if not output_file:
        output_file = os.path.join(output_folder, f"{(amp * 1000):.0f}nA.png")
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
) -> None:
    optim = ACTOptimizer(simulation_config=simulation_config)
    params = [
        p["channel"] for p in simulation_config["optimization_parameters"]["params"]
    ]
    simulated_data = optim.simulate(
        amp, params, predicted_params_values.detach().numpy()
    )
    simulated_data = optim.resample_voltage(
        V=simulated_data.reshape((1, -1)), num_obs=target_V.shape[1]
    )

    save_plot(amp, output_folder, simulated_data, target_V)


def save_mse_corr(
    target_V: torch.Tensor,
    simulation_config: SimulationConfig,
    predicted_params_values: torch.Tensor,
    output_folder: str,
) -> None:
    with open(os.path.join(output_folder, "metrics.csv"), "w") as file:
        file.write(f"amp,mse,corr\n")

    optim = ACTOptimizer(simulation_config=simulation_config)
    for ind, amp in enumerate(simulation_config["optimization_parameters"]["amps"]):
        params = [
            p["channel"] for p in simulation_config["optimization_parameters"]["params"]
        ]
        sim_data = optim.simulate(amp, params, predicted_params_values.detach().numpy())
        simulated_data = optim.resample_voltage(
            V=sim_data.reshape((1, -1)), num_obs=target_V.shape[1]
        )
        mse = mse_score(target_V[ind].reshape(-1, 1), simulated_data.reshape(-1, 1))
        corr = correlation_score(
            target_V[ind].reshape(1, -1), simulated_data.reshape(1, -1)
        )

        with open(os.path.join(output_folder, "metrics.csv"), "a") as file:
            file.write(f"{amp},{mse},{corr}\n")


def print_run_stats(config: SimulationConfig):
    output_folder = config["output"]["folder"]
    run_mode = config["run_mode"]
    target_params = config["optimization_parameters"].get("target_params")
    passive_json_path = os.path.join(
        output_folder, run_mode, "pred_passive_properties.json"
    )

    metrics = pd.read_csv(os.path.join(output_folder, run_mode, "metrics.csv"))
    preds_df = pd.read_csv(
        os.path.join(output_folder, run_mode, "pred.csv"), index_col=0
    )
    passive_json = None
    if os.path.isfile(passive_json_path):
        with open(passive_json_path, "r") as fp:
            passive_json = json.load(fp)

    preds = np.array(preds_df)
    print(output_folder, ":", run_mode)
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
    if passive_json:
        print()
        print("Passive properties:")
        print(json.dumps(passive_json, indent=2))
        print("----------\n")
