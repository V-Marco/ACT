import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from act.act_types import SimulationConfig
from act.metrics import correlation_score, mse_score
from act.optim import ACTOptimizer


def save_prediction_plots(
    target_V: torch.Tensor,
    amp: list,
    simulation_config: SimulationConfig,
    predicted_params_values: torch.Tensor,
    output_folder: str,
) -> None:
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
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

    title = f"I = {amp} nA"
    ax.plot(simulated_data.flatten(), label="Simulated")
    ax.plot(target_V.flatten(), label="Target")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.grid()

    plt.savefig(os.path.join(output_folder, f"{amp * 1000}nA.png"))
    plt.close()


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

    metrics = pd.read_csv(os.path.join(output_folder, run_mode, "metrics.csv"))
    preds = np.array(
        pd.read_csv(os.path.join(output_folder, run_mode, "pred.csv"), index_col=0)
    )
    print(output_folder, ":", run_mode)
    print(f"Med MSE: {metrics['mse'].median():.4f} ({metrics['mse'].std():.4f})")
    print(f"Med Corr: {metrics['corr'].median():.4f} ({metrics['corr'].std():.4f})")
    if target_params:
        print(f"Pred MAE: {np.mean(np.abs(target_params - preds)):.4f}")
    print("----------\n")
