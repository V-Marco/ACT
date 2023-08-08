import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from act.optim import ACTOptimizer
from act.metrics import correlation_score, mse_score

def save_prediction_plots(target_V: torch.Tensor, simulation_constants: object, predicted_params_values: torch.Tensor, output_folder: str) -> None:

    _, ax = plt.subplots(1, 5, figsize = (20, 4))
    for ind_amp_to_plot in range(5):
        optim = ACTOptimizer()
        simulated_data = optim.simulate(simulation_constants.amps[ind_amp_to_plot], simulation_constants.params, predicted_params_values.detach().numpy())
        simulated_data = optim.resample_voltage(V = simulated_data.reshape((1, -1)), num_obs = target_V.shape[1])

        title = f"I = {simulation_constants.amps[ind_amp_to_plot]}"
        ax[ind_amp_to_plot].plot(simulated_data.flatten(), label = "Simulated")
        ax[ind_amp_to_plot].plot(target_V[ind_amp_to_plot], label = "Target")
        ax[ind_amp_to_plot].set_title(title)
        ax[ind_amp_to_plot].set_xlabel("Time (ms)")
        ax[ind_amp_to_plot].set_ylabel("V (mV)")
        ax[ind_amp_to_plot].legend()
        ax[ind_amp_to_plot].grid()

    fig_title = f"{np.min(simulation_constants.amps)}-{np.max(simulation_constants.amps)}_na.png"
    plt.savefig(os.path.join(output_folder, fig_title))
    plt.close()

def save_mse_corr(target_V: torch.Tensor, simulation_constants: object, predicted_params_values: torch.Tensor, output_folder: str) -> None:

    with open(os.path.join(output_folder, "metrics.csv"), "w") as file:
        file.write(f"amp,mse,corr\n")

    optim = ACTOptimizer(simulation_constants = simulation_constants)
    for ind, amp in enumerate(simulation_constants.amps):
        sim_data = optim.simulate(amp, simulation_constants.params, predicted_params_values.detach().numpy())
        simulated_data = optim.resample_voltage(V = sim_data.reshape((1, -1)), num_obs = target_V.shape[1])
        mse = mse_score(target_V[ind].reshape(-1, 1), simulated_data.reshape(-1, 1))
        corr = correlation_score(target_V[ind].reshape(-1, 1), simulated_data.reshape(-1, 1))

        with open(os.path.join(output_folder, "metrics.csv"), "a") as file:
            file.write(f"{amp},{mse},{corr}\n")

