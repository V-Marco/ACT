import torch
import numpy as np
import matplotlib.pyplot as plt

def correlation_score(optimizer, observed_data, estimates):
    '''
    Compute correlation score between observed data and data simulated with the optimizer's estimates.

    Parameters:
    ----------
    optimizer: ACTOptimizer
        Optimizer which was used to obtain estimates.

    observed_data: torch.tensor(shape = num_current_injections, 1024)
        Target data to compute correlation with. Most probably, it is the data which was used to train the optimizer.

    esimates: ndarray(shape = num_current_injections, num_parameters)
        Estimates obtained with the optimizer.

    Returns:
    ----------
    corr: ndarray(shape = (num_current_injections, ))
        Sample correlations.
    '''

    simulated_data = optimizer.simulate(estimates, optimizer.parameters)

    cov = (observed_data - torch.mean(observed_data, dim = 1, keepdim = True)) * (simulated_data - torch.mean(simulated_data, dim = 1, keepdim = True))
    cov = torch.sum(cov, dim = 1)

    var0 = torch.sum(torch.square(observed_data - torch.mean(observed_data, dim = 1, keepdim = True)), dim = 1)
    var1 = torch.sum(torch.square(simulated_data - torch.mean(simulated_data, dim = 1, keepdim = True)), dim = 1)
    corr = cov / torch.sqrt(var0 * var1)

    return corr.detach().numpy()

def mse_score(optimizer, observed_data, estimates):
    '''
    Compute MSE score between observed data and data simulated with the optimizer's estimates.

    Parameters:
    ----------
    optimizer: ACTOptimizer
        Optimizer which was used to obtain estimates.

    observed_data: torch.tensor(shape = num_current_injections, 1024)
        Target data to compute correlation with. Most probably, it is the data which was used to train the optimizer.

    esimates: ndarray(shape = num_current_injections, num_parameters)
        Estimates obtained with the optimizer.

    Returns:
    ----------
    mse: ndarray(shape = (num_current_injections, ))
        MSE values.
    '''

    simulated_data = optimizer.simulate(estimates, optimizer.parameters)
    return torch.mean(torch.square(observed_data - simulated_data), dim = 1).detach().numpy()

def plot_score(optimizer, observed_data, estimates):
    '''
    Compute correlation score between observed data and data simulated with the optimizer's estimates.

    Parameters:
    ----------
    optimizer: ACTOptimizer
        Optimizer which was used to obtain estimates.

    observed_data: torch.tensor(shape = num_current_injections, 1024)
        Target data to compute correlation with. Most probably, it is the data which was used to train the optimizer.

    esimates: ndarray(shape = num_current_injections, num_parameters)
        Estimates obtained with the optimizer.

    Returns:
    ----------
    corr: ndarray(shape = (num_current_injections, ))
        Sample correlations.
    '''

    simulated_data = optimizer.simulate(estimates, optimizer.parameters).T
    observed_data = observed_data.T

    fig, ax = plt.subplots(nrows = 1, ncols = observed_data.shape[1], figsize = (30, 4))
    for i in range(observed_data.shape[1]):
        ax[i].plot(observed_data[:, i], label = "Observed")
        ax[i].plot(simulated_data[:, i], label = "Simulated")
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Voltage")
        ax[i].set_title(f"I: {optimizer.current_injections[i]}, esitmate: {round(float(estimates[i]), 4)}")
        ax[i].legend()
