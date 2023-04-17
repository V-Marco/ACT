import torch
import matplotlib.pyplot as plt

def correlation_score(optimizer, observed_data, estimates):
    simulated_data = optimizer.simulate(estimates, optimizer.parameters)

    cov = (observed_data - torch.mean(observed_data, dim = 1, keepdim = True)) * (simulated_data - torch.mean(simulated_data, dim = 1, keepdim = True))
    cov = torch.sum(cov, dim = 1)

    var0 = torch.sum(torch.square(observed_data - torch.mean(observed_data, dim = 1, keepdim = True)), dim = 1)
    var1 = torch.sum(torch.square(simulated_data - torch.mean(simulated_data, dim = 1, keepdim = True)), dim = 1)

    return cov / torch.sqrt(var0 * var1)

def mse_score(optimizer, observed_data, estimates):
    simulated_data = optimizer.simulate(estimates, optimizer.parameters)
    return torch.mean(torch.square(observed_data - simulated_data), dim = 1)

def plot_score(optimizer, observed_data, estimates):
    simulated_data = optimizer.simulate(estimates, optimizer.parameters).reshape((1024, -1))
    observed_data = observed_data.reshape((1024, -1))

    fig, ax = plt.subplots(nrows = 1, ncols = observed_data.shape[1], figsize = (30, 4))
    for i in range(observed_data.shape[1]):
        ax[i].plot(observed_data[:, i], label = "Observed")
        ax[i].plot(simulated_data[:, i], label = "Simulated")
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Voltage")
        ax[i].set_title(f"I: {optimizer.current_injections[i]}")
