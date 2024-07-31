import os
import numpy as np
from matplotlib import pyplot as plt

def create_overlapped_v_plot(x, y1, y2, module_foldername, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label='Target')
    plt.plot(x, y2, label='Prediction')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    plt.legend()
    plt.savefig(module_foldername + "/results/" + filename)
    plt.close()  # Close the figure to free up memory

def plot_v_comparison(predicted_g_data_file, target_data_folder, module_foldername, amps):
    results_folder = module_foldername + "/results/"
    os.makedirs(results_folder, exist_ok=True)

    # load target traces
    target_traces = np.load(target_data_folder + "/combined_out.npy")
    target_v = target_traces[:,:,0]

    # load final prediction voltage traces
    selected_traces = np.load(predicted_g_data_file)
    selected_v = selected_traces[:,:,0]

    time = np.linspace(0, len(target_v[0]), len(target_v[0]))

    # Plot all pairs of traces
    for i in range(len(selected_v)):
        create_overlapped_v_plot(time, target_v[i], selected_v[i], module_foldername, f"V Trace Comparison: {amps} nA", f"V_trace_{amps[i]}nA.png")


def plot_fi_comparison(fi_data_filepath, amps):
    # Plot the FI curves of predicted and target
    results_folder = os.path.dirname(fi_data_filepath) + "/"
    
    os.makedirs(results_folder, exist_ok=True)
    dataset = np.load(fi_data_filepath)
    predicted_fi = dataset[0,:]
    target_fi = dataset[1,:]
    
    plt.figure(figsize=(8, 6))
    plt.plot(amps, target_fi, 'o', label='Target FI')
    plt.plot(amps, predicted_fi, 'o', label='Prediction FI')
    plt.xlabel('Current Injection Intensity (nA)')
    plt.ylabel('Frequency (Hz)')
    plt.title("FI Curve Comparison")
    plt.legend()
    plt.savefig(results_folder + "FI_Curve_Comparison.png")
    plt.close()