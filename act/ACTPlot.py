import os
import numpy as np
from matplotlib import pyplot as plt

def create_overlapped_v_plot(self, x, y1, y2, module_foldername, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label='Target')
    plt.plot(x, y2, label='Prediction')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    plt.legend()
    plt.savefig(module_foldername + "results/" + filename)
    plt.close()  # Close the figure to free up memory

def plot_v_comparison(self, g_final_idx, module_foldername, amps):
    results_folder = module_foldername + "results/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # load target traces
    target_traces = np.load(module_foldername + "target/combined_out.npy")
    target_v = target_traces[:,:,0]

    # load final prediction voltage traces
    selected_traces = np.load(module_foldername + "prediction_eval" + str(g_final_idx) + "/combined_out.npy")
    selected_v = selected_traces[:,:,0]

    time = np.linspace(0, len(target_v[0]), len(target_v[0]))

    # Plot all pairs of traces
    for i in range(len(selected_v)):
        self.create_overlapped_v_plot(time, target_v[i], selected_v[i], module_foldername, f"V Trace Comparison: {amps} nA", f"V_trace_{amps[i]}nA.png")


def plot_fi_comparison(self, predicted_fi, target_fi, amps, module_foldername):
    # Plot the FI curves of predicted and target
    plt.figure(figsize=(8, 6))
    plt.plot(amps, target_fi, label='Target FI')
    plt.plot(amps, predicted_fi, label='Prediction FI')
    plt.xlabel('Current Injection Intensity (nA)')
    plt.ylabel('Frequency (Hz)')
    plt.title("FI Curve Comparison")
    plt.legend()
    plt.savefig(module_foldername + "results/" + "FI_Curve_Comparison.png")
    plt.close()