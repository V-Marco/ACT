import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from act.DataProcessor import DataProcessor
from act.Metrics import Metrics

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
    
def plot_training_mae_surface_spiker_cell(target_data_filepath, training_data_filepath, delay, dt):
    # load target data
    target_dataset = np.load(target_data_filepath)
    
    target_V = target_dataset[:,:,0]
    target_I = target_dataset[:,:,1]
    
    # load training data
    train_dataset = np.load(training_data_filepath)
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:,2]
    
    index_where_inj_occurs = int(delay / dt + 1)
    target_I_sample = target_I[0]
    #print(target_I_sample[index_where_inj_occurs])
    
    metrics = Metrics()
    maes = []

    # Calculate MAE for matching current injection traces
    for target_idx, target_V_sample in enumerate(target_V):
        target_I_sample = target_I[target_idx]
        

        matching_training_indices = np.where(train_I[:,index_where_inj_occurs] == target_I_sample[index_where_inj_occurs])[0]
        print(f"found indices: {len(matching_training_indices)}")
        if len(matching_training_indices) == 0:
            continue  

        for train_idx in matching_training_indices:
            train_V_sample = train_V[train_idx]

            mae = metrics.mae_score(target_V_sample, train_V_sample)
            conductance_values = train_g[train_idx]
            
            #print((conductance_values[0], conductance_values[1], mae))
            maes.append((conductance_values[0], conductance_values[1], mae))

    # Convert results to numpy array for plotting
    maes = np.array(maes)
    #print(maes.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    #print(maes[:, 0])
    scatter = ax.scatter(maes[:, 0], maes[:, 1], maes[:, 2], c=maes[:, 2], cmap='viridis', marker='o')

    ax.set_xlabel("gNA_bar")
    ax.set_ylabel("gK_bar")
    
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)  # Add a color bar to help interpret the colors
    cbar.set_label('MAE')

    plt.title('MAE Plot')
    
    ax.view_init(elev=20, azim=135)
    
    fig.savefig('MAE_Surface.png')
    plt.show()
    
    
    
    
    