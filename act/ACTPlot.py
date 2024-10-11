import os
import numpy as np
from matplotlib import pyplot as plt
from act.DataProcessor import DataProcessor
from act.Metrics import Metrics
from matplotlib import cm
import plotly.graph_objects as go

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

def plot_v_comparison(predicted_g_data_file, module_foldername, amps, dt):
    results_folder = module_foldername + "/results/"
    os.makedirs(results_folder, exist_ok=True)

    # load target traces
    target_traces = np.load(module_foldername + "/target/combined_out.npy")
    target_v = target_traces[:,:,0]

    # load final prediction voltage traces
    selected_traces = np.load(predicted_g_data_file)
    selected_v = selected_traces[:,:,0]

    time = np.linspace(0, len(target_v[0]), len(target_v[0])) * dt

    # Plot all pairs of traces
    for i in range(len(selected_v)):
        create_overlapped_v_plot(time, target_v[i], selected_v[i], module_foldername, f"V Trace Comparison: {amps} nA", f"V_trace_{amps[i]}nA.png")


def plot_fi_comparison(module_foldername, amps):
    # Plot the FI curves of predicted and target
    results_folder = f"{module_foldername}/results/"
    
    os.makedirs(results_folder, exist_ok=True)
    dataset = np.load(f"{results_folder}/frequency_data.npy")
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
    
    
def plot_training_v_mae_surface(module_foldername, amps, inj_dur, delay, dt, index1, index2, g_names, results_filename):
    dp = DataProcessor()
    metrics = Metrics()
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    # load target data
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    # load training data
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    # Get a list of unique conductance sets (should be n copies consolidated where n is len(amps))
    conductance_values = np.unique(train_g, axis=0)
    
    # Go through unique conductance sets and find the indices where there are copies
    v_sample_sets = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        # Now we want to ensure the sets are ordered properly.
        # Find where I value == amps
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        # Get a set of ordered voltage traces with the matching conductances. Get FI curve from this
        V_subset = train_V[ordered_indices]
        
        v_sample_sets.append(V_subset)

    # Get MAE of v for each I injection
    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], metrics.mae_score(target_V, v_sample_sets[idx])))
    maes = np.array(maes)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Create meshgrid for contour  map
    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    # Initialize a Z matrix with NaNs
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # Fill Z with MAE values
    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    
    ax.view_init(elev=30, azim=60)

    # Customize the z axis.
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    ax.set_zlabel("Voltage MAE")

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Voltage MAE')

    plt.title('Voltage Trace MAE Surface Plot')
    # Save an interactive figure
    fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

    # Customize layout
    fig2.update_layout(
        title='Voltage Trace MAE Surface Plot',
        scene=dict(
            xaxis_title=g_name1,
            yaxis_title=g_name2,
            zaxis_title='Voltage MAE'
        ),
        coloraxis_colorbar=dict(
            title='Voltage MAE'
        )
    )

    # Save the plot as an interactive HTML file
    fig2.write_html(results_filename)
    plt.show()
    

def plot_training_v_mae_contour_plot(module_foldername, amps, delay, dt, index1, index2, g_names, num_levels=100, results_filename=None):
    dp = DataProcessor()
    metrics = Metrics()
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    # load target data
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    # load training data
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    # Get a list of unique conductance sets (should be n copies consolidated where n is len(amps))
    conductance_values = np.unique(train_g, axis=0)
    
    # Go through unique conductance sets and find the indices where there are copies
    v_sample_sets = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        # Now we want to ensure the sets are ordered properly.
        # Find where I value == amps
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        # Get a set of ordered voltage traces with the matching conductances. Get FI curve from this
        V_subset = train_V[ordered_indices]
        
        v_sample_sets.append(V_subset)

    # Get MAE of v for each I injection
    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], metrics.mae_score(target_V, v_sample_sets[idx])))
    maes = np.array(maes)
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest MAE values ({g_name1}, {g_name2}, V MAE): ")
    print(sorted_maes[:6])

    # Create meshgrid for contour  map
    g1_bar = np.unique(maes[:, 0])
    g2_bar = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1_bar, g2_bar)

    # Initialize a Z matrix with NaNs
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # Fill Z with MAE values
    for i in range(len(maes)):
        x_idx = np.where(g1_bar == maes[i, 0])[0][0]
        y_idx = np.where(g2_bar == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(contour)
    cbar.set_label('Voltage MAE')
    
    plt.title('Voltage MAE Contour Plot')
    
    # Save the plot as a PNG file
    if(not results_filename):
        results_filename = f"{module_foldername}/results/Voltage_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    # Show the plot
    plt.show()
    
def plot_training_feature_mae_contour_plot(module_foldername, amps, delay, dt, index1, index2, g_names, train_features, threshold=0, first_n_spikes=20, num_levels=100, results_filename=None):
    dp = DataProcessor()
    metrics = Metrics()
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    # load target data
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    target_I = target_dataset[:,:,1]
    
    target_V_features, _ = dp.extract_features(train_features=train_features, V=target_V,I=target_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
    
    # load training data
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]
    #print(train_g.shape)

    # Get a list of unique conductance sets (should be n copies consolidated where n is len(amps))
    conductance_values = np.unique(train_g, axis=0)
    #print(conductance_values.shape)
    
    # Go through unique conductance sets and find the indices where there are copies
    v_sample_feature_sets = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        # Now we want to ensure the sets are ordered properly.
        # Find where I value == amps
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        # Get a set of ordered voltage traces with the matching conductances. Get features from these
        V_subset = train_V[ordered_indices]
        I_subset = train_I[ordered_indices]
        
        V_subset_features, _ = dp.extract_features(train_features=train_features, V=V_subset, I=I_subset, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
        
        v_sample_feature_sets.append(V_subset_features)
    #print(len(v_sample_feature_sets))
    # Get MAE of v for each I injection
    maes = []
    
    for idx, g in enumerate(conductance_values):
        i_inj_mae = []
        #print(len(target_V_features))
        for i in range(len(target_V_features)):  #for each current injection
            # get feature mae for each current injection
            
            #print(f"tar:{target_V_features[i]}")
            #print(f"sub:{v_sample_feature_sets[idx][i]}")
            mae = metrics.mae_score(target_V_features[i], v_sample_feature_sets[idx][i])
            #print(mae)
            i_inj_mae.append(mae)

        maes.append((g[index1], g[index2], np.mean(i_inj_mae)))
    maes = np.array(maes)
    #print(maes.shape)
    #print(maes)
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest MAE values ({g_name1}, {g_name2}, Summary Stats MAE): ")
    print(sorted_maes[:6])

    # Create meshgrid for contour  map
    g1_bar = np.unique(maes[:, 0])
    g2_bar = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1_bar, g2_bar)

    # Initialize a Z matrix with NaNs
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # Fill Z with MAE values
    for i in range(len(maes)):
        x_idx = np.where(g1_bar == maes[i, 0])[0][0]
        y_idx = np.where(g2_bar == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(contour)
    cbar.set_label('Summary Stats MAE')
    
    plt.title('Summary Stats MAE Contour Plot')
    
    # Save the plot as a PNG file
    if(not results_filename):
        results_filename = f"{module_foldername}/results/SummaryStats_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    # Show the plot
    plt.show()
    
def plot_training_fi_mae_surface_spiker_cell(module_foldername, amps, inj_dur, delay, dt, index1, index2, g_names, results_filename):
    dp = DataProcessor()
    metrics = Metrics()
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    # load target data
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    # Get FI curve of target data
    target_frequencies = dp.get_fi_curve(target_V, amps, inj_dur=inj_dur).flatten()
    
    # load training data
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    # Get FI curves of training data (conductance grid). This will be tricky because parallelization shuffles the grid
    # Get a list of unique conductance sets (should be n copies consolidated where n is len(amps))
    conductance_values = np.unique(train_g, axis=0)
    
    # Go through unique conductance sets and find the indices where there are copies
    fi_curves = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        # Now we want to ensure the sets are ordered properly.
        # Find where I value == amps
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        # Get a set of ordered voltage traces with the matching conductances. Get FI curve from this
        V_subset = train_V[ordered_indices]
        
        fi_curve = dp.get_fi_curve(V_subset, amps, inj_dur=inj_dur)
        
        fi_curves.append(fi_curve)

    # Get MAE of FI curve for each conductance set
    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], metrics.mae_score(target_frequencies, fi_curves[idx])))
    maes = np.array(maes)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Create meshgrid for contour  map
    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    # Initialize a Z matrix with NaNs
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # Fill Z with MAE values
    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    
    ax.view_init(elev=30, azim=60)

    # Customize the z axis.
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    ax.set_zlabel("MAE")

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('MAE')

    plt.title('FI-curve MAE Surface Plot')
    # Save an interactive figure
    fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

    # Customize layout
    fig2.update_layout(
        title='FI-curve MAE Surface Plot',
        scene=dict(
            xaxis_title=g_name1,
            yaxis_title=g_name2,
            zaxis_title='FI MAE'
        ),
        coloraxis_colorbar=dict(
            title='FI MAE'
        )
    )

    # Save the plot as an interactive HTML file
    fig2.write_html(results_filename)
    plt.show()
    
    
def plot_training_fi_mae_contour_plot(module_foldername, amps, inj_dur, delay, dt, index1, index2, g_names, num_levels=100, results_filename=None):
    dp = DataProcessor()
    metrics = Metrics()
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    # load target data
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    # Get FI curve of target data
    target_frequencies = dp.get_fi_curve(target_V, amps, inj_dur=inj_dur).flatten()
    
    # load training data
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    # Get FI curves of training data (conductance grid). This will be tricky because parallelization shuffles the grid
    # Get a list of unique conductance sets (should be n copies consolidated where n is len(amps))
    conductance_values = np.unique(train_g, axis=0)
    
    # Go through unique conductance sets and find the indices where there are copies
    fi_curves = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        # Now we want to ensure the sets are ordered properly.
        # Find where I value == amps
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        # Get a set of ordered voltage traces with the matching conductances. Get FI curve from this
        V_subset = train_V[ordered_indices]
        
        fi_curve = dp.get_fi_curve(V_subset, amps, inj_dur=inj_dur)
        
        fi_curves.append(fi_curve)

    # Get MAE of FI curve for each conductance set
    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], metrics.mae_score(target_frequencies, fi_curves[idx])))
    maes = np.array(maes)
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest FI MAE values ({g_name1}, {g_name2}, FI MAE): ")
    print(sorted_maes[:6])

    # Create meshgrid for contour  map
    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    # Initialize a Z matrix with NaNs
    Z = np.empty(X.shape)
    Z[:] = np.nan

    # Fill Z with MAE values
    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(contour)
    cbar.set_label('FI MAE')
    
    plt.title('FI-Curve MAE Contour Plot')
    
    # Save the plot as a PNG file
    if(not results_filename):
        results_filename = f"{module_foldername}/results/FI_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    # Show the plot
    plt.show()