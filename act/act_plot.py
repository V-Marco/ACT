import os
import numpy as np
from matplotlib import pyplot as plt, cm
import plotly.graph_objects as go

from act.data_processing import *
from act.metrics import *

# A collection of plotting functions that can be used to assess the quality of automatic tuning.

def create_overlapped_v_plot(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, module_foldername: str, title: str, filename:str) -> None:
    '''
    A helper function for plot_v_comparison to plot the Target cell voltage trace vs the Predicted 
    cell voltage trace.
    Parameters:
    -----------
    x: np.ndarray
        (ms)
    
    y1: np.ndarray
        (mV)
    
    y2: np.ndarray
        (mV)
        
    module_foldername: str
    
    tile: str
    
    filename: str
    
    Returns:
    -----------
    None
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label='Target', c = 'blue')
    plt.plot(x, y2, label='Prediction', ls = '--', c = 'red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    plt.legend()
    plt.savefig(module_foldername + "/results/" + filename)
    plt.close()
    

def plot_v_comparison(module_foldername: str,predicted_g_data_file: str, current_injections: list, dt: float) -> None:
    '''
    Plots the Target cell voltage trace vs the Predicted cell voltage trace. (Line Plot)
    Parameters:
    -----------
    module_foldername: str
    
    predicted_g_data_file: str
        Path to prediction data
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    dt: float
        Timestep
        
    Returns:
    ----------
    None
    '''
    results_folder = module_foldername + "/results/"
    os.makedirs(results_folder, exist_ok=True)
    
    amps = [current_injection.amp for current_injection in current_injections]

    target_traces = np.load(module_foldername + "/target/combined_out.npy")
    target_v = target_traces[:,:,0]

    selected_traces = np.load(predicted_g_data_file)
    selected_v = selected_traces[:,:,0]

    time = np.linspace(0, len(target_v[0]), len(target_v[0])) * dt

    for i in range(len(selected_v)):
        create_overlapped_v_plot(time, target_v[i], selected_v[i], module_foldername, f"V Trace Comparison: {amps} nA", f"V_trace_{amps[i]}nA.png")


def plot_fi_comparison(module_foldername: str, current_injections: list) -> None:
    '''
    Plots the spike frequencies at the preselected current injection intensities
    for both the Target cell and the Predicted cell (Dot Plot)
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
    
    Returns:
    -----------
    None
    '''
    results_folder = f"{module_foldername}/results/"
    
    os.makedirs(results_folder, exist_ok=True)
    dataset = np.load(f"{results_folder}/frequency_data.npy")
    predicted_fi = dataset[0,:]
    target_fi = dataset[1,:]
    
    amps = [current_injection.amp for current_injection in current_injections]
    
    plt.figure(figsize=(8, 6))
    plt.plot(amps, target_fi, 'o', label='Target FI')
    plt.plot(amps, predicted_fi, 'o', label='Prediction FI')
    plt.xlabel('Current Injection Intensity (nA)')
    plt.ylabel('Frequency (Hz)')
    plt.title("FI Curve Comparison")
    plt.legend()
    plt.savefig(results_folder + "FI_Curve_Comparison.png")
    plt.close()
    
    
def plot_training_v_mae_surface(module_foldername: str, current_injections: list, delay: float, dt: float, index1: int, index2: int, g_names: list, results_filename: str) -> None:
    '''
    Generates an interactive HTML file of a 3D surface plot.
    The 3D surface consists of the following:
    - Axis 1: the conductance values used in one of channels [selected by "index1"]
    - Axis 2: the conductance values used in another channel [selected by "index2"]
    - Axis 3: the MAE of the raw voltage traces of the Target cell and Predicted Cell 
        (averaging varying current injection intensities)
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    delay: float
        Current injection delay from start of simulation
    
    dt: float
        Timestep
    
    index1: int
        X-axis selection for MAE comparison (options are the list of channels)
        
    index2: int
        Y-axis selection for MAE comparison (options are the list of channels)
        
    g_names: list[str]
        Conductance names
    
    results_filename: str
        Filename for the output graph
    
    Returns:
    -----------
    None
    '''
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    conductance_values = np.unique(train_g, axis=0)
    
    v_sample_sets = []
    for g in conductance_values:

        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        index_where_inj_occurs = int(delay / dt + 1)
        
        amps = [current_injection.amp for current_injection in current_injections]
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        V_subset = train_V[ordered_indices]
        
        v_sample_sets.append(V_subset)

    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], mae_score(target_V, v_sample_sets[idx])))
    maes = np.array(maes)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    Z = np.empty(X.shape)
    Z[:] = np.nan

    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    
    ax.view_init(elev=30, azim=60)

    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    ax.set_zlabel("Voltage MAE")

    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Voltage MAE')

    plt.title('Voltage Trace MAE Surface Plot')
    fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

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

    fig2.write_html(results_filename)
    plt.show()


def plot_training_v_mae_contour_plot(module_foldername: str, current_injections: list, delay: float, dt: float, index1: int, index2: int, g_names: list, num_levels: int = 100, results_filename: str = None) -> None:
    '''
    Generates a 2D contour plot version of plot_training_v_mae_surface (See directly above)
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    delay: float
        Current injection delay from start of simulation
    
    dt: float
        Timestep
    
    index1: int
        X-axis selection for MAE comparison (options are the list of channels)
        
    index2: int
        Y-axis selection for MAE comparison (options are the list of channels)
        
    g_names: list[str]
        Conductance names
        
    num_levels: int, default = 100
        Max Resolution for the contour plot (100 is sufficiently higher than # of slices)
    
    results_filename: str, default = None
        Filename for the output graph
    
    Returns:
    -----------
    None
    '''
    amps = [current_injection.amp for current_injection in current_injections]
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    conductance_values = np.unique(train_g, axis=0)
    
    v_sample_sets = []
    for g in conductance_values:
        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        V_subset = train_V[ordered_indices]
        
        v_sample_sets.append(V_subset)

    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], mae_score(target_V, v_sample_sets[idx])))
    maes = np.array(maes)
    
    #x = np.linspace(0, 10000,10000)
    #for idx, g in enumerate(conductance_values):
        #for i in range(len(amps)):
            #print(f"Conductance values: {g}")
            #print(maes[idx][2])
            #plt.plot(target_V[i], label='Target', c = 'blue')
            #plt.plot(v_sample_sets[idx][i], label=f"Conductance: {g}", ls = '--', c = 'red')
            #plt.fill_between(x, target_V[i], v_sample_sets[idx][i], color="blue", alpha=0.3)
            #plt.show()
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest MAE values ({g_name1}, {g_name2}, V MAE): ")
    print(sorted_maes[:6])

    g1_bar = np.unique(maes[:, 0])
    g2_bar = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1_bar, g2_bar)

    Z = np.empty(X.shape)
    Z[:] = np.nan

    for i in range(len(maes)):
        x_idx = np.where(g1_bar == maes[i, 0])[0][0]
        y_idx = np.where(g2_bar == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    cbar = fig.colorbar(contour)
    cbar.set_label('Voltage MAE')
    
    plt.title('Voltage MAE Contour Plot')
    
    if(not results_filename):
        results_filename = f"{module_foldername}/results/Voltage_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    plt.show()
    
    
def plot_training_feature_mae_contour_plot(module_foldername: str, current_injections: list, delay: float, dt: float, index1: int, index2: int, g_names: list, train_features: list, threshold: float = 0, first_n_spikes: int = 20, num_levels: int = 100, results_filename: str = None) -> None:
    '''
    Generates a 2D contour plot
    The contour consists of the following:
    - Axis 1: the conductance values used in one of channels [selected by "index1"]
    - Axis 2: the conductance values used in another channel [selected by "index2"]
    - Axis 3: the MAE of the pre-selected features of the Target cell and Predicted Cell 
        (averaging varying current injection intensities)
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    delay: float
        Current injection delay from start of simulation
    
    dt: float
        Timestep
    
    index1: int
        X-axis selection for MAE comparison (options are the list of channels)
        
    index2: int
        Y-axis selection for MAE comparison (options are the list of channels)
        
    g_names: list[str]
        Conductance names
        
    train_features: list[str]
        List of feature names that are extracted from the simulation data
        
    threshold: float, default = 0
        Spiking threshold
        
    first_n_spikes: int, default = 20
        First number of spikes considered for calculations
        
    num_levels: int, default = 100
        Max Resolution for the contour plot (100 is sufficiently higher than # of slices)
    
    results_filename: str, default = None
        Filename for the output graph
    
    Returns:
    -----------
    None
    '''
    amps = [current_injection.amp for current_injection in current_injections]
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    target_I = target_dataset[:,:,1]
    target_lto_hto = target_dataset[:,1,2]
    
    target_V_features, _ = extract_features(train_features=train_features, V=target_V,I=target_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt, lto_hto=target_lto_hto, current_inj_combos=current_injections)
    #print(f"target_v_features: {len(target_V_features)}: {target_V_features}")
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]
    train_lto_hto = train_dataset[:,1,3]
    
    #print(f"train_g: {len(train_g)} - {train_g}")

    conductance_values = np.unique(train_g, axis=0)
    
    #print(f"conductance_values: {conductance_values}")
    
    v_sample_feature_sets = []
    #ordered_V = []
    for g in conductance_values:
        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
        
        ordered_indices = np.array(ordered_indices)
        
        V_subset = train_V[ordered_indices]
        I_subset = train_I[ordered_indices]
        lto_hto_subset = train_lto_hto[ordered_indices]
        #print(f"V_subset: {V_subset}")
        #ordered_V.append(V_subset)
        
        V_subset_features, _ = extract_features(train_features=train_features, V=V_subset, I=I_subset, threshold=threshold, num_spikes=first_n_spikes, dt=dt, lto_hto=lto_hto_subset, current_inj_combos=current_injections)
        
        v_sample_feature_sets.append(V_subset_features)
    #print(f"vsample_features: {len(v_sample_feature_sets)}: {v_sample_feature_sets}")
    maes = []
    
    for idx, g in enumerate(conductance_values):
        i_inj_mae = []
        for i in range(len(target_V_features)):
            #print(f"v_samplefeature_set: {v_sample_feature_sets[idx][i]}")
            mae = mae_score(target_V_features[i], v_sample_feature_sets[idx][i])
            i_inj_mae.append(mae)

        maes.append((g[index1], g[index2], np.mean(i_inj_mae)))
    maes = np.array(maes)
    
    #print(f"ordered_V: {ordered_V}")
    # plot the points here
    #for idx, g in enumerate(conductance_values):
        #for i in range(len(amps)):
            #print(f"Conductance values: {g}")
            #print(maes[idx][2])
            #plt.plot(target_V[i], label='Target', c = 'blue')
            #plt.plot(ordered_V[idx][i], label=f"Conductance: {g}", ls = '--', c = 'red')
            #plt.show()

    #print(f"maes: {maes}")
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest MAE values ({g_name1}, {g_name2}, Summary Stats MAE): ")
    print(sorted_maes[:6])

    g1_bar = np.unique(maes[:, 0])
    g2_bar = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1_bar, g2_bar)

    Z = np.empty(X.shape)
    Z[:] = np.nan

    for i in range(len(maes)):
        x_idx = np.where(g1_bar == maes[i, 0])[0][0]
        y_idx = np.where(g2_bar == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    cbar = fig.colorbar(contour)
    cbar.set_label('Summary Stats MAE')
    
    plt.title('Summary Stats MAE Contour Plot')
    
    if(not results_filename):
        results_filename = f"{module_foldername}/results/SummaryStats_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    plt.show()
    
    
def plot_training_fi_mae_surface(module_foldername: str, current_injections: list, delay: float, dt: float, index1: int, index2: int, g_names: list, results_filename: str, spike_threshold: float = 0) -> None:
    '''
    Generates an interactive HTML file of a 3D surface plot.
    The 3D surface consists of the following:
    - Axis 1: the conductance values used in one of channels [selected by "index1"]
    - Axis 2: the conductance values used in another channel [selected by "index2"]
    - Axis 3: the MAE of the FI curves of the Target cell and Predicted Cell 
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    delay: float
        Current injection delay from start of simulation
    
    dt: float
        Timestep
    
    index1: int
        X-axis selection for MAE comparison (options are the list of channels)
        
    index2: int
        Y-axis selection for MAE comparison (options are the list of channels)
        
    g_names: list[str]
        Conductance names
    
    results_filename: str, default = None
        Filename for the output graph
        
    spike_threshold: float, default = 0
        Spiking threshold
    
    Returns:
    -----------
    None
    '''
    amps = [current_injection.amp for current_injection in current_injections]
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)
    
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    target_frequencies = get_fi_curve(target_V, spike_threshold=spike_threshold, CI_list=amps).flatten()
    
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    conductance_values = np.unique(train_g, axis=0)
    
    fi_curves = []
    for g in conductance_values:
        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        V_subset = train_V[ordered_indices]
        
        fi_curve = get_fi_curve(V_subset, spike_threshold=spike_threshold, CI_list=amps)
        
        fi_curves.append(fi_curve)

    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], mae_score(target_frequencies, fi_curves[idx])))
    maes = np.array(maes)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    Z = np.empty(X.shape)
    Z[:] = np.nan

    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    
    ax.view_init(elev=30, azim=60)

    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    ax.set_zlabel("MAE")

    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('MAE')

    plt.title('FI-curve MAE Surface Plot')
    fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

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

    fig2.write_html(results_filename)
    plt.show()
    
    
def plot_training_fi_mae_contour_plot(module_foldername: str, current_injections: list, delay: float, dt: float, index1: int, index2: int, g_names: list, spike_threshold: float = 0, num_levels: int = 100, results_filename: str = None) -> None:
    '''
    Generates a 2D contour plot version of plot_training_fi_mae_surface (See directly above)
    Parameters:
    -----------
    module_foldername: str
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        List of Current Injection classes
        
    delay: float
        Current injection delay from start of simulation
    
    dt: float
        Timestep
    
    index1: int
        X-axis selection for MAE comparison (options are the list of channels)
        
    index2: int
        Y-axis selection for MAE comparison (options are the list of channels)
        
    g_names: list[str]
        Conductance names
        
    spike_threshold: float, default = 0
        Spiking threshold
    
    num_levels: int, default = 100
        Max Resolution for the contour plot (100 is sufficiently higher than # of slices)
    
    results_filename: str, default = None
        Filename for the output graph
    
    
    Returns:
    -----------
    None
    '''
    amps = [current_injection.amp for current_injection in current_injections]
    
    g_name1 = g_names[index1]
    g_name2 = g_names[index2]
    length_g = len(g_names)

    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    
    target_frequencies = get_fi_curve(target_V, spike_threshold=spike_threshold, CI_list=current_injections).flatten()
    
    train_dataset = np.load(f"{module_foldername}/train/combined_out.npy")
    train_V = train_dataset[:,:,0]
    train_I = train_dataset[:,:,1]
    train_g = train_dataset[:,:length_g,2]

    conductance_values = np.unique(train_g, axis=0)
    
    fi_curves = []
    for g in conductance_values:
        conductance_idx = np.where((train_g == g).all(axis=1))[0]
        
        index_where_inj_occurs = int(delay / dt + 1)
        
        ordered_indices = []
        for amp in amps:
            ordered_idx = np.where(train_I[conductance_idx,index_where_inj_occurs] == amp)[0]
            ordered_indices.append(conductance_idx[ordered_idx][0])
            
        ordered_indices = np.array(ordered_indices)
            
        V_subset = train_V[ordered_indices]
        
        fi_curve = get_fi_curve(V_subset, spike_threshold=spike_threshold, CI_list=current_injections)
        
        fi_curves.append(fi_curve)

    maes = []
    
    for idx, g in enumerate(conductance_values):
        maes.append((g[index1], g[index2], mae_score(target_frequencies, fi_curves[idx])))
    maes = np.array(maes)
    
    #print(f"maes: {maes}")
    #for idx, g in enumerate(conductance_values):
        #print(f"Conductance values: {g}")
        #print(maes[idx][2])
        #plt.plot(target_frequencies, label='Target', c = 'blue')
        #plt.plot(fi_curves[idx], label=f"Conductance: {g}", ls = '--', c = 'red')
        #plt.show()
    
    sorted_maes = maes[maes[:, 2].argsort()]
    print(f"Smallest FI MAE values ({g_name1}, {g_name2}, FI MAE): ")
    print(sorted_maes[:6])

    g1 = np.unique(maes[:, 0])
    g2 = np.unique(maes[:, 1])
    X, Y = np.meshgrid(g1, g2)

    Z = np.empty(X.shape)
    Z[:] = np.nan

    for i in range(len(maes)):
        x_idx = np.where(g1 == maes[i, 0])[0][0]
        y_idx = np.where(g2 == maes[i, 1])[0][0]
        Z[y_idx, x_idx] = maes[i, 2]

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=num_levels, cmap='viridis')
    ax.set_xlabel(g_name1)
    ax.set_ylabel(g_name2)
    
    cbar = fig.colorbar(contour)
    cbar.set_label('FI MAE')
    
    plt.title('FI-Curve MAE Contour Plot')
    
    if(not results_filename):
        results_filename = f"{module_foldername}/results/FI_MAE_Contour_Plot.png"
    
    plt.savefig(results_filename)
    
    plt.show()