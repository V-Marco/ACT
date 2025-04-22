import os
import json
import glob
import numpy as np
import pandas as pd
from itertools import product
import pickle

#---------------------------------------------
# Feature Extraction
#---------------------------------------------
def find_events(v: np.ndarray, spike_threshold: float = -20) -> list:
    """
    Counts the number of spikes in a voltage trace. Returns a list of event indices
    Parameters:
    -----------
    v: np.ndarray of shape = (T,)
        A single voltage trace.
    
    spike_threshold: float, default = -20
        Threshold for spike detection (mV).
        
    Returns:
    -----------
    event_indices: list[float]
        List of voltage trace indices where spikes occur.
    """
    # Find the indices where the voltage is above the threshold
    above_threshold_indices = np.where(v[:-1] > spike_threshold)[0]

    # Find the indices where the slope changes from positive to negative
    slope = np.diff(v)
    positive_to_negative_indices = np.where((slope[:-1] > 0) & (slope[1:] < 0))[0]

    # Find the intersection of the two sets of indices
    event_indices = np.intersect1d(above_threshold_indices, positive_to_negative_indices)

    return event_indices

def VI_summary_features(V: np.ndarray, I: np.ndarray = None, spike_threshold: int = -20, max_n_spikes: int = 20) -> pd.DataFrame:
    '''
    Compute voltage and current summary features.

    Parameters:
    ----------
    V: np.ndarray of shape (n_trials, T)
        Voltage traces.
    
    I: np.ndarray of shape (n_trials, T), default = None
        Current traces.
    
    spike_threshold: int, default = -20
        Threshold for spike detection (mV).
    
    max_n_spikes: int, default = 20
        Maximum number of spike times to save.

    Returns:
    ----------
    stat_df: pd.DataFrame
        Dataframe with extracted summary features.
    '''
    stat_df = []
    for trial_idx in range(len(V)):
        row = pd.DataFrame(index = [0])

        # Extract voltage features
        v_trial = V[trial_idx]

        # Save overall voltage stats
        row["mean_v"] = np.nanmean(v_trial)
        row("std_v") = np.nanstd(v_trial)

        # Extract # of spikes and # of troughs and save the number
        spike_idxs = find_events(v_trial, spike_threshold)
        trough_idxs = find_events(-v_trial, -spike_threshold) #TODO: valid trough check -- do we need it?

        for events, event_type_name in zip([spike_idxs, trough_idxs], ["spike", "trough"]):
            # Save the number of events
            row[f"n_{event_type_name}s"] = len(events)
        
            # Save first max_n_spikes event times; pad if less events than max_n_spikes
            event_idxs_corrected = np.ones(max_n_spikes) * np.nan
            event_idxs_corrected[:len(events)] = events
            for event_idx in range(len(max_n_spikes)):
                row[f"spike_time_{event_idx}"] = event_idxs_corrected[event_idx]

             # Save event stats
            row[f'min_{event_type_name}_amp'] = np.min(v_trial[events])
            row[f'max_{event_type_name}_amp'] = np.max(v_trial[events])
            row[f'mean_{event_type_name}_amp'] = np.mean(v_trial[events])
            row[f'std_{event_type_name}_amp'] = np.std(v_trial[events])
        
        # Save ISIs for spikes only
        isi = np.diff(spike_idxs)
        isi_corrected = np.ones(max_n_spikes - 1) * np.nan
        isi_corrected[:len(isi)] = isi
        for isi_idx in range(len(max_n_spikes - 1)):
            row[f"isi_{isi_idx}"] = isi_corrected[isi_idx]

        # Get I stats if needed
        if I is not None:
            i_trial = I[trial_idx]
            row["mean_i"] = np.nanmean(i_trial)
            row["std_i"] = np.std(i_trial)

        stat_df.append(row)
    
    stat_df = pd.concat(stat_df, axis = 0).reset_index(drop = True)
    return stat_df

#TODO: redo in the same format as VI_summary_features()
def get_hto_lto_stats(V: np.ndarray, lto_hto: list, train_features: list = None, dt: float = 1, CI_settings: list = None) -> tuple:
    '''
    Gets the LTO and HTO amplitude and frequency from the voltage traces.
    Packages the info into features and column names.
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
    
    lto_hto: list[float]
        A list of labels for lto/hto
        
    train_features: list[str], default = None
        A list of features that are extracted
    
    dt: float, default = 1
        Timestep
        
    CI_settings: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings
    
    Returns:
    -----------
    features_final: list[float]
        List of features
    
    column_names: list[str]
        List of column names
    '''
    features = []
    column_names = []
    
    if "lto-hto_frequency" in train_features:
        hto_lto_frequencies = calculate_hto_lto_frequency(V, CI_settings, dt, lto_hto)
        features.append(hto_lto_frequencies)
        column_names += ["lto-hto_frequency"]
    if "lto-hto_amplitude" in train_features:
        hto_lto_amplitudes = calculate_hto_lto_amplitude(V, CI_settings, dt, lto_hto)
        features.append(hto_lto_amplitudes)
        column_names += ["lto-hto_amplitude"]
    
    features_final = np.column_stack(features)
    
    return features_final, column_names

#TODO: redo in the same format as VI_summary_features()
def calculate_hto_lto_frequency(V: np.ndarray, CI_settings: list, dt: float, lto_hto: list) -> list:
    '''
    Directly calculates the frequency from LTO and HTO
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
        
    CI_settings: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings
    
    dt: float, default = 1
        Timestep
    
    lto_hto: list[float]
        A list of labels for lto/hto
        
    Returns:
    -----------
    frequencies: list[float]
        List of frequencies for each trace
    
    '''
    frequencies = []
    for i, v_trace in enumerate(V):
        if lto_hto[i] == 1:
            start_time = CI_settings[i].delay
            stop_time = CI_settings[i].delay + CI_settings[i].dur
            start_idx = int(np.round(start_time / dt))
            end_idx = int(np.round(stop_time / dt))
            end_idx = min(end_idx, len(v_trace))

            window = v_trace[start_idx:end_idx]

            samples = len(window)

            fft_result = np.fft.rfft(window)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.rfftfreq(samples, d=dt)

            if freqs[0] == 0:
                fft_magnitude[0] = 0

            max_index = np.argmax(fft_magnitude)

            principal_freq = freqs[max_index]
            frequencies.append(principal_freq)
        else:
            frequencies.append(1e6)

    return np.array(frequencies)

#TODO: redo in the same format as VI_summary_features()
def calculate_hto_lto_amplitude(V: np.ndarray, CI_settings: list, dt: float, lto_hto: list) -> list:
    '''
    Directly calculates the amplitude from LTO and HTO
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
        
    CI_settings: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings
    
    dt: float, default = 1
        Timestep
    
    lto_hto: list[float]
        A list of labels for lto/hto
        
    Returns:
    -----------
    amplitudes: list[float]
        List of amplitudes for each trace
    '''
    amplitudes = []
    for i, v_trace in enumerate(V):
        if lto_hto[i] == 1:
            start_time = CI_settings[i].delay
            stop_time = CI_settings[i].delay + CI_settings[i].dur
            start_idx = int(np.round(start_time / dt))
            end_idx = int(np.round(stop_time / dt))
            end_idx = min(end_idx, len(v_trace)) 

            window = v_trace[start_idx:end_idx]
            
            min_voltage = min(window)
            max_voltage = max(window)
            
            amplitude = max_voltage - min_voltage
            
            amplitudes.append(amplitude)
        else:
            amplitudes.append(1e6)
    
    return np.array(amplitudes)
    
#---------------------------------------------
# FILTERING
#---------------------------------------------
def remove_saturated_traces(data: np.ndarray, window_of_inspection: tuple, saturation_threshold: float = -50) -> np.ndarray:
    '''
    Filters training data to only voltage traces that have not saturated.

    Parameters:
    -----------
    data: np.ndarray of shape (n_trials, T, 4)
        Array (V, I, g, lto/hto/sim_idx).
        
    window_of_inspection: tuple(win_start, win_end)
        Boundaries for the window of inspection.
    
    saturation_threshold: float, default = -50
        A voltage threshold above which the trace is potentially saturated (mV).
        
    Returns:
    -----------
    filtered_data: np.ndarray of shape (n_trials, T, 4)
        Filtered data containing nonsaturated traces only.
    '''
    V = data[:, window_of_inspection[0] : window_of_inspection[1], 0]  
    nonsaturated_idxs = ~np.all(V > saturation_threshold, axis = 1)
    filtered_data = data[nonsaturated_idxs, :, :]
    print(f"Saturation filter: dropped {len(data) - len(filtered_data)} traces.")
    return filtered_data


def get_traces_with_spikes(data: np.ndarray, spike_threshold: float = -20) -> np.ndarray:
    '''
    Filters training data to only voltage traces that have spikes.

    Parameters:
    -----------
    data: np.ndarray of shape (n_trials, T, 4)
        Array (V, I, g, lto/hto/sim_idx).
    
    spike_threshold: int, default = -20
        Threshold for spike detection (mV).
        
    Returns:
    -----------
    traces_with_spikes: np.ndarray of shape (n_trials, T, 4)
        Filtered data containing only traces with spikes.
    '''
    V = data[:, :, 0]

    idx_with_spikes = []
    for trial_idx in range(len(V)):
        if len(find_events(V[trial_idx], spike_threshold)) > 0:
            idx_with_spikes.append(trial_idx)

    traces_with_spikes = data[idx_with_spikes]
    print(f"Spike filter: dropped {len(data) - len(traces_with_spikes)} traces.")
    return traces_with_spikes


#---------------------------------------------
# UTILS 
#---------------------------------------------
#TODO: check if we need everything here
def apply_decimate_factor(traces: np.ndarray, decimate_factor: float = None) -> np.ndarray:
    '''
    A method for downsampling data traces (voltage, current) from training data
    Parameters:
    -----------
    traces: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of timeseries data
    
    decimate_factor: float, default = None
        Downsampling factor
        
    Returns:
    -----------
    traces: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of timeseries data
    '''
    if decimate_factor:

        print(f"Reducing dataset by {decimate_factor}x")
        from scipy import signal

        traces = signal.decimate(
            traces, decimate_factor
        ).copy()  
    return traces


def combine_data(output_path: str) -> None:
    '''
    ACTSimulator outputs .npy files for each simulation run. This method combines this data
    into a single .npy file for ease of processing.
    Parameters:
    -----------
    output_path: str
        Output path
        
    Returns:
    -----------
    None
    '''
    print(output_path)
    file_list = sorted(glob.glob(os.path.join(output_path, "out_*.npy")))

    data_list = []

    for file_name in file_list:
        data = np.load(file_name)
        data_list.append(data)
        
    sorted_data_list = sorted(data_list, key=lambda arr: arr[0][3])

    final_data = np.stack(sorted_data_list, axis=0)
    np.save(os.path.join(output_path, f"combined_out.npy"), final_data)


def clean_g_bars(dataset: np.ndarray) -> np.ndarray:
    '''
    ACTSimulator pads the 3rd column (i.e. conductance set) with NANs. This method removes the pad.
    Parameters:
    -----------
    dataset: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>, 4)
        Voltage traces, current traces, conductance settings, and lto_hto/sim_idx information
    
    Returns:
    ----------
    cleaned_g_bars: np.ndarray of size (<num trials>, <num channels>)
        Array of conductance values for each trail
    '''
    def remove_nan_from_sample(sample):
        return sample[~np.isnan(sample)]
        
    cleaned_g_bars = np.array([remove_nan_from_sample(sample) for sample in dataset[:,:,2]])
    return cleaned_g_bars


def generate_I_g_combinations(channel_ranges: list, channel_slices: list, current_injections: list) -> tuple:
    '''
    This method directly calculates the combinations of conductances and current injection intensities
    based on the user input of conductance ranges/number of slices, and current injection intensity
    selections.
    Parameters:
    -----------
    channel_ranges: list[list[float]]
        A list of 2 element lists for low and high conductance values for each channel
    
    channel_slices: list[float]
        A list of the number of slices involved in each channel
    
    current_injections: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings
    
    Returns:
    ----------
    conductance_groups: list[list[float]]
        A list of conductance groups for each channel
    
    current_groups: list[list[float]]
        A list of current injection groups
    '''
    channel_values = [
        np.linspace(low, high, num=slices)
        for (low, high), slices in zip(channel_ranges, channel_slices)
    ]
    
    conductance_combinations = list(product(*channel_values))
    
    all_combinations = list(product(conductance_combinations, current_injections))
    
    conductance_groups = [comb[0] for comb in all_combinations]
    current_groups = [comb[1] for comb in all_combinations]
    
    return conductance_groups, current_groups


def get_fi_curve(V_matrix: np.ndarray, spike_threshold: float, CI_list: list, ignore_negative_CI = True) -> list:
    '''
    Given current injection intensities and voltage traces, this method calculates
    a list of frequencies (FI curve)
    Parameters:
    -----------
    V_matrix: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
    
    spike_threshold: float
        Spike threshold
        
    CI_list: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        A list of Current injection settings
        
    ignore_negative_CI: bool, default = True
        Does not include negative current injections when True
    
    Returns:
    ---------
    frequencies: list[float]
        List of frequencies
    
    '''
    # Count spikes
    raise RuntimeError
    counts = np.apply_along_axis(len(find_events(V_matrix)), axis = 1, arr = V_matrix, spike_threshold = spike_threshold)
    
    # Find injection durations
    amps = np.array([current_inj.amp for current_inj in CI_list])
    injection_durations = np.array([current_inj.dur for current_inj in CI_list])

    if ignore_negative_CI:
        non_neg_idx = np.where(amps >= 0)
        counts = counts[non_neg_idx]
        injection_durations = injection_durations[non_neg_idx]

    frequencies =  counts / (injection_durations / 1000)  # Convert to Hz: spikes / time (sec)

    return frequencies


def save_to_json(value, key: str, filename: str) -> None:
    '''
    A simple helper function to save a key-value pair to a given json file.
    Parameters:
    -----------
    value: Any
        Value
    
    key: str
        Key
    
    filename: str
        Filename
    
    Returns:
    ---------
    None
        
    '''
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    new_entry = {key: value}
    
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    existing_data.update(new_entry)
    
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

        
def load_metric_data(filenames:list) -> tuple:
    """
    Loads in all values from a saved metrics file.
    Parameters:
    -----------
    filenames: list[str]
        List of filepaths to metric files (json)
    
    Returns:
    -----------
    num_spikes_in_isi_calc_list: list[int]
        List of the number of spikes used for interspike interval calculations
        
    mean_interspike_interval_mae_list: list[float]
        List of mean interspike interval MAEs
        
    stdev_interspike_interval_mae_list: list[float]
        List of stdev interspike interval MAEs
        
    final_g_prediction_list: list[list[float]]
        List of final g predictions
        
    model_train_mean_mae_list: list[float]
        List of mean model training MAEs
        
    model_train_stdev_mae_list: list[float]
        List of stdev model training MAEs
        
    mae_final_predicted_g_list: list[float]
        List of conductance MAEs for final predictions
        
    prediction_evaluation_method_list: list[str]
        List of prediction evaluation methods
        
    final_prediction_fi_mae_list: list[float]
        List of FI curve MAEs for final predictions
        
    final_prediction_voltage_mae_list: list[float]
        List of voltage MAEs for final predictions
        
    feature_mae_list: list[float]
        List of feature MAEs for final predictions
        
    module_runtime_list: list[float]
        List of runtimes
        
    """
    num_spikes_in_isi_calc_list = []
    mean_interspike_interval_mae_list = []
    stdev_interspike_interval_mae_list = []
    final_g_prediction_list = []
    model_train_mean_mae_list = []
    model_train_stdev_mae_list = []
    mae_final_predicted_g_list = []
    prediction_evaluation_method_list = []
    final_prediction_fi_mae_list = []
    final_prediction_voltage_mae_list = []
    feature_mae_list = []
    module_runtime_list = []
    
    for filename in filenames:
        with open(filename, 'r') as file:
            data = json.load(file)
        
        num_spikes_in_isi_calc_list.append(data.get("num_spikes_in_isi_calc"))
        mean_interspike_interval_mae_list.append(data.get("mean_interspike_interval_mae"))
        stdev_interspike_interval_mae_list.append(data.get("stdev_interspike_interval_mae"))
        final_g_prediction_list.append(data.get("final_g_prediction"))
        model_train_mean_mae_list.append(data.get("model_train_mean_mae"))
        model_train_stdev_mae_list.append(data.get("model_train_stdev_mae"))
        mae_final_predicted_g_list.append(data.get("mae_final_predicted_g"))
        prediction_evaluation_method_list.append(data.get("prediction_evaluation_method"))
        final_prediction_fi_mae_list.append(data.get("final_prediction_fi_mae"))
        final_prediction_voltage_mae_list.append(data.get("final_prediction_voltage_mae"))
        feature_mae_list.append(data.get("summary_stats_mae_final_prediction"))
        module_runtime_list.append(data.get("module_runtime"))

    return (
        num_spikes_in_isi_calc_list, 
        mean_interspike_interval_mae_list, 
        stdev_interspike_interval_mae_list, 
        final_g_prediction_list, 
        model_train_mean_mae_list,
        model_train_stdev_mae_list,
        mae_final_predicted_g_list,
        prediction_evaluation_method_list, 
        final_prediction_fi_mae_list, 
        final_prediction_voltage_mae_list, 
        feature_mae_list,
        module_runtime_list
    )
    

def clear_directory(directory: str) -> None:
    """
    This method clears every file in a directory (to be used for deleting training data)
    clear_directory is developed with the help of OpenAI: o3-mini-high
    Parameters:
    -----------
    directory: str
        Directory to be cleared
    
    Returns:
    ---------
    None
    """
    print(f"Deleting Train Data: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)

        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def convert_target_csv_to_npy(num_traces: int, target_traces_filepath: str, output_folder_name: str) -> None:
    '''
    Helper function for users with voltage trace data in CSV format
    Parameters:
    -----------
    num_traces: int
        Number of traces in CSV
    
    target_traces_filepath: str
        Target traces filepath
    
    output_folder_name: str
        Output folder name (path)
    
    Returns:
    ----------
    None
    '''
    data = np.loadtxt(target_traces_filepath, delimiter=',', skiprows=1)
    
    csv_num_rows = data.shape[0]
    num_samples = csv_num_rows // num_traces
    
    V_I_data = data.reshape(num_traces, num_samples, 3)
    os.makedirs(output_folder_name + 'target/', exist_ok=True)
    np.save(output_folder_name + 'target/combined_out.npy', V_I_data)
    
    
def pickle_rf(rf_model, filename: str) -> None:
    '''
    Saves the trained RF model
    Parameters:
    -----------
    rf_model: Any
        Random Forest model
    
    filename: str
        Filename
    
    Returns:
    -----------
    None
    '''
    with open(filename, 'wb') as file:
        pickle.dump(rf_model, file)