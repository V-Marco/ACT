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

def select_features(df: pd.DataFrame, feature_keys: list) -> pd.DataFrame:
    """
    Returns a subset of the features dataframe based on a list of user selected feature keys.
    Useful for keyword shortcuts like "spike_times" and "isi" to get all columns related.
    Parameters:
    -----------
    df: pd.DataFrame
        The original full feature dataframe
    
    feature_keys: list[str]
        A list of strings describing the features that the user wants to extract
        
    Returns:
    -----------
    df_sub: pd.DataFrame
        A smaller pandas dataframe constrained to only select columns
    """
    cols = []
    for key in feature_keys:
        if key == "spike_times":
            cols += [c for c in df.columns if c.startswith("spike_time_")]
        elif key == "interspike_intervals":
            cols += [c for c in df.columns if c.startswith("isi_")]
        else:
            # fall back to substring‐match
            cols += [c for c in df.columns if key in c]
    # drop duplicates & preserve original order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return df[cols]


def get_summary_features(
        V: np.ndarray,
        CI: list = None,
        lto_hto = None,
        spike_threshold: int = -20, 
        max_n_spikes: int = 20
        ) -> pd.DataFrame:
    '''
    Compute voltage and current summary features.

    Parameters:
    ----------
    V: np.ndarray of shape (n_trials, T)
        Voltage traces (mV over ms).

    CI: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings for all trials.
    
    lto_hto: list[float], default = None
        A list of labels for lto/hto.
    
    spike_threshold: int, default = -20
        Threshold for spike detection (mV).
    
    max_n_spikes: int, default = 20
        Maximum number of spike times to save.

    Returns:
    ----------
    stat_df: pd.DataFrame
        Dataframe with extracted summary features.
    '''
    if len(V.shape) != 2:
        raise ValueError

    stat_df = []
    for trial_idx, v_trial in enumerate(V):
        row = pd.DataFrame(index = [0])
        
        # Crop data to only extract features within the current injection time
        start_time = CI[trial_idx].delay
        stop_time = CI[trial_idx].delay + CI[trial_idx].dur
        start_idx = int(np.round(start_time))
        end_idx = int(np.round(stop_time))
        end_idx = min(end_idx, len(v_trial))

        window = v_trial[start_idx:end_idx]
        
        # Save overall voltage stats
        row["mean_v"] = np.nanmean(window)
        row["std_v"] = np.nanstd(window)

        # Extract # of spikes and # of troughs and save the number
        spike_idxs = find_events(window, spike_threshold)
        trough_idxs = find_events(-window, -spike_threshold)

        for events, event_type_name in zip([spike_idxs, trough_idxs], ["spike", "trough"]):

            # Save the number of events
            row[f"n_{event_type_name}s"] = len(events)
            
            # Save spike frequency
            if event_type_name == "spike":
                row[f"spike_frequency"] = len(events) * 1000 / len(window) 
        
            # Save first max_n_spikes event times
            event_idxs_corrected = np.ones(max_n_spikes) * np.nan
            events_length = np.min([len(events), max_n_spikes])
            event_idxs_corrected[:events_length] = events[:events_length]

            for event_idx in range(max_n_spikes):
                row[f"{event_type_name}_{event_idx}"] = event_idxs_corrected[event_idx] + start_idx

            # Save event stats
            for opertaion, operation_name in zip([np.min, np.max, np.mean, np.std], ["min", "max", "mean", "std"]):
                if len(events) > 0:
                    row[f'{operation_name}_{event_type_name}_volt'] = opertaion(window[events])
                else:
                    row[f'{operation_name}_{event_type_name}_volt'] = np.nan
        
        # Save ISIs for spikes only
        isi = np.diff(spike_idxs)
        isi_corrected = np.ones(max_n_spikes - 1) * np.nan
        isi_length = np.min([len(isi), max_n_spikes - 1])
        isi_corrected[:isi_length] = isi[:isi_length]
        for isi_idx in range(max_n_spikes - 1):
            row[f"isi_{isi_idx}"] = isi_corrected[isi_idx] + start_idx
            
        # Save pricipal frequency and amplitude
        if lto_hto is not None:
            if lto_hto[trial_idx] == 1:
                # Get FFT Magnitudes
                samples = len(window)
                fft_result = np.fft.rfft(window)
                fft_magnitude = np.abs(fft_result)
                
                # Get FFT Frequencies
                freqs = np.fft.rfftfreq(samples, d=dt) #TODO
                
                # Ingnore DC component
                if freqs[0] == 0:
                    fft_magnitude[0] = 0

                max_magnitude_index = np.argmax(fft_magnitude)

                principal_freq = freqs[max_magnitude_index]
                row["principal_frequency"] = principal_freq
                row["amplitude"] = max(window) - min(window)
            else:
                row["principal_frequency"] = 1e6
                row["amplitude"] = 1e6

        # Save I stats
        row["mean_i"] = CI[trial_idx].amp_mean
        row["std_i"] = CI[trial_idx].amp_std

        stat_df.append(row)
    
    stat_df = pd.concat(stat_df, axis = 0).reset_index(drop = True)
    return stat_df

    
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
def combine_data(output_path: str) -> None:
    '''
    ACTSimulator outputs .npy files for each simulation run. This method combines this data
    into a single .npy file for ease of processing.
    Parameters:
    -----------
    output_path: str
        Output path.
        
    Returns:
    -----------
    None
    '''
    save_path = os.path.join(output_path, f"combined_out.npy")
    if os.path.exists(save_path):
        raise RuntimeError(save_path + " already exists.")

    file_list = sorted(glob.glob(os.path.join(output_path, "out_*.npy")))

    data_list = []

    for file_name in file_list:
        data = np.load(file_name)
        data_list.append(data)
        
    sorted_data_list = sorted(data_list, key=lambda arr: arr[0][3])

    final_data = np.stack(sorted_data_list, axis = 0)

    
    print(save_path)
    np.save(save_path, final_data)


def clean_g_bars(dataset: np.ndarray) -> np.ndarray:
    '''
    ACTSimulator pads the 3rd column (i.e. conductance set) with NANs. This method removes the pad for use in the model.
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

        
# def load_metric_data(filenames:list) -> tuple:
#     """
#     Loads in all values from a saved metrics file.
#     Parameters:
#     -----------
#     filenames: list[str]
#         List of filepaths to metric files (json)
    
#     Returns:
#     -----------
#     num_spikes_in_isi_calc_list: list[int]
#         List of the number of spikes used for interspike interval calculations
        
#     mean_interspike_interval_mae_list: list[float]
#         List of mean interspike interval MAEs
        
#     stdev_interspike_interval_mae_list: list[float]
#         List of stdev interspike interval MAEs
        
#     final_g_prediction_list: list[list[float]]
#         List of final g predictions
        
#     model_train_mean_mae_list: list[float]
#         List of mean model training MAEs
        
#     model_train_stdev_mae_list: list[float]
#         List of stdev model training MAEs
        
#     mae_final_predicted_g_list: list[float]
#         List of conductance MAEs for final predictions
        
#     prediction_evaluation_method_list: list[str]
#         List of prediction evaluation methods
        
#     final_prediction_fi_mae_list: list[float]
#         List of FI curve MAEs for final predictions
        
#     final_prediction_voltage_mae_list: list[float]
#         List of voltage MAEs for final predictions
        
#     feature_mae_list: list[float]
#         List of feature MAEs for final predictions
        
#     module_runtime_list: list[float]
#         List of runtimes
        
#     """
#     num_spikes_in_isi_calc_list = []
#     mean_interspike_interval_mae_list = []
#     stdev_interspike_interval_mae_list = []
#     final_g_prediction_list = []
#     model_train_mean_mae_list = []
#     model_train_stdev_mae_list = []
#     mae_final_predicted_g_list = []
#     prediction_evaluation_method_list = []
#     final_prediction_fi_mae_list = []
#     final_prediction_voltage_mae_list = []
#     feature_mae_list = []
#     module_runtime_list = []
    
#     for filename in filenames:
#         with open(filename, 'r') as file:
#             data = json.load(file)
        
#         num_spikes_in_isi_calc_list.append(data.get("num_spikes_in_isi_calc"))
#         mean_interspike_interval_mae_list.append(data.get("mean_interspike_interval_mae"))
#         stdev_interspike_interval_mae_list.append(data.get("stdev_interspike_interval_mae"))
#         final_g_prediction_list.append(data.get("final_g_prediction"))
#         model_train_mean_mae_list.append(data.get("model_train_mean_mae"))
#         model_train_stdev_mae_list.append(data.get("model_train_stdev_mae"))
#         mae_final_predicted_g_list.append(data.get("mae_final_predicted_g"))
#         prediction_evaluation_method_list.append(data.get("prediction_evaluation_method"))
#         final_prediction_fi_mae_list.append(data.get("final_prediction_fi_mae"))
#         final_prediction_voltage_mae_list.append(data.get("final_prediction_voltage_mae"))
#         feature_mae_list.append(data.get("summary_stats_mae_final_prediction"))
#         module_runtime_list.append(data.get("module_runtime"))

#     return (
#         num_spikes_in_isi_calc_list, 
#         mean_interspike_interval_mae_list, 
#         stdev_interspike_interval_mae_list, 
#         final_g_prediction_list, 
#         model_train_mean_mae_list,
#         model_train_stdev_mae_list,
#         mae_final_predicted_g_list,
#         prediction_evaluation_method_list, 
#         final_prediction_fi_mae_list, 
#         final_prediction_voltage_mae_list, 
#         feature_mae_list,
#         module_runtime_list
#     )
    

# def clear_directory(directory: str) -> None:
#     """
#     This method clears every file in a directory (to be used for deleting training data)
#     clear_directory is developed with the help of OpenAI: o3-mini-high
#     Parameters:
#     -----------
#     directory: str
#         Directory to be cleared
    
#     Returns:
#     ---------
#     None
#     """
#     print(f"Deleting Train Data: {directory}")
#     if not os.path.isdir(directory):
#         print(f"Error: {directory} is not a valid directory.")
#         return

#     # Iterate over all items in the directory
#     for item in os.listdir(directory):
#         item_path = os.path.join(directory, item)
#         try:
#             if os.path.isfile(item_path) or os.path.islink(item_path):
#                 os.unlink(item_path)

#         except Exception as e:
#             print(f"Failed to delete {item_path}. Reason: {e}")
    
    
# def pickle_rf(rf_model, filename: str) -> None:
#     '''
#     Saves the trained RF model
#     Parameters:
#     -----------
#     rf_model: Any
#         Random Forest model
    
#     filename: str
#         Filename
    
#     Returns:
#     -----------
#     None
#     '''
#     with open(filename, 'wb') as file:
#         pickle.dump(rf_model, file)