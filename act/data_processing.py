import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq

#---------------------------------------------
# Feature Extraction
#---------------------------------------------

# In **all** files:
#TODO: make sure all functions have """ for docs (not '''). COMPLETED
#TODO: make sure there is an empty line between func description and the Parameters word in all files. COMPLETED
#TODO: make sure self is not in Parameters. COMPLETED
#TODO: make sure every non-private has a description; make sure no description starts with "This function". Same for classes.
#TODO: make sure there are no unused imports; make sure imports are organized as: internal/external libs -- empty line -- act files, e.g. COMPLETED
# import numpy as np
# import os
# ...
# import pandas as pd
#
# from act.metrics import pp_error

def find_events(v: np.ndarray, spike_threshold: float = -20) -> np.ndarray:
    """
    Counts the number of spikes in a voltage trace. Returns a list of event indices.

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
    potential_spikes = signal.argrelextrema(v, np.greater, order = 5)[0]
    return np.array([spike_time for spike_time in potential_spikes if v[spike_time] > spike_threshold])

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
        I: np.ndarray,
        window: np.ndarray = None,
        spike_threshold: int = -20, 
        max_n_spikes: int = 20
        ) -> pd.DataFrame:
    """
    Compute voltage and current summary features.

    Parameters:
    ----------
    V: np.ndarray of shape (n_trials, T)
        Voltage traces (mV over ms).

    I: np.ndarray of shape (n_trials, T)
        Corresponding current traces (time un over ms).
    
    spike_threshold: int, default = -20
        Threshold for spike detection (mV).
    
    max_n_spikes: int, default = 20
        Maximum number of spike times to save.

    Returns:
    ----------
    stat_df: pd.DataFrame
        Dataframe with extracted summary features.
    """
    if len(V.shape) != 2:
        raise ValueError

    stat_df = []
    for trial_idx, v_trial in enumerate(V):
        row = pd.DataFrame(index = [0])

        # Remove edge effects
        if window is not None:
            v_trial = v_trial[window[0] : window[1]]
        else:
            window = [0, len(v_trial) + 1]
        
        # Save overall voltage stats
        row["mean_v"] = np.nanmean(v_trial)
        row["std_v"] = np.nanstd(v_trial)
        row["max_ampl_v"] = np.nanmax(v_trial) - np.nanmin(v_trial)

        # Extract # of spikes and # of troughs and save the number
        spike_idxs = find_events(v_trial, spike_threshold)
        trough_idxs = find_events(-v_trial, -spike_threshold)

        for events, event_type_name in zip([spike_idxs, trough_idxs], ["spike", "trough"]):

            # Save the number of events
            row[f"n_{event_type_name}s"] = len(events)
            
            # Save spike frequency
            if event_type_name == "spike":
                row[f"spike_frequency"] = len(events) * 1000 / len(v_trial) 
        
            # Save first max_n_spikes event times
            event_idxs_corrected = np.ones(max_n_spikes) * np.nan
            events_length = np.min([len(events), max_n_spikes])
            event_idxs_corrected[:events_length] = events[:events_length]

            for event_idx in range(max_n_spikes):
                row[f"{event_type_name}_{event_idx}"] = event_idxs_corrected[event_idx] + window[0]

            # Save event stats
            for opertaion, operation_name in zip([np.min, np.max, np.mean, np.std], ["min", "max", "mean", "std"]):
                if len(events) > 0:
                    row[f'{operation_name}_{event_type_name}_volt'] = opertaion(v_trial[events])
                else:
                    row[f'{operation_name}_{event_type_name}_volt'] = np.nan
        
        # Save ISIs for spikes only
        isi = np.diff(spike_idxs)
        isi_corrected = np.ones(max_n_spikes - 1) * np.nan
        isi_length = np.min([len(isi), max_n_spikes - 1])
        isi_corrected[:isi_length] = isi[:isi_length]
        for isi_idx in range(max_n_spikes - 1):
            row[f"isi_{isi_idx}"] = isi_corrected[isi_idx]
            
        # Save main frequency for LTO / HTO
        magnitude = np.abs(rfft(v_trial))[1:] # Remove DC
        freqs = rfftfreq(len(v_trial), d = 1 / 1000)
        row['main_freq'] = freqs[np.argmax(magnitude)]

        # Save I stats
        row["mean_i"] = np.nanmean(I[trial_idx])
        row["std_i"] = np.nanstd(I[trial_idx])

        stat_df.append(row)
    
    stat_df = pd.concat(stat_df, axis = 0).reset_index(drop = True)
    return stat_df

    
#---------------------------------------------
# FILTERING
#---------------------------------------------
def remove_saturated_traces(data: np.ndarray, window_of_inspection: tuple, saturation_threshold: float = -50) -> np.ndarray:
    """
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
    """
    V = data[:, window_of_inspection[0] : window_of_inspection[1], 0]  
    nonsaturated_idxs = ~np.all(V > saturation_threshold, axis = 1)
    filtered_data = data[nonsaturated_idxs, :, :]
    print(f"Saturation filter: dropped {len(data) - len(filtered_data)} traces.")
    return filtered_data


def get_traces_with_spikes(data: np.ndarray, spike_threshold: float = -20) -> np.ndarray:
    """
    Filters training data to only voltage traces that have spikes.

    Parameters:
    -----------
    data: np.ndarray of shape (n_trials, T, 3)
        Array (V, I, g).
    
    spike_threshold: int, default = -20
        Threshold for spike detection (mV).
        
    Returns:
    -----------
    traces_with_spikes: np.ndarray of shape (n_trials, T, 3)
        Filtered data containing only traces with spikes.
    """
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
    """
    ACTSimulator outputs .npy files for each simulation run. This method combines this data
    into a single .npy file for ease of processing.
    
    Parameters:
    -----------
    output_path: str
        Output path.
        
    Returns:
    -----------
    None
    """
    save_path = os.path.join(output_path, f"combined_out.npy")
    if os.path.exists(save_path):
        #raise RuntimeError(save_path + " already exists.")
        pass

    file_list = glob.glob(os.path.join(output_path, "out_*.npy"))

    # Sort by simulation index
    file_list = sorted(file_list, key = lambda s : int(s.split(".npy")[0].split("_")[-1]))

    # Read and combine data
    data_list = []
    for file_name in file_list:
        data = np.load(file_name)
        data_list.append(data)
    final_data = np.stack(data_list, axis = 0)
    
    print(save_path)
    np.save(save_path, final_data)


def clean_g_bars(dataset: np.ndarray) -> np.ndarray:
    """
    ACTSimulator pads the 3rd column (i.e. conductance set) with NANs. This method removes the pad for use in the model.
    
    Parameters:
    -----------
    dataset: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>, 3)
        Voltage traces, current traces, conductance settings
    
    Returns:
    ----------
    cleaned_g_bars: np.ndarray of size (<num trials>, <num channels>)
        Array of conductance values for each trail
    """
    def remove_nan_from_sample(sample):
        return sample[~np.isnan(sample)]
        
    cleaned_g_bars = np.array([remove_nan_from_sample(sample) for sample in dataset[:, :, 2]])
    return cleaned_g_bars