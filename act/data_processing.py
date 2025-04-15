import os
import json
from io import StringIO
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from itertools import product

import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import pickle

from multiprocessing import Pool

'''
Data Processor: A collection of helper methods for the Automatic Cell Tuner
Broken into main sections:

1. TRACE FEATURE EXTRACTION:
  : methods for extracting features from Voltage and Current traces
    and shaping the data into a final output numpy array
      a. ARIMA STATS  : methods for generating ARIMA stats from a numpy array of voltage traces
      b. VOLT  STATS  : spike height, trough height, frequency, etc.
      c. CURRENT STATS: average intensity, etc.
2. FILTERING
  : methods for slimming training data according to varying parameters
3. UTILS
  : general helper functions (MISC)
'''


#---------------------------------------------
# Feature Extraction
#---------------------------------------------

def extract_features(train_features: list = None, V: np.ndarray = None, I: np.ndarray = None, arima_file: str = None, threshold: float = 0, 
                     num_spikes: int = 20, dt: float = 1, lto_hto: list = None, current_inj_combos: list = None) -> tuple:
    """
    Extract features from voltage (V) and current (I) traces.
    If V and I are provided as lists (multiple traces), this method uses multiprocessing
    to process each trace in parallel. The method header remains unchanged.
    Method Developed by OpenAI's o3 mini high model
    
    Parameters:
    -----------
    train_features: list[str], default = None
        List of train feature categories
    
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
    
    I: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of current injection traces
    
    arima_file: str, default = None
        Filepath for file containing ARIMA Coefficients
        
    threshold: float, default = 0
        Spike threshold
        
    num_spikes: int, default = 20
        Number of spikes
    
    dt: float, default 1
        Timestep
    
    lto_hto: list[float], default = None
        List of labels 0 or 1 for if the trace is lto/hto or not
        
    current_inj_combos: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings

    
    Returns:
    -----------
    features_list: list[float]
        A list of the actual features extracted from voltage traces
    
    columns_list: list[str]]
        A list of column names labeling the features

    """
    if isinstance(V, list) and isinstance(I, list):
        arg_list = [
            (train_features, V_trace, I_trace, arima_file, threshold, num_spikes, dt, lto_hto, current_inj_combos)
            for V_trace, I_trace in zip(V, I)
        ]
        with Pool() as pool:
            results = pool.starmap(_extract_features_single, arg_list)
        features_list, columns_list = zip(*results)
        return list(features_list), list(columns_list)
    else:
        return _extract_features_single(train_features, V, I, arima_file, threshold, num_spikes, dt, lto_hto, current_inj_combos)


def _extract_features_single(train_features: list, V: np.ndarray, I: np.ndarray, arima_file: str, threshold: float, 
                             num_spikes: int, dt: float, lto_hto: list, current_inj_combos: list) -> tuple:
    """
    Core function to extract features from a single V and I trace.
    This is the same code as your original method.
    Method Developed by OpenAI's o3 mini high model
    
    Parameters:
    -----------
    train_features: list[str]
        List of train feature categories
    
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of voltage traces
    
    I: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of current injection traces
    
    arima_file: str
        Filepath for file containing ARIMA Coefficients
        
    threshold: float
        Spike threshold
        
    num_spikes: int
        Number of spikes
    
    dt: float
        Timestep
    
    lto_hto: list[float]
        List of labels 0 or 1 for if the trace is lto/hto or not
        
    current_inj_combos: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection]
        A list of Current injection settings
    
    Returns:
    -----------
    summary_features: list[float]
        A list of the actual features extracted from voltage traces
    
    columns: list[str]]
        A list of column names labeling the features
    """
    if train_features is None:
        train_features = ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", 
                            "trough_times", "trough_height_stats", "spike_intervals", 
                            "lto-hto_amplitude", "lto-hto_frequency", "v_arima_coefs"]

    columns = []
    summary_features = None

    def concatenate_features(existing_features, new_features):
        if existing_features is None:
            return new_features
        else:
            return np.concatenate((existing_features, new_features), axis=1)

    if "i_trace_stats" in train_features and I is not None:
        features, column_names = get_current_stats(I)
        columns += column_names
        summary_features = concatenate_features(summary_features, features)

    if ("number_of_spikes" in train_features or 
        "spike_times" in train_features or 
        "spike_height_stats" in train_features or 
        "trough_times" in train_features or
        "trough_height_stats" in train_features or
        "spike_intervals" in train_features):
        features, column_names = get_voltage_stats(V, train_features=train_features,
                                                        threshold=threshold, num_spikes=num_spikes, dt=dt)
        columns += column_names
        summary_features = concatenate_features(summary_features, features)

    if "lto-hto_amplitude" in train_features or "lto-hto_frequency" in train_features:
        features, column_names = get_hto_lto_stats(V, lto_hto, train_features=train_features, dt=dt, CI_settings=current_inj_combos)
        columns += column_names
        summary_features = concatenate_features(summary_features, features)

    if "v_arima_coefs" in train_features and (arima_file or V is not None):
        features, column_names = get_arima_features(V=V, arima_file=arima_file)
        columns += column_names
        summary_features = concatenate_features(summary_features, features)

    return summary_features, columns


#Deprecated
def get_arima_features(V: np.ndarray = None, arima_file: str = None) -> tuple:
    '''
    Gets the arima coefficients either by loading in pre-calculated values, or via calculation.
    
    --------------
    .. Deprecated
    --------------
    
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
        
    arima_file: str, default = None
        Filepath to file containing ARIMA coefficients
    
    Returns:
    -----------
    features: list[float]
        Features 
    
    column_names: list[str]
        Column labels
    '''
    if arima_file and os.path.exists(arima_file):
        features = load_arima_coefs(arima_file)
        column_names =  [f"arima{i}" for i in range(features.shape[0])]
        return features, column_names
    elif isinstance(V, np.ndarray):
        generate_arima_coefficients(V)

        features = load_arima_coefs("./arima_output/arima_stats.json")
        column_names =  [f"arima{i}" for i in range(len(features[0]))]
        return features, column_names
    else:
        print("Arima file not found and voltage data not provided.")
        return None


#Deprecated
def load_arima_coefs(input_file: str) -> list:
    '''
    Loads the ARIMA coefficients if they have been saved before.
    --------------
    .. Deprecated
    --------------
        
    Parameters:
    -----------
    input_file: str
        Filepath to json file containing ARIMA coefficients
    
    Returns:
    -----------
    arima_coefs: list
        Array of ARIMA coefficients
    '''
    with open(input_file) as json_file:
        arima_dict = json.load(json_file)
    return arima_dict["arima_coefs"]


#Deprecated
def generate_arima_coefficients(V: np.ndarray, arima_order: tuple = (4,0,4), output_folder: str = "./arima_output/", num_procs: int = 64) -> list:
    '''
    Calculates the ARIMA coefficients given the voltage traces and ARIMA order.
    Multiprocessing wrapper.
    
    --------------
    .. Deprecated
    --------------
        
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
    
    arima_order: tuple(int,int,int), default = (4,0,4)
        ARIMA order for AR, I, and MA
    
    output_folder: str, default = "./arima_output/"
        Filepath for save file
    
    num_procs: int, default = 64
        Number of processors in pool
    
    Returns:
    -----------
    coefs_list: list[dict]
        A list of ARIMA coefficients
    '''
    
    print("-------------------------------------------------")
    print("GENERATING ARIMA STATS")
    print("-------------------------------------------------")
    output_file = output_folder + "arima_stats.json"

    print(f"ARIMA order set to {arima_order}")

    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")

    trace_list = []
    traces = V.tolist()
    num_traces = len(traces)
    num_arima_vals = 2 + arima_order[0] + arima_order[2]
    for i, trace in enumerate(traces):
        trace_list.append(
            {
                "cell_id": i,
                "trace": trace,
                "num_traces": num_traces,
                "arima_order": arima_order,
                "num_coeffs" : num_arima_vals
            }
        )
    with mp.Pool(num_procs) as pool:
        pool_output = list(tqdm(pool.imap(arima_processor, trace_list), total=len(trace_list)))
        
    pool_dict = {}
    for out in pool_output:
        pool_dict[out["cell_id"]] = out["coefs"]
    coefs_list = []
    
    for i in range(num_traces):
        if pool_dict.get(i):
            coefs_list.append(pool_dict[i])
        else:  
            coefs_list.append([0 for _ in range(num_arima_vals)])

    output_dict = {}
    output_dict["arima_coefs"] = coefs_list

    os.makedirs(output_folder, exist_ok=True)

    with open(output_file, "w") as fp:
        json.dump(output_dict, fp, indent=4)

    return coefs_list


#Deprecated
def arima_processor(trace_dict: dict) -> dict:
    '''
    Called by generate_arima_coefficients
    Handles warnings and exceptions for individual processes.
    Calls get_arima_coefs
    
    --------------
    .. Deprecated
    --------------
    
    Parameters:
    -----------
    trace_dict: dict
        A dictionary containing the trace, the ARIMA order, and the number of coefficients
    
    Returns:
    -----------
    trace_dict: dict
        Fills the coefs key with a list of coefficients
    '''
    trace = trace_dict["trace"]
    arima_order = trace_dict["arima_order"]
    num_coeffs = trace_dict["num_coeffs"]

    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
    
    try:
        coefs = get_arima_coefs(trace, arima_order)
    except Exception as e:
        coefs = [0.0 for _ in range(num_coeffs)]

    trace_dict["coefs"] = coefs
    return trace_dict


#Deprecated
@timeout_decorator.timeout(180, use_signals=True, timeout_exception=Exception)
def get_arima_coefs(trace: np.array, order: tuple = (10, 0, 10)) -> list:
    '''
    Directly calls the ARIMA processes from statsmodels and gets the ARIMA coefficients.
    
    --------------
    .. Deprecated
    --------------
    
    Parameters:
    -----------
    trace: np.array
        A single voltage trace
        
    order: tuple(AR,I,MA), default = (10,0,10)
        ARIMA settings
    Returns:
    -----------
    coefs: list[float]
        A list of ARIMA coefficients
    '''
    
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
    
    model = ARIMA(endog=trace, order=order).fit()
    stats_df = pd.read_csv(
        StringIO(model.summary().tables[1].as_csv()),
        index_col=0,
        skiprows=1,
        names=["coef", "std_err", "z", "significance", "0.025", "0.975"],
    )
    stats_df.loc[stats_df["significance"].astype(float) > 0.05, "coef"] = 0
    coefs = stats_df.coef.tolist()
    return coefs


def get_current_stats(I: np.ndarray, decimate_factor: float = None) -> tuple:
    '''
    Gets the mean and stdev of the current injection trace and packages the info into features and
    column names
    Parameters:
    -----------
    I: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of current injection traces
    
    decimate_factor: float
        Decimate factor
    
    Returns:
    -----------
    features: list[float]
        List of features
    
    column_names: list[str]
        List of column names
    '''
    I = apply_decimate_factor(I, decimate_factor)
    
    features = extract_i_traces_stats(I)
    column_names = ["I_mean", "I_stdev"]
    
    return features, column_names


def extract_i_traces_stats(I: np.ndarray) -> list:
    '''
    Directly calculates the mean and standard deviation of the current injection trace.
    Parameters:
    -----------
    I: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>)
        List of current injection traces
    
    Returns:
    -----------
    I_features: list[float]
        List of features
    
    '''
    I_features = []
    for i_trace in I:
        sample_features = np.array([np.mean(i_trace), np.std(i_trace)])
        I_features.append(sample_features)
    return np.stack(I_features)


def get_voltage_stats(V: np.ndarray, train_features: list = None, decimate_factor: float = None, threshold: float = 0, num_spikes: int = 20, dt: float = 1) -> tuple:
    '''
    Main method for extracting features from the voltage trace along a variety of features.
    Packages the information into features and column names.
    Parameters:
    -----------
    V: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
    
    train_features: list[str], default = None
        A list of features that are extracted
        
    decimate_factor: float, default = None
        A factor for downsampling the data
        
    threshold: float, default = 0
        Spike threshold
    
    num_spikes: int, default = 20
        Number of spikes in analysis
    
    dt: float, default = 1
        Timestep
    
    Returns:
    -----------
    features_final: list[float]
        List of features
    
    column_names: list[str]
        List of column names
    
    '''
    
    V = apply_decimate_factor(V, decimate_factor)

    (
    num_of_spikes_list, 
    spike_times_list, 
    interspike_intervals_list, 
    min_spike_height_list, 
    max_spike_height_list,
    avg_spike_height_list,
    std_spike_height_list,
    num_of_troughs_list,
    trough_times_list,
    min_trough_height_list,
    max_trough_height_list,
    avg_trough_height_list,
    std_trough_height_list,
    mean_voltage_list,
    std_voltage_list
    ) = extract_v_traces_features(V, spike_threshold=threshold, num_spikes=num_spikes, dt=dt)
    
    features = []
    column_names = []
    
    if "number_of_spikes" in train_features:
        features.append(num_of_spikes_list)
        column_names += ["Num Spikes"]
    if "spike_times" in train_features:
        features.append(spike_times_list)
        num_times = spike_times_list.shape[1]
        column_names += [f"Spike Time {i+1}" for i in range(num_times)]
    if "spike_height_stats" in train_features:
        features.append(min_spike_height_list)
        features.append(max_spike_height_list)
        features.append(avg_spike_height_list)
        features.append(std_spike_height_list)
        column_names += ["Min Spike Height", 
                        "Max Spike Height", 
                        "Avg Spike Height", 
                        "Std Spike Height"]
    if "number_of_troughs" in train_features:
        features.append(num_of_troughs_list)
        column_names += ["Num Troughs"]
    if "trough_times" in train_features:
        features.append(trough_times_list)
        num_times = trough_times_list.shape[1]
        column_names += [f"Trough Time {i+1}" for i in range(num_times)]
    if "trough_height_stats" in train_features:
        features.append(min_trough_height_list)
        features.append(max_trough_height_list)
        features.append(avg_trough_height_list)
        features.append(std_trough_height_list)
        column_names += ["Min Trough Height", 
                        "Max Trough Height", 
                        "Avg Trough Height",
                        "Std Trough Height"]
    if "spike_intervals" in train_features:
        features.append(interspike_intervals_list)
        num_intervals = interspike_intervals_list.shape[1]
        column_names += [f"Interspike Interval {i+1}" for i in range(num_intervals)]
        
    features_final = np.column_stack(features)

    return features_final, column_names


def extract_v_traces_features(traces: np.ndarray, spike_threshold: float = 0, num_spikes: int = 20, dt: float = 1) -> tuple:
    '''
    Directly calculates many of the voltage trace features
    Parameters:
    -----------
    traces: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>), default = None
        List of voltage traces
        
    spike_threshold: float, default = 0
        Spike threshold
        
    num_spikes: int, default = 20
        Number of spikes in analysis
    
    dt: float, default = 1
        Timestep
    
    Returns:
    -----------
    num_of_spikes_list: list[int]
        A list of the number of spikes in each voltage trace
        
    spike_times_list: list[list[float]]
        A list of the first n spike times in each voltage trace
        
    interspike_intervals_list: list[list[float]]
        A list of the first n interspike intervals in each voltage trace
        
    min_spike_height_list: list[float]
        A list of the minimum spike height in each voltage trace
        
    max_spike_height_list: list[float]
        A list of the maximum spike height in each voltage trace
        
    avg_spike_height_list: list[float]
        A list of the average spike height in each voltage trace
        
    std_spike_height_list: list[float]
        A list of the stdev of spike height in each voltage trace
        
    num_of_troughs_list: list[int]
        A list of the number of troughs in each voltage trace
        
    trough_times_list: list[list[float]]
        A list of the first n trough times in each voltage trace
        
    min_trough_height_list: list[float]
        A list of the minimum trought height in each voltage trace
        
    max_trough_height_list: list[float]
        A list of the maximum trough height in each voltage trace
        
    avg_trough_height_list: list[float]
        A list of the average trough height in each voltage trace
        
    std_trough_height_list: list[float]
        A list of the stdev trough height in each voltage trace
        
    mean_voltage_list: list[float]
        A list of the mean voltage in each voltage trace
        
    std_voltage_list: list[flaot]
        A list of the stdev voltage in each voltage trace
        
    '''
    num_of_spikes_list = []
    spike_times_list = []
    interspike_intervals_list = []
    min_spike_height_list = []
    max_spike_height_list = []
    avg_spike_height_list = []
    std_spike_height_list = []
    num_of_troughs_list = []
    trough_times_list = []
    min_trough_height_list = []
    max_trough_height_list = []
    avg_trough_height_list = []
    std_trough_height_list = []
    mean_voltage_list = []
    std_voltage_list = []
    
    pad_value=1e6
    
    for trace in traces:
        peak_idxs, _ = find_peaks(trace, threshold=spike_threshold, height=spike_threshold)
        
        num_of_spikes = len(peak_idxs)
        num_of_spikes_list.append(num_of_spikes)
        
        spike_times = peak_idxs[:num_spikes] * dt
        spike_times_padded = np.pad(spike_times, (0, num_spikes - len(spike_times)), 'constant', constant_values=pad_value)
        spike_times_list.append(spike_times_padded)
        
        interspike_intervals = np.diff(spike_times)
        interspike_intervals_padded = np.pad(interspike_intervals, (0,num_spikes - 1 - len(interspike_intervals)), 'constant', constant_values=pad_value)
        interspike_intervals_list.append(interspike_intervals_padded)
        
        spike_heights = trace[peak_idxs]
        min_spike_height = np.min(spike_heights) if len(spike_heights) > 0 else 1e6
        max_spike_height = np.max(spike_heights) if len(spike_heights) > 0 else 1e6
        avg_spike_height = np.mean(spike_heights) if len(spike_heights) > 0 else 1e6
        std_spike_height = np.std(spike_heights) if len(spike_heights) > 0 else 1e6
        
        min_spike_height_list.append(min_spike_height)
        max_spike_height_list.append(max_spike_height)
        avg_spike_height_list.append(avg_spike_height)
        std_spike_height_list.append(std_spike_height)
        
        mean_voltage = np.mean(trace)
        std_voltage = np.std(trace)
        
        mean_voltage_list.append(mean_voltage)
        std_voltage_list.append(std_voltage)
        
        inverted_trace = -trace  
        trough_idxs, _ = find_peaks(inverted_trace, threshold=-spike_threshold) 
        
        valid_trough_idx = []
        for idx in trough_idxs:
            before_spike = peak_idxs[peak_idxs < idx]
            after_spike = peak_idxs[peak_idxs > idx]
            
            if len(before_spike) > 0 and len(after_spike) > 0:
                valid_trough_idx.append(idx)

        valid_trough_idx = np.array(valid_trough_idx, dtype=int)
        
        num_of_troughs = len(valid_trough_idx)
        num_of_troughs_list.append(num_of_troughs)
        
        trough_times = valid_trough_idx[:num_spikes] * dt
        trough_times_padded = np.pad(trough_times, (0, num_spikes - len(trough_times)), 'constant', constant_values=pad_value)
        trough_times_list.append(trough_times_padded)
        
        trough_heights = trace[valid_trough_idx]
        min_trough_height = np.min(trough_heights) if len(trough_heights) > 0 else 1e6
        max_trough_height = np.max(trough_heights) if len(trough_heights) > 0 else 1e6
        avg_trough_height = np.mean(trough_heights) if len(trough_heights) > 0 else 1e6
        std_trough_height = np.std(trough_heights) if len(trough_heights) > 0 else 1e6
        
        min_trough_height_list.append(min_trough_height)
        max_trough_height_list.append(max_trough_height)
        avg_trough_height_list.append(avg_trough_height)
        std_trough_height_list.append(std_trough_height)
        
    return (
        np.array(num_of_spikes_list), 
        np.array(spike_times_list), 
        np.array(interspike_intervals_list), 
        np.array(min_spike_height_list), 
        np.array(max_spike_height_list),
        np.array(avg_spike_height_list),
        np.array(std_spike_height_list),
        np.array(num_of_troughs_list),
        np.array(trough_times_list),
        np.array(min_trough_height_list),
        np.array(max_trough_height_list),
        np.array(avg_trough_height_list),
        np.array(std_trough_height_list),
        np.array(mean_voltage_list),
        np.array(std_voltage_list)
    )

    
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
def get_nonsaturated_traces(data: np.ndarray, window_of_inspection: tuple, threshold: float = -50, dt: float = 1) -> np.ndarray:
    '''
    Filters training data to only voltage traces that have not saturated.
    Parameters:
    -----------
    data: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>, 4)
        Voltage traces, current traces, conductance settings, and lto_hto/sim_idx information
        
    window_of_inspection: tuple(<window_start>,<window_end>)
        Boundaries for the window of inspection
    
    threshold: float, default = -50
        A voltage threshold above which is potentially saturation
    
    dt: float, default = 1
        Timestep
        
    Returns:
    -----------
    filtered_data: np.ndarray of size = (<number of trials>, <number of timesteps>, 4)
        Filtered data to nonsaturated traces only
    '''
    inspection_window_start = int(window_of_inspection[0] / dt)
    inspection_window_end = int(window_of_inspection[1] / dt)
    
    V_traces = data[:, :, 0]  
    V_traces_inspection_window = V_traces[:, inspection_window_start:inspection_window_end]

    nonsaturated_ind = ~np.all(V_traces_inspection_window > threshold, axis=1)
    
    filtered_data = data[nonsaturated_ind, :, :]
    
    print(f"Dropping {len(data) - len(filtered_data)} traces where: All values in the window ({inspection_window_start}:{inspection_window_end}) are saturated (above {threshold} mV).")
    
    return filtered_data


def get_traces_with_spikes(data: np.ndarray, spike_threshold: float = 0, dt: float = 1) -> np.ndarray:
    '''
    Filters training data to only voltage traces that have spikes
    Parameters:
    -----------
    data: np.ndarray of shape = (<number of trials>, <number of timesteps in sim>, 4)
        Voltage traces, current traces, conductance settings, and lto_hto/sim_idx information
    
    spike_threshold: float, default = -50
        A voltage threshold above which is potentially saturation
    
    dt: float, default = 1
        Timestep
        
    Returns:
    -----------
    traces_with_spikes: np.ndarray of size = (<number of trials>, <number of timesteps>, 4)
        Filtered data to spiking traces only
    '''
    V_traces = data[:, :, 0]  
    num_of_spikes_list, *_ = extract_v_traces_features(V_traces, spike_threshold=spike_threshold, dt=dt)
    traces_with_spikes = data[num_of_spikes_list > 0]
    
    print(f"Dropping {len(data) - len(traces_with_spikes)} traces where: No spikes are in the voltage trace.")

    return traces_with_spikes


#---------------------------------------------
# UTILS 
#---------------------------------------------
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


def count_spikes(v: np.ndarray, spike_threshold: float = -20) -> list:
    """
    Counts the number of spikes in a voltage trace. Returns a list of event indices
    Parameters:
    -----------
    v: np.ndarray of shape = (<number of timesteps in sim>)
        A single voltage trace
    
    spike_threshold: float, default = -20
        Spike threshold
        
    Returns:
    -----------
    event_indices: list[float]
        List of voltage trace indices where spikes occur
    
    """
    # Find the indices where the voltage is above the threshold
    above_threshold_indices = np.where(v[:-1] > spike_threshold)[0]

    # Find the indices where the slope changes from positive to negative
    slope = np.diff(v)
    positive_to_negative_indices = np.where((slope[:-1] > 0) & (slope[1:] < 0))[0]

    # Find the intersection of the two sets of indices
    event_indices = np.intersect1d(above_threshold_indices, positive_to_negative_indices)

    return len(event_indices)


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
    counts = np.apply_along_axis(count_spikes, axis = 1, arr = V_matrix, spike_threshold = spike_threshold)
    
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