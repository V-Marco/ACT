import os
import json
from io import StringIO
import glob
import warnings

from typing import List, Union
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from itertools import product

import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

from act.act_types import ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection
from act.cell_model import ACTCellModel


'''
Data Processor: A helper class for the Automatic Cell Tuner
Broken into main sections:

1. PASSIVE PROPERTIES:
  : methods for simulating negative current injection and calculating 
    passive properties from voltage recording stats.

2. TRACE FEATURE EXTRACTION:
  : methods for extracting features from Voltage and Current traces
    and shaping the data into a final output numpy array
      a. ARIMA STATS  : methods for generating ARIMA stats from a numpy array of voltage traces
      b. VOLT  STATS  : spike height, trough height, frequency, etc.
      c. CURRENT STATS: average intensity, etc.
3. FILTERING
  : methods for slimming training data according to varying parameters
4. UTILS
  : general helper functions (MISC)
'''


class DataProcessor:

    def __init__(self):
        pass

    #---------------------------------------------
    # Feature Extraction
    #---------------------------------------------

    '''
    extract_features
    The main method for extracting features (uses many helper functions) given a list of features.
    Feature options: ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "spike_intervals", "lto-hto_amplitude", "lto-hto_frequency","v_arima_coefs"]
    '''
    
    def extract_features(self,train_features=None, V=None, I=None, arima_file=None, threshold=0, num_spikes=20, dt=1, lto_hto = None, current_inj_combos = None):

        if(train_features is None):
            train_features = ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "spike_intervals", "lto-hto_amplitude", "lto-hto_frequency","v_arima_coefs"]

        columns = []
        summary_features = None
        
        def concatenate_features(existing_features, new_features):
            if existing_features is None:
                return new_features
            else:
                return np.concatenate((existing_features, new_features), axis=1)
       
        if "i_trace_stats" in train_features and I is not None:
            features, column_names = self.get_current_stats(I)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)

        if( "number_of_spikes" in train_features or 
            "spike_times" in train_features or 
            "spike_height_stats" in train_features or 
            "trough_times" in train_features or
            "trough_height_stats" in train_features or
            "spike_intervals" in train_features):
            features, column_names = self.get_voltage_stats(V, train_features=train_features,threshold=threshold,num_spikes=num_spikes,dt=dt)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)
            
        if "lto-hto_amplitude" in train_features or "lto-hto_frequency" in train_features:
            features, column_names = self.get_hto_lto_stats(V, lto_hto, train_features=train_features, dt=dt, CI_settings=current_inj_combos)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)
            
        if "v_arima_coefs" in train_features and (arima_file or V is not None):
            features, column_names = self.get_arima_features(V=V, arima_file=arima_file)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)
    
        return summary_features, columns
    
    '''
    get_arima_features
    Gets the arima coefficients either by loading in pre-calculated values, or calculating 
    them.
    '''
    
    def get_arima_features(self, V=None, arima_file=None):
        if arima_file and os.path.exists(arima_file):
            features = self.load_arima_coefs(arima_file)
            column_names =  [f"arima{i}" for i in range(features.shape[0])]
            return features, column_names
        elif isinstance(V, np.ndarray):
            self.generate_arima_coefficients(V)

            features = self.load_arima_coefs("./arima_output/arima_stats.json")
            column_names =  [f"arima{i}" for i in range(len(features[0]))]
            return features, column_names
        else:
            print("Arima file not found and voltage data not provided.")
            return None
    
    '''
    load_arima_coefs
    Loads the ARIMA coefficients if they have been saved before.
    '''

    def load_arima_coefs(self, input_file):
        with open(input_file) as json_file:
            arima_dict = json.load(json_file)
        return arima_dict["arima_coefs"]
    
    '''
    generate_arima_coefficients
    Calculates the ARIMA coefficients given the voltage traces and ARIMA order.
    Multiprocessing wrapper.
    '''

    def generate_arima_coefficients(self, V, arima_order=(4,0,4), output_folder="./arima_output/", num_procs=64):
        
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
            pool_output = list(tqdm(pool.imap(self.arima_processor, trace_list), total=len(trace_list)))
            
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
    
    '''
    arima_processor
    Called by generate_arima_coefficients
    Handles warnings and exceptions for individual processes.
    Calls get_arima_coefs
    '''

    def arima_processor(self, trace_dict):
        trace = trace_dict["trace"]
        arima_order = trace_dict["arima_order"]
        num_coeffs = trace_dict["num_coeffs"]

        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
        warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
        
        try:
            coefs = self.get_arima_coefs(trace, arima_order)
        except Exception as e:
            coefs = [0.0 for _ in range(num_coeffs)]

        trace_dict["coefs"] = coefs
        return trace_dict
    
    '''
    get_arima_coefs
    Directly calls the ARIMA processes from statsmodels and gets the ARIMA coefficients
    '''

    @timeout_decorator.timeout(180, use_signals=True, timeout_exception=Exception)
    def get_arima_coefs(self, trace: np.array, order=(10, 0, 10)):
        
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
    
    '''
    get_current_stats
    Gets the mean and stdev of the current injection trace and packages the info into features and
    column names
    '''

    def get_current_stats(self, I, decimate_factor=None):
        I = self.apply_decimate_factor(I, decimate_factor)
        
        features = self.extract_i_traces_stats(I)
        column_names = ["I_mean", "I_stdev"]
        
        return features, column_names

    '''
    extract_i_traces_stats
    Directly calculates the mean and standard deviation of the current injection trace.
    '''

    def extract_i_traces_stats(self, I):
        I_features = []
        for i_trace in I:
            sample_features = np.array([np.mean(i_trace), np.std(i_trace)])
            I_features.append(sample_features)
        return np.stack(I_features)
    
    '''
    get_voltage_stats
    Main method for extracting features from the voltage trace along a variety of features.
    Packages the information into features and column names.
    '''
    def get_voltage_stats(self, V, train_features=None, decimate_factor=None, threshold=0, num_spikes=20, dt=1):
        
        V = self.apply_decimate_factor(V, decimate_factor)

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
        ) = self.extract_v_traces_features(V, spike_threshold=threshold, num_spikes=num_spikes, dt=dt)
        
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

    '''
    extract_v_traces_features
    Directly calculates many of the voltage trace features
    '''

    def extract_v_traces_features(self, traces, spike_threshold = 0, num_spikes=20, dt=1):
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
  
    '''
    get_hto_lto_stats
    Gets the LTO and HTO amplitude and frequency from the voltage traces.
    Packages the info into features and column names.
    '''
        
    def get_hto_lto_stats(self, V, lto_hto, train_features=None, dt=1, CI_settings = None):
                
        features = []
        column_names = []
        
        print(train_features)
        print("lto-hto_frequency" in train_features)
        print("lto-hto_amplitude" in train_features)
        
        if "lto-hto_frequency" in train_features:
            hto_lto_frequencies = self.calculate_hto_lto_frequency(V, CI_settings, dt, lto_hto)
            print(hto_lto_frequencies)
            features.append(hto_lto_frequencies)
            column_names += ["lto-hto_frequency"]
        elif "lto-hto_amplitude" in train_features:
            hto_lto_amplitudes = self.calculate_hto_lto_amplitude(V, CI_settings, dt, lto_hto)
            print(hto_lto_amplitudes)
            features.append(hto_lto_amplitudes)
            column_names += ["lto-hto_amplitude"]
        
        features_final = np.column_stack(features)
        
        return features_final, column_names
    
    '''
    calculate_hto_lto_frequency
    Directly calculates the frequency from LTO and HTO
    '''
        
    def calculate_hto_lto_frequency(self, V, CI_settings, dt, lto_hto):
        frequencies = []
        for i, v_trace in enumerate(V):
            if lto_hto[i] == 1:
                start_time = CI_settings[i].delay
                stop_time = CI_settings[i].delay + CI_settings[i].dur
                start_idx = int(np.round(start_time / dt))
                end_idx = int(np.round(stop_time / dt))
                end_idx = min(end_idx, len(v_trace))

                window = v_trace[start_idx:end_idx]
                #print(f"window: {window}")

                n = len(window)

                if n == 0:
                    raise ValueError("Delay larger than simulation duration.")

                fft_result = np.fft.rfft(window)
                fft_magnitude = np.abs(fft_result)

                freqs = np.fft.rfftfreq(n, d=dt)

                if freqs[0] == 0:
                    fft_magnitude[0] = 0

                max_index = np.argmax(fft_magnitude)

                principal_freq = freqs[max_index]
                frequencies.append(principal_freq)
            else:
                frequencies.append(1e6)
        
        #print(f"frequencies: {frequencies}")

        return np.array(frequencies)
    
    '''
    calculate_hto_lto_amplitude
    Directly calculates the amplitude from LTO and HTO
    '''
    
    def calculate_hto_lto_amplitude(self, V, CI_settings, dt, lto_hto):
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
        print(amplitudes)
        
        return np.array(amplitudes)
        
    #---------------------------------------------
    # FILTERING
    #---------------------------------------------
    '''
    get_nonsaturated_traces
    Filters training data to only voltage traces that have not saturated.
    '''
    def get_nonsaturated_traces(self,data, window_of_inspection, threshold=-50, dt=1):
        inspection_window_start = int(window_of_inspection[0] / dt)
        inspection_window_end = int(window_of_inspection[1] / dt)
        
        V_traces = data[:, :, 0]  
        V_traces_inspection_window = V_traces[:, inspection_window_start:inspection_window_end]

        nonsaturated_ind = ~np.all(V_traces_inspection_window > threshold, axis=1)
        
        filtered_data = data[nonsaturated_ind, :, :]
        
        print(f"Dropping {len(data) - len(filtered_data)} traces where: All values in the window ({inspection_window_start}:{inspection_window_end}) are saturated (above {threshold} mV).")
        
        return filtered_data
    
    '''
    get_traces_with_spikes
    Filters training data to only voltage traces that have spikes
    '''
        
    def get_traces_with_spikes(self, data, spike_threshold=0, dt=1):
        V_traces = data[:, :, 0]  
        num_of_spikes_list, *_ = self.extract_v_traces_features(V_traces, spike_threshold=spike_threshold, dt=dt)
        traces_with_spikes = data[num_of_spikes_list > 0]
        
        print(f"Dropping {len(data) - len(traces_with_spikes)} traces where: No spikes are in the voltage trace.")

        return traces_with_spikes


    #---------------------------------------------
    # UTILS 
    #---------------------------------------------
    
    '''
    apply_decimate_factor
    A method for downsampling data traces (voltage, current) from training data
    '''
    def apply_decimate_factor(self, traces, decimate_factor=None):
        if decimate_factor:

            print(f"Reducing dataset by {decimate_factor}x")
            from scipy import signal

            traces = signal.decimate(
                traces, decimate_factor
            ).copy()  
        return traces
    
    '''
    combine_data
    ACTSimulator outputs .npy files for each simulation run. This method combines this data
    into a single .npy file for ease of processing.
    '''

    def combine_data(self, output_path: str):
        print(output_path)
        file_list = sorted(glob.glob(os.path.join(output_path, "out_*.npy")))

        data_list = []

        for file_name in file_list:
            data = np.load(file_name)
            data_list.append(data)
            
        sorted_data_list = sorted(data_list, key=lambda arr: arr[0][3])

        final_data = np.stack(sorted_data_list, axis=0)
        np.save(os.path.join(output_path, f"combined_out.npy"), final_data)
    
    '''
    clean_g_bars
    ACTSimulator pads the 3rd column (i.e. conductance set) with NANs. This method removes the pad.
    '''

    def clean_g_bars(self, dataset):
        def remove_nan_from_sample(sample):
            return sample[~np.isnan(sample)]
            
        cleaned_g_bars = np.array([remove_nan_from_sample(sample) for sample in dataset[:,:,2]])
        return cleaned_g_bars
    
    '''
    generate_I_g_combinations
    This method directly calculates the combinations of conductances and current injection intensities
    based on the user input of conductance ranges/number of slices, and current injection intensity
    selections.
    '''
    
    def generate_I_g_combinations(self, channel_ranges: list, channel_slices: list, current_injections: List[Union[ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection]]):
        channel_values = [
            np.linspace(low, high, num=slices)
            for (low, high), slices in zip(channel_ranges, channel_slices)
        ]
        
        conductance_combinations = list(product(*channel_values))
        
        all_combinations = list(product(conductance_combinations, current_injections))
        
        conductance_groups = [comb[0] for comb in all_combinations]
        current_groups = [comb[1] for comb in all_combinations]
        
        return conductance_groups, current_groups
    
    '''
    get_fi_curve
    Given current injection intensities and voltage traces, this method calculates
    a list of frequencies (FI curve)
    '''

    def get_fi_curve(self, V_test, current_injections: List[Union[ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection]], ignore_negative=True):
        num_of_spikes_list,*_ = self.extract_v_traces_features(V_test)
        amps = [current_inj.amp for current_inj in current_injections]
        injection_durations = np.array([current_inj.dur for current_inj in current_injections])
        if ignore_negative:
            non_neg_idx = [i for i, amp in enumerate(amps) if amp >= 0]
            amps = [amp for i, amp in enumerate(amps) if amp >= 0]
            num_of_spikes_list = num_of_spikes_list[non_neg_idx]

        frequencies =  num_of_spikes_list / (injection_durations / 1000)  # Convert to Hz: spikes / time (sec)

        return frequencies
    
    '''
    save_to_json
    A simple helper function to save a key-value pair to a given json file.
    '''
    
    def save_to_json(self, value, key, filename):
        
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
    
    '''
    load_metric_data
    Loads in all values from a saved metrics file.
    '''
            
    def load_metric_data(self, filenames):
        num_spikes_in_isi_calc_list = []
        mean_interspike_interval_mae_list = []
        stdev_interspike_interval_mae_list = []
        final_g_prediction_list = []
        rf_mean_g_score_mae_list = []
        rf_stdev_g_score_mae_list = []
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
            rf_mean_g_score_mae_list.append(data.get("rf_mean_g_score_mae"))
            rf_stdev_g_score_mae_list.append(data.get("rf_stdev_g_score_mae"))
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
            rf_mean_g_score_mae_list,
            rf_stdev_g_score_mae_list,
            mae_final_predicted_g_list,
            prediction_evaluation_method_list, 
            final_prediction_fi_mae_list, 
            final_prediction_voltage_mae_list, 
            feature_mae_list,
            module_runtime_list
        )