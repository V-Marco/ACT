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
from datetime import datetime, timedelta

from act.act_types import PassiveProperties
from act.cell_model import ACTCellModel



# Data Processor 
# Broken into main sections:

# 1. PASSIVE PROPERTIES:
#   : methods for simulating negative current injection and calculating 
#     passive properties from voltage recording stats.

# 2. TRACE FEATURE EXTRACTION:
#   : methods for extracting features from Voltage and Current traces
#     and shaping the data into a final output numpy array
#       a. ARIMA STATS  : methods for generating ARIMA stats from a numpy array of voltage traces
#       b. VOLT  STATS  : 
#       c. CURRENT STATS:
# 3. FILTERING
# 4. UTILS


class DataProcessor:

    def __init__(self):
        pass

    # Spike features automatically included. Must provide a voltage trace
    def extract_features(self,train_features=None, V=None, I=None, arima_file=None, threshold=0, num_spikes=20, dt=1, n_steps = None, step_time=None, dur=None, delay=None):

        # If a specific list is not provided, extract all features
        if(train_features is None):
            train_features = ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "spike_intervals", "lto-hto_amplitude", "lto-hto_frequency","v_arima_coefs"]

        # I and V should be the same length
        columns = []
        summary_features = None
        
        def concatenate_features(existing_features, new_features):
            if existing_features is None:
                return new_features
            else:
                return np.concatenate((existing_features, new_features), axis=1)
       
        # Mean and StdDev of the current input trace as features
        if "i_trace_stats" in train_features and I is not None:
            features, column_names = self.get_current_stats(I)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)
                
        # Spike Features (Necessary Feature Extraction)
        # Includes number of spikes, interspike interval, average minimum spike height, and average max spike height

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
            features, column_names = self.get_hto_lto_stats(V, train_features=train_features, n_steps=n_steps, step_time=step_time, dur=dur, delay=delay, dt=dt)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)

        # ARIMA coefficients of voltage trace as features # Calculated as one unit for all voltage traces. utilizing multithreading
        if "v_arima_coefs" in train_features and (arima_file or V is not None):
            features, column_names = self.get_arima_features(V=V, arima_file=arima_file)
            columns += column_names
            summary_features = concatenate_features(summary_features, features)
    
        return summary_features, columns

    #---------------------------------------------
    #PASSIVE PROPERTIES
    #---------------------------------------------

    # V: 1D numpy array holding the voltage trace data
    # dt: sample time (ms)
    # recording_duration: total duration of the recording (ms)
    # I_tstart: time when the current clamp starts (ms)
    # I_intensity: amps
    # leak_conductance_var: the variable name used in the .hoc file for the leak conductance.
    def calculate_passive_properties(self, V, train_cell: ACTCellModel, dt, I_tend, I_tstart, I_intensity, leak_conductance_var: str, leak_reversal_var: str) -> PassiveProperties:

        # Get the initial and final voltage states of the step input (getting index first)
        index_v_rest = int(I_tstart / dt)
        index_v_final = int(I_tend / dt) - 1

        v_rest = V[index_v_rest]
        v_final = V[index_v_final]

        # Get voltage for tau calculation
        v_diff = v_rest - v_final
        v_t_const = v_rest - (v_diff * 0.632)

        # Find index of the first occurance where the voltage is less than the time constant for tau
        # Looking only after the start of the step input (where index_v_rest is)
        index_v_tau = next(
            index for index, voltage_value in enumerate(list(V[index_v_rest:]))
            if voltage_value < v_t_const
        )
        
        # Cell area
        train_cell._build_cell()
        train_cell.set_surface_area()                               # cm^2

        tau = index_v_tau * dt                                      # ms
        r_in = (v_diff) / (0 - I_intensity)                         # MOhms
        g_leak = 1 / r_in                                           # uS
        Cm = ((tau * g_leak)/1000) / train_cell.cell_area           # uF/cm^2
        g_bar_leak = (g_leak / train_cell.cell_area) / 1e6  # g_leak in uS. Divide by 1e6 gives us S/cm^2

        # Initialize a dictionary to hold all of the passive properties data
        passive_props: PassiveProperties = PassiveProperties(
            V_rest=float(v_rest),                                   # mV
            R_in=float(r_in),                                       # MOhm
            tau=float(tau),                                         # ms
            Cm=float(Cm),                                           # uF/cm^2
            g_bar_leak=float(g_bar_leak),                           # S/cm^2
            cell_area=train_cell.cell_area,                         # cm^2
            leak_conductance_variable=leak_conductance_var,
            leak_reversal_variable=leak_reversal_var
        )
        
        return passive_props

    #---------------------------------------------
    #ARIMA STATS
    #---------------------------------------------
    # Provide either a voltage trace or an arima file
    def get_arima_features(self, V=None, arima_file=None):
        if arima_file and os.path.exists(arima_file):
            # Use already generated data
            features = self.load_arima_coefs(arima_file)
            column_names =  [f"arima{i}" for i in range(features.shape[0])]
            return features, column_names
        elif isinstance(V, np.ndarray):
            # Generate new data
            self.generate_arima_coefficients(V)

            features = self.load_arima_coefs("./arima_output/arima_stats.json")
            column_names =  [f"arima{i}" for i in range(len(features[0]))]
            return features, column_names
        else:
            print("Arima file not found and voltage data not provided.")
            return None
    

    def load_arima_coefs(self, input_file):
        with open(input_file) as json_file:
            arima_dict = json.load(json_file)
        return arima_dict["arima_coefs"]
    

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
        # ensure ordering
        pool_dict = {}
        for out in pool_output:
            pool_dict[out["cell_id"]] = out["coefs"]
        coefs_list = []
        
        for i in range(num_traces):
            if pool_dict.get(i):
                coefs_list.append(pool_dict[i])
            else:  # we didn't complete that task, was not found
                coefs_list.append([0 for _ in range(num_arima_vals)])

        output_dict = {}
        output_dict["arima_coefs"] = coefs_list

        os.makedirs(output_folder, exist_ok=True)

        with open(output_file, "w") as fp:
            json.dump(output_dict, fp, indent=4)

        return coefs_list
    

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
    

    #---------------------------------------------
    #CURRENT STATS
    #---------------------------------------------
    def get_current_stats(self, I, decimate_factor=None):
        I = self.apply_decimate_factor(I, decimate_factor)
        
        features = self.extract_i_traces_stats(I)
        column_names = ["I_mean", "I_stdev"]
        
        return features, column_names

    def extract_i_traces_stats(self, I):
        I_features = []
        for i_trace in I:
            sample_features = np.array([np.mean(i_trace), np.std(i_trace)])
            I_features.append(sample_features)
        return np.stack(I_features)
    
    #---------------------------------------------
    #VOLTAGE STATS
    #---------------------------------------------
    def get_voltage_stats(self, V, train_features=None, decimate_factor=None, threshold=0, num_spikes=20, dt=1):
        
        V = self.apply_decimate_factor(V, decimate_factor)

        # Extract spike summary features
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
            
            # Filter troughs that are between spikes (within the current injection time)
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
  
        
    def get_hto_lto_stats(self, V, train_features=None, dt=1, n_steps = None, step_time=None, dur=None, delay=None):
        start_time = delay + (step_time * (n_steps-1))
        end_time = delay + dur
        
        features = []
        column_names = []
        
        if "hto_lto_frequency" in train_features:
            features.append(self.calculate_hto_lto_frequency(V, start_time, end_time, dt))
            column_names += ["hto_lto_frequency"]
        elif "hto_lto_amplitude" in train_features:
            features.append(self.calculate_hto_lto_amplitude(V, start_time, end_time, dt))
            column_names += ["hto_lto_amplitude"]
        
        features_final = np.column_stack(features)
        
        return features_final, column_names
        
        
    def calculate_hto_lto_frequency(V, start_time, stop_time, dt):
        frequencies = []
        for v_trace in V:
            start_idx = int(np.round(start_time / dt))
            end_idx = int(np.round(stop_time / dt))
            end_idx = min(end_idx, len(v_trace)) 

            # Extract the window of the voltage trace
            window = v_trace[start_idx:end_idx]

            # Number of samples in the window
            n = len(window)

            # Check if the window has enough data
            if n == 0:
                raise ValueError("Delay larger than simulation duration.")

            # Perform FFT on the real-valued window
            fft_result = np.fft.rfft(window)
            fft_magnitude = np.abs(fft_result)

            # Frequencies corresponding to the FFT components
            freqs = np.fft.rfftfreq(n, d=dt)

            # Exclude the zero frequency (DC component) if necessary
            if freqs[0] == 0:
                fft_magnitude[0] = 0

            # Find the index of the maximum magnitude in the FFT result
            max_index = np.argmax(fft_magnitude)

            # Principal frequency corresponding to the maximum magnitude
            principal_freq = freqs[max_index]
            frequencies.append(principal_freq)

        return np.array(frequencies)
    
    def calculate_hto_lto_amplitude(V, start_time, stop_time, dt):
        amplitudes = []
        for v_trace in V:
            start_idx = int(np.round(start_time / dt))
            end_idx = int(np.round(stop_time / dt))
            end_idx = min(end_idx, len(v_trace)) 

            # Extract the window of the voltage trace
            window = v_trace[start_idx:end_idx]
            
            min_voltage = min(window)
            max_voltage = max(window)
            
            amplitudes.append(max_voltage - min_voltage)
        
        return np.array(amplitudes)
        
    #---------------------------------------------
    # FILTERING
    #---------------------------------------------
    def get_nonsaturated_traces(self,data, window_of_inspection, threshold=-50, dt=1):
        inspection_window_start = int(window_of_inspection[0] / dt)
        inspection_window_end = int(window_of_inspection[1] / dt)
        
        V_traces = data[:, :, 0]  
        V_traces_inspection_window = V_traces[:, inspection_window_start:inspection_window_end]

        nonsaturated_ind = ~np.all(V_traces_inspection_window > threshold, axis=1)
        
        filtered_data = data[nonsaturated_ind, :, :]
        
        print(f"Dropping {len(data) - len(filtered_data)} traces where: All values in the window ({inspection_window_start}:{inspection_window_end}) are saturated (above {threshold} mV).")
        
        return filtered_data
        
    def get_traces_with_spikes(self, data, spike_threshold=0, dt=1):
        V_traces = data[:, :, 0]  
        num_of_spikes_list, *_ = self.extract_v_traces_features(V_traces, spike_threshold=spike_threshold, dt=dt)
        traces_with_spikes = data[num_of_spikes_list > 0]
        
        print(f"Dropping {len(data) - len(traces_with_spikes)} traces where: No spikes are in the voltage trace.")

        return traces_with_spikes


    #---------------------------------------------
    # UTILS 
    #---------------------------------------------
    def apply_decimate_factor(self, traces, decimate_factor=None):
        if decimate_factor:

            print(f"Reducing dataset by {decimate_factor}x")
            from scipy import signal

            traces = signal.decimate(
                traces, decimate_factor
            ).copy()  
        return traces

    def combine_data(self, output_path: str):
        print(output_path)
        # Combine individual run outputs to a single file.
        file_list = sorted(glob.glob(os.path.join(output_path, "out_*.npy")))

        # Initialize an empty list to store the data from each file
        data_list = []

        # Loop through each file, load the data, and append to the list
        for file_name in file_list:
            data = np.load(file_name)
            data_list.append(data)

        # Concatenate all the arrays along a new axis (axis=0)
        final_data = np.stack(data_list, axis=0)
        np.save(os.path.join(output_path, f"combined_out.npy"), final_data)

    def clean_g_bars(self, dataset):
        def remove_nan_from_sample(sample):
            return sample[~np.isnan(sample)]
            
        cleaned_g_bars = np.array([remove_nan_from_sample(sample) for sample in dataset[:,:,2]])
        return cleaned_g_bars
    
    def generate_I_g_combinations(self, channel_ranges: list, channel_slices: list, current_intensities: list):
        channel_values = [
            np.linspace(low, high, num=slices)
            for (low, high), slices in zip(channel_ranges, channel_slices)
        ]
        
        # Generate all combinations of conductance values
        conductance_combinations = list(product(*channel_values))
        
        # Create a list of all combinations with current intensities
        all_combinations = list(product(conductance_combinations, current_intensities))
        
        # Separate conductance groups and current intensities
        conductance_groups = [comb[0] for comb in all_combinations]
        current_intensities = [comb[1] for comb in all_combinations]
        
        return conductance_groups, current_intensities
    

    def get_fi_curve(self, V_test, amps, ignore_negative=True, inj_dur=1000):
        num_of_spikes_list,*_ = self.extract_v_traces_features(V_test)

        if ignore_negative:
            non_neg_idx = [i for i, amp in enumerate(amps) if amp >= 0]
            amps = [amp for i, amp in enumerate(amps) if amp >= 0]
            num_of_spikes_list = num_of_spikes_list[non_neg_idx]

        frequencies =  num_of_spikes_list / (inj_dur / 1000)  # Convert to Hz: spikes / time (sec)
        
        #print(f"Number of spikes: {num_spikes}")
        #print(f"Frequencies: {frequencies}")

        return frequencies
    
    
    def save_to_json(self, value, key, filename):
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        new_entry = {key: value}
        
        # Get existing data if available
        try:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}
        
        existing_data.update(new_entry)
        
        # Write all the data back to the file
        with open(filename, 'w') as file:
            json.dump(existing_data, file, indent=4)
            
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
        

        # Go through each metrics file and get the data
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