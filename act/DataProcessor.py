import os
import json
from io import StringIO
import glob

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from scipy import signal
import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm

from neuron import h
from act.act_types import PassiveProperties
from typing import Tuple
from itertools import product

from act.act_types import SimulationParameters
from act.cell_model import TrainCell, ACTCellModel
from act.simulator import Simulator

import warnings

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
# 3. VOLTAGE TRACE MANIPULATION AND FILTERING


class DataProcessor:

    def __init__(self):
        pass

    # Spike features automatically included. Must provide a voltage trace
    def extract_features(self,list_of_features=None, V=None, I=None, arima_file=None, inj_dur=None, inj_start=None, fs=None):

        # If a specific list is not provided, extract all features
        if(list_of_features is None):
            list_of_features = ["i_mean_stdev", "v_spike_stats", "v_mean_potential", "v_amplitude_frequency", "v_arima_coefs"]

        # I and V should be the same length
        columns = []
        summary_features = None
       
        # Mean and StdDev of the current input trace as features
        if "i_mean_stdev" in list_of_features and I is not None:
            features, column_names = self.get_current_stats(I)
            columns = columns + column_names
            if(not isinstance(summary_features, np.ndarray)):
                summary_features = features
            else:
                summary_features =  np.concatenate(
                        (
                            summary_features,
                            features
                        ),
                        axis=0,
                    )
                
        # Spike Features (Necessary Feature Extraction)
        # Includes number of spikes, interspike interval, average minimum spike height, and average max spike height
        if "v_spike_stats" in list_of_features and V is not None:
            features, column_names = self.get_spike_stats(V)
            columns = columns + column_names
            if(not isinstance(summary_features, np.ndarray)):
                summary_features = features
            else:
                summary_features =  np.concatenate(
                        (
                            summary_features,
                            features
                        ),
                        axis=0,
                    )

        # Mean potential of voltage trace as a feature
        if "v_mean_potential" in list_of_features and V is not None:
            features, column_names = self.get_mean_potential(V, inj_dur, inj_start)
            columns = columns + column_names
            if(not isinstance(summary_features, np.ndarray)):
                summary_features = features
            else:
                summary_features =  np.concatenate(
                        (
                            summary_features,
                            features
                        ),
                        axis=0,
                    )

        # Amplitude of voltage trace as a feature
        if "v_amplitude_frequency" in list_of_features and V is not None:
            features, column_names = self.get_amplitude_frequency(V, inj_dur, inj_start)
            columns = columns + column_names
            if(not isinstance(summary_features, np.ndarray)):
                summary_features = features
            else:
                summary_features =  np.concatenate(
                        (
                            summary_features,
                            features
                        ),
                        axis=0,
                    )

        # ARIMA coefficients of voltage trace as features # Calculated as one unit for all voltage traces. utilizing multithreading
        if "v_arima_coefs" in list_of_features and (arima_file or V is not None):
            features, column_names = self.get_arima_features(V=V, arima_file=arima_file)
            columns = columns + column_names
            if(not isinstance(summary_features, np.ndarray)):
                summary_features = features
            else:
                summary_features =  np.concatenate(
                        (
                            summary_features,
                            features
                        ),
                        axis=0,
                    )
    
        return np.transpose(summary_features), columns


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

        # Find index of first occurance where the voltage is less than the time constant for tau
        # Looking only after the start of the step input (where index_v_rest is)
        index_v_tau = next(
            index for index, voltage_value in enumerate(list(V[index_v_rest:]))
            if voltage_value < v_t_const
        )
        
        # Cell area
        train_cell._build_cell()
        train_cell.set_surface_area()            # cm^2

        tau = index_v_tau * dt                   # ms
        print(f"tau: {tau}")
        r_in = (v_diff) / (0 - I_intensity)      # MOhms
        print(f"r_in: {r_in}")
        g_leak = 1 / r_in                        # uS
        print(f"g_leak: {g_leak}")
        Cm = ((tau * g_leak)/1000) / train_cell.cell_area           # uF/cm^2
        g_bar_leak = (g_leak / train_cell.cell_area) / 1e6  # was in uS/cm^2. Divide by 1e6 gives us S/cm^2

        # Initialize a dictionary to hold all of the passive properties data
        passive_props: PassiveProperties = PassiveProperties(
            V_rest=float(v_rest),         # mV
            R_in=float(r_in),             # MOhm
            tau=float(tau),               # ms
            Cm=float(Cm),                 # uF/cm^2
            g_bar_leak=float(g_bar_leak), # S/cm^2
            cell_area=train_cell.cell_area,
            leak_conductance_variable=leak_conductance_var,
            leak_reversal_variable=leak_reversal_var,
            
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
            # "./arima_output/arima_coefs.json"
            features = self.load_arima_coefs("./arima_output/arima_stats.json")
            column_names =  [f"arima{i}" for i in range(features.shape[0])]
            return features, column_names
        else:
            print("Arima file not found and voltage data not provided.")
            return None
    

    def load_arima_coefs(self, input_file):
        with open(input_file) as json_file:
            arima_dict = json.load(json_file)
        return np.transpose(arima_dict["arima_coefs"])
    

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

        features = self.extract_current_stats(I)

        column_names = []
        column_names.append("I_mean")
        column_names.append("I_stdev")
        return features, column_names

    def extract_current_stats(self, I):
        I_features = []
        for i in range(len(I)):
            sample_features = np.array([np.mean(I[i]), np.std(I[i])])
            I_features.append(sample_features)
        return np.stack(I_features).T
    
    #---------------------------------------------
    #VOLTAGE STATS
    #---------------------------------------------
    def get_spike_stats(self, V, decimate_factor=None):
        
        V = self.apply_decimate_factor(V, decimate_factor)

        # Extract spike summary features
        (   num_spikes_simulated,
            simulated_interspike_times,
            first_n_spikes, 
            avg_spike_min,
            avg_spike_max
        ) = self.extract_spike_features(V)

        features = np.stack(
                (
                    np.array(num_spikes_simulated).flatten(),
                    np.array(simulated_interspike_times).flatten(),
                    np.array(avg_spike_min).flatten(),
                    np.array(avg_spike_max).flatten(),
                )
            )

        column_names = [
            "Num Spikes",
            "Interspike Interval",
            "Avg Min Spike Height",
            "Avg Max Spike Height"
        ]

        return features, column_names

    # Also useful for current traces if needed
    def apply_decimate_factor(self, traces, decimate_factor=None):
        if decimate_factor:

            print(f"Reducing dataset by {decimate_factor}x")
            from scipy import signal

            traces = signal.decimate(
                traces, decimate_factor
            ).copy()  
        return traces


    def extract_spike_features(self, V, spike_threshold=0, n_spikes=20):
        num_spikes_list = []
        interspike_times_list = []
        first_n_spikes_scaled_list = []
        avg_spike_min_list = []
        avg_spike_max_list = []
        for i in range(len(V)):
            V_sample = np.expand_dims(np.array(V[i]), axis=0)

            threshold_crossings = np.diff((V_sample > spike_threshold).astype(int), axis=1)
            num_spikes = np.sum(threshold_crossings > 0, axis=1)
            interspike_times = np.zeros((V_sample.shape[0], 1))
            for j in range(threshold_crossings.shape[0]):
                spike_times = np.where(threshold_crossings[j, :] > 0)[0]
                if len(spike_times) > 1:
                    interspike_times[j, 0] = np.mean(np.diff(spike_times))
                else:
                    interspike_times[j, 0] = 0

            first_n_spikes = np.zeros((V_sample.shape[0], n_spikes))
            avg_spike_min = np.zeros((V_sample.shape[0], 1))
            avg_spike_max = np.zeros((V_sample.shape[0], 1))
            for j in range(threshold_crossings.shape[0]):
                spike_times = np.where(threshold_crossings[j, :] > 0)[0]
                spike_maxes = []
                spike_mins = []
                for k in range(len(spike_times)):
                    start_ind = max(0, spike_times[k] - 1)
                    end_ind = min(spike_times[k] + 5, V_sample.shape[1]) if k < len(spike_times) - 1 else V_sample.shape[1]
                    spike_maxes.append(np.max(V_sample[j, start_ind:end_ind]))
                    spike_mins.append(np.min(V_sample[j, start_ind:end_ind]))
                
                first_n_spikes[j, :min(n_spikes, len(spike_times))] = spike_times[:n_spikes]
                avg_spike_max[j] = np.mean(spike_maxes) if spike_maxes else 0
                avg_spike_min[j] = np.mean(spike_mins) if spike_mins else 0
            
            first_n_spikes_scaled = first_n_spikes / V_sample.shape[1]
            
            num_spikes_list.append(num_spikes)
            interspike_times_list.append(interspike_times)
            first_n_spikes_scaled_list.append(first_n_spikes_scaled.squeeze(0))
            avg_spike_min_list.append(avg_spike_min)
            avg_spike_max_list.append(avg_spike_max)

        first_n_spikes_final = np.stack(first_n_spikes_scaled_list)

        return np.array(num_spikes_list), np.array(interspike_times_list), first_n_spikes_final, np.array(avg_spike_min_list), np.array(avg_spike_max_list)
    
    def get_amplitude_frequency(self, V, inj_dur, inj_start, fs=1000):
        amplitude, frequency = self.extract_amplitude_frequency(V, inj_dur, inj_start, fs=fs)

        features = np.concatenate(
            (amplitude.reshape(-1, 1), frequency.reshape(-1, 1)), axis=1
        ).T
        column_names = []
        column_names.append("amplitude")
        column_names.append("frequency")

        return features, column_names
        
        
    def extract_amplitude_frequency(self, V, inj_dur, inj_start, fs=1000):
        amplitudes = []
        frequencies = []
        for V_sample in V:
            x = V_sample[inj_start : inj_start + inj_dur]
            secs = len(x) / fs
            peaks = signal.find_peaks(x, prominence=0.1)[0].tolist()
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                amplitude = np.mean(x[peaks]) if peaks else 0
            frequency = int(len(peaks) / (secs)) if secs > 0 else 0
            amplitudes.append(amplitude)
            frequencies.append(frequency)

        amplitudes = np.array(amplitudes)
        frequencies = np.array(frequencies)

        amplitudes[np.isnan(amplitudes)] = 0
        frequencies[np.isnan(frequencies)] = 0

        return amplitudes, frequencies
        
    def get_mean_potential(self, V, inj_dur, inj_start, decimate_factor=None):
        V = self.apply_decimate_factor(V, decimate_factor)

        features = self.extract_mean_potential(V, inj_dur, inj_start)
        columns =  ["V_mean"]

        return features, columns
        
    def extract_mean_potential(self, V, inj_dur, inj_start):
        mean_potential_list = []
        for V_sample in V:
            mean_potential = V_sample[inj_start:inj_start+inj_dur].mean(axis=0)
            mean_potential_list.append(mean_potential)
        return np.expand_dims(mean_potential_list, axis=0)
    
    #---------------------------------------------
    # UTILS 
    #---------------------------------------------

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

        (   num_spikes,
            interspike_times,
            first_n_spikes, 
            avg_spike_min,
            avg_spike_max
        ) = self.extract_spike_features(V_test)

        if ignore_negative:
            non_neg_idx = [i for i, amp in enumerate(amps) if amp >= 0]
            amps = [amp for i, amp in enumerate(amps) if amp >= 0]
            num_spikes = num_spikes[non_neg_idx]

        frequencies =  num_spikes / (inj_dur / 1000)  # Convert to Hz: spikes / time (sec)

        return frequencies