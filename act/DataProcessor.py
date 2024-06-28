import os
import json
from io import StringIO
import glob

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from scipy import signal
import torch
import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm

from neuron import h
from act.act_types import PassiveProperties, SimulationConfig
from typing import Tuple

from act.act_types import SimulationParameters
from act.cell_model import TrainCell
from act.simulator import Simulator
from act.act_types import SimulationParameters
import matplotlib.pyplot as plt

import warnings

# Data Processor 
# Broken into main sections:

# 1. PASSIVE PROPERTIES:
#   : methods for simulating negative current injection and calculating 
#     passive properties from voltage recording stats.

# 2. TRACE FEATURE EXTRACTION:
#   : methods for extracting features from Voltage and Current traces
#     and shaping the data into a final output tensor
#       a. ARIMA STATS  : methods for generating ARIMA stats from a tensor of voltage traces
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
            if(not torch.is_tensor(summary_features)):
                summary_features = features
            else:
                summary_features = torch.cat(
                        (
                            summary_features,
                            features
                        ),
                        dim=0,
                    )
                
        # Spike Features (Necessary Feature Extraction)
        # Includes number of spikes, interspike interval, average minimum spike height, and average max spike height
        if "v_spike_stats" in list_of_features and V is not None:
            features, column_names = self.get_spike_stats(V)
            columns = columns + column_names
            if(not torch.is_tensor(summary_features)):
                summary_features = features
            else:
                summary_features = torch.cat(
                        (
                            summary_features,
                            features
                        ),
                        dim=0,
                    )

        # Mean potential of voltage trace as a feature
        if "v_mean_potential" in list_of_features and V is not None:
            features, column_names = self.get_mean_potential(V, inj_dur, inj_start)
            columns = columns + column_names
            if(not torch.is_tensor(summary_features)):
                summary_features = features
            else:
                summary_features = torch.cat(
                        (
                            summary_features,
                            features
                        ),
                        dim=0,
                    )

        # Amplitude of voltage trace as a feature
        if "v_amplitude_frequency" in list_of_features and V is not None:
            features, column_names = self.get_amplitude_frequency(V, inj_dur, inj_start)
            columns = columns + column_names
            if(not torch.is_tensor(summary_features)):
                summary_features = features
            else:
                summary_features = torch.cat(
                        (
                            summary_features,
                            features
                        ),
                        dim=0,
                    )

        # ARIMA coefficients of voltage trace as features # Calculated as one unit for all voltage traces. utilizing multithreading
        if "v_arima_coefs" in list_of_features and (arima_file or V is not None):
            features, column_names = self.get_arima_features(V=V, arima_file=arima_file)
            columns = columns + column_names
            if(not torch.is_tensor(summary_features)):
                summary_features = features
            else:
                summary_features = torch.cat(
                        (
                            summary_features,
                            features
                        ),
                        dim=0,
                    )
    
        return summary_features.T, columns


    #---------------------------------------------
    #PASSIVE PROPERTIES
    #---------------------------------------------
    # We need the surface area of the cell
    def get_surface_area(self, cell: TrainCell):
        h.load_file('stdrun.hoc')

        # Initialize the cell
        h.load_file(cell.hoc_file)
        init_Cell = getattr(h, cell.cell_name)
        cell = init_Cell()

        # Print out all of the sections that are found
        section_list = list(h.allsec())
        print(f"Found {len(section_list)} section(s) in this cell. Calculating the total surface area of the cell.")

        cell_area = 0

        # Loop through all sections and segments and add up the areas
        for section in section_list:  
            for segment in section:
                segment_area = h.area(segment.x, sec=section)
                cell_area += segment_area
        return cell_area
    

    # leak_conductance_var: the variable name used in the .hoc file for the leak conductance.
    def simulate_negative_CI(self, cell: TrainCell, leak_conductance_var: str):
        # Sim params
        h_dt = 0.01 #ms
        h_tstop = 1500 #ms

        # Current clamp params
        I_tstart = 500 #ms
        I_duration = 1000 # ms
        I_intensity = -0.1 # pA

        # Set all conductances other than the leak channel to 0
        non_leak = [g for g in cell.g_names if g != leak_conductance_var]
        cell.set_g(non_leak,[0] * len(non_leak))

        # Simulate the cell with just the leak channel set to non-zero conductance
        simulator = Simulator()
        simulator.submit_job(
            cell,
            SimulationParameters(
                sim_name = "passive_props",
                sim_idx=0,
                h_v_init = -70, # (mV)
                h_tstop = h_tstop,  # (ms)
                h_dt = h_dt, # (ms)
                h_celsius = 37, # (deg C)
                CI = {
                    "type": "constant",
                    "amp": I_intensity,
                    "dur": I_duration,
                    "delay": I_tstart
                }
            )
        )

        simulator.run(cell.mod_folder)

        # Returning needed simulation information
        # IMPORTANT: simulator.run() caps dt at 1 ms when it saves the data
        # For future calculations of passive props, dt = 1
        dt = 1 #ms
        cell_area = self.get_surface_area(cell) * 1e-8 # function returns um^2, want it in cm^2

        data = np.load(simulator.path + "/passive_props/out_0.npy")
        V = data[:,0]
        

        return V, dt, h_tstop, I_tstart, I_intensity, cell_area

    # V: 1D tensor holding the voltage trace data
    # dt: sample time (ms)
    # recording_duration: total duration of the recording (ms)
    # I_tstart: time when the current clamp starts (ms)
    # I_intensity: amps
    # leak_conductance_var: the variable name used in the .hoc file for the leak conductance.
    def calculate_passive_properties(self, V: torch.Tensor, dt, recording_duration, I_tstart, I_intensity, cell_area, leak_conductance_var: str) -> Tuple[PassiveProperties, np.ndarray]:

        # Get the initial and final voltage states of the step input (getting index first)
        index_v_rest = int(I_tstart / dt)
        index_v_final = int(recording_duration / dt) - 1

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

        tau = index_v_tau * dt                   # ms
        r_in = (v_diff) / (0 - I_intensity)     # MOhms
        g_leak = 1 / r_in                       # uS
        Cm = tau * g_leak                       # nF
        g_bar_leak = (g_leak / cell_area) / 1e6 # was in uS/cm^2. Divide by 1e6 gives us S/cm^2

        # Initialize a dictionary to hold all of the passive properties data
        passive_props: PassiveProperties = {
            "leak_conductance_variable": leak_conductance_var,
            "g_bar_leak": float(g_bar_leak),    # S/cm^2
            "r_in": float(r_in),                # MOhm
            "tau": float(tau),                  # ms
            "v_rest": float(v_rest),            # mV
            "Cm": float(Cm)                     # nF
        }

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
        elif os.path.exists("./arima_output/arima_stats.json"):
            # Use already generated data
            features = self.load_arima_coefs("./arima_output/arima_stats.json")
            column_names =  [f"arima{i}" for i in range(features.shape[0])]
            return features, column_names
        elif isinstance(V, (np.ndarray, torch.Tensor)):
            # Generate new data
            V = torch.Tensor(V)
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
        return torch.tensor(arima_dict["arima_coefs"]).T
    

    def generate_arima_coefficients(self, V: torch.Tensor, arima_order=(4,0,4), output_folder="./arima_output/", num_procs=64):
        
        print("-------------------------------------------------")
        print("GENERATING ARIMA STATS")
        print("-------------------------------------------------")
        output_file = output_folder + "arima_stats.json"

        print(f"ARIMA order set to {arima_order}")

        trace_list = []
        traces = V.cpu().detach().tolist()
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

        try:
            coefs = self.get_arima_coefs(trace, arima_order)
        except Exception as e:
            #print(f"problem processing cell {cell_id}: {e} | setting all values to 0.0")
            coefs = [0.0 for _ in range(num_coeffs)]

        trace_dict["coefs"] = coefs
        return trace_dict
    

    @timeout_decorator.timeout(180, use_signals=True, timeout_exception=Exception)
    def get_arima_coefs(self, trace: np.array, order=(10, 0, 10)):
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
    def get_current_stats(self, I: torch.Tensor, decimate_factor=None):

        I = self.apply_decimate_factor(I, decimate_factor)

        features = self.extract_current_stats(I)

        column_names = []
        column_names.append("I_mean")
        column_names.append("I_stdev")
        return features, column_names

    def extract_current_stats(self, I: torch.Tensor):
        I_features = []
        for i in range(len(I)):
            sample_features = torch.Tensor([torch.mean(I[i]), torch.std(I[i])])
            I_features.append(sample_features)
        return torch.stack(I_features).T
    
    #---------------------------------------------
    #VOLTAGE STATS
    #---------------------------------------------
    def get_spike_stats(self, V: torch.Tensor, decimate_factor=None):
        
        V = self.apply_decimate_factor(V, decimate_factor)

        # Extract spike summary features
        (   num_spikes_simulated,
            simulated_interspike_times,
            first_n_spikes, 
            avg_spike_min,
            avg_spike_max
        ) = self.extract_spike_features(V)

        features = torch.stack(
                (
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )

        column_names = []
        column_names.append("Num Spikes")
        column_names.append("Interspike Interval")
        column_names.append("Avg Min Spike Height")
        column_names.append("Avg Max Spike Height")

        return features, column_names

    # Also useful for current traces if needed
    def apply_decimate_factor(self, traces, decimate_factor=None):
        if decimate_factor:
            if isinstance(traces, torch.Tensor):
                traces = traces.cpu().detach().numpy()
            print(f"Reducing dataset by {decimate_factor}x")
            from scipy import signal

            traces = signal.decimate(
                traces, decimate_factor
            ).copy()  
        return torch.tensor(traces)


    def extract_spike_features(self, V, spike_threshold=0, n_spikes=20):
        num_spikes_list = []
        interspike_times_list = []
        first_n_spikes_scaled_list = []
        avg_spike_min_list = []
        avg_spike_max_list = []
        for i in range(len(V)):

            V_sample = torch.Tensor(V[i]).unsqueeze(0)

            threshold_crossings = torch.diff(V_sample > spike_threshold, dim=1)
            num_spikes = torch.round(torch.sum(threshold_crossings, dim=1) * 0.5)
            interspike_times = torch.zeros((V_sample.shape[0], 1))
            for i in range(threshold_crossings.shape[0]):
                interspike_times[i, :] = torch.mean(
                    torch.diff(
                        torch.arange(threshold_crossings.shape[1])[threshold_crossings[i, :]]
                    ).float()
                )
            interspike_times[torch.isnan(interspike_times)] = 0

            first_n_spikes = torch.zeros((V_sample.shape[0], n_spikes)) * V_sample.shape[1]
            avg_spike_min = torch.zeros((V_sample.shape[0], 1))
            avg_spike_max = torch.zeros((V_sample.shape[0], 1))
            for i in range(threshold_crossings.shape[0]):
                threshold_crossing_times = torch.arange(threshold_crossings.shape[1])[
                    threshold_crossings[i, :]
                ]
                spike_times = []
                spike_mins = []
                spike_maxes = []
                for j in range(0, threshold_crossing_times.shape[0], 2):
                    spike_times.append(threshold_crossing_times[j])
                    ind = threshold_crossing_times[j : j + 2].cpu().tolist()
                    end_ind = ind[1] if len(ind) == 2 else V.shape[1]
                    spike_maxes.append(
                        V_sample[i][max(0, ind[0] - 1) : min(end_ind + 5, V_sample.shape[1])].max()
                    )
                    spike_mins.append(
                        V_sample[i][max(0, ind[0] - 1) : min(end_ind + 5, V_sample.shape[1])].min()
                    )
                first_n_spikes[i][: min(n_spikes, len(spike_times))] = torch.tensor(
                    spike_times
                ).flatten()[: min(n_spikes, len(spike_times))]
                avg_spike_max[i] = torch.mean(torch.tensor(spike_maxes).flatten())
                avg_spike_min[i] = torch.mean(torch.tensor(spike_mins).flatten())
                first_n_spikes_scaled = (
                    first_n_spikes / V_sample.shape[1]
                )  # may be good to return this
            # Accumulate the stats across all samples
            num_spikes_list.append(num_spikes)
            interspike_times_list.append(interspike_times)
            first_n_spikes_scaled_list.append(first_n_spikes_scaled.squeeze(0))
            avg_spike_min_list.append(avg_spike_min)
            avg_spike_max_list.append(avg_spike_max)

        num_spikes_final = torch.Tensor(num_spikes_list)
        interspike_times_final = torch.Tensor(interspike_times_list)
        first_n_spikes_final = torch.stack(first_n_spikes_scaled_list)
        avg_spike_min_final = torch.Tensor(avg_spike_min_list)
        avg_spike_max_final = torch.Tensor(avg_spike_max_list)

        return num_spikes_final, interspike_times_final, first_n_spikes_final, avg_spike_min_final, avg_spike_max_final
    
    def get_amplitude_frequency(self, V, inj_dur, inj_start, fs=1000):
        amplitude, frequency = self.extract_amplitude_frequency(V, inj_dur, inj_start, fs=fs)

        features = torch.cat(
            (amplitude.reshape(-1, 1), frequency.reshape(-1, 1)), dim=1
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
                amplitude = x[peaks].mean()
            frequency = int(len(peaks) / (secs))
            amplitudes.append(amplitude)
            frequencies.append(frequency)

        amplitudes = torch.tensor(amplitudes)
        frequencies = torch.tensor(frequencies)

        amplitudes[torch.isnan(amplitudes)] = 0
        frequencies[torch.isnan(frequencies)] = 0

        return amplitudes, frequencies
    
    def get_mean_potential(self, V: torch.Tensor, inj_dur, inj_start, decimate_factor=None):
        V = self.apply_decimate_factor(V, decimate_factor)

        features = self.extract_mean_potential(V, inj_dur, inj_start)
        columns =  ["V_mean"]

        return features, columns
        
    def extract_mean_potential(self, V: torch.Tensor, inj_dur, inj_start):
        mean_potential_list = []
        for V_sample in V:
            mean_potential = V_sample[inj_start:inj_start+inj_dur].mean(dim=0)
            mean_potential_list.append(mean_potential)
        return torch.Tensor(mean_potential_list).unsqueeze(0)
    
    #---------------------------------------------
    # UTILS 
    #---------------------------------------------

    def combine_data(self, output_path: str):
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
            return sample[np.nonzero(sample)]
            
        cleaned_g_bars = np.array([remove_nan_from_sample(sample) for sample in dataset[:,:,2]])
        return cleaned_g_bars