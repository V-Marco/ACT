#import os
import json
from io import StringIO

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

#from scipy import signal
import torch
import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm

from neuron import h
#from act.act_types import PassiveProperties, SimulationConfig
from act import utils
#from typing import Tuple

#import warnings

# Data Processor 
# Broken into 3 main sections:
# 1. ARIMA STATS
# 2. FEATURE EXTRACTION
# 3. VOLTAGE TRACE MANIPULATION AND FILTERING


class DataProcessor:

    def __init__():
        pass

    #["arima", "spike", "current"]
    def extract_features(self, list_of_features, arima_file=None):
        feature_columns = []
        features = []
        if "arima" in list_of_features:
            features, column_names = self.get_arima_featues(arima_file)
        
        return features, column_names

    #---------------------------------------------
    #ARIMA STATS
    #---------------------------------------------

    def get_arima_featues(arima_file):
        features = DataProcessor.load_arima_coefs(input_file=arima_file)
        column_names =  [f"arima{i}" for i in range(features.shape[1])]
        return features, column_names
    

    def load_arima_coefs(output_folder, input_file=None):

        if input_file is None:
            input_file= output_folder + "arima_stats.json"

        with open(input_file) as json_file:
            arima_dict = json.load(json_file)
        return torch.tensor(arima_dict["arima_coefs"])
    

    def generate_arima_coefficients(self, selected_config, output_folder=None, num_procs=64):

        if output_folder is None:
            output_folder  = utils.get_sim_output_folder_name(selected_config)
        
        print("-------------------------------------------------")
        print("GENERATING ARIMA STATS")
        print("-------------------------------------------------")
        traces, params, amps = utils.load_parametric_traces(selected_config)
        output_file = output_folder + "arima_stats.json"

        segregation_index = utils.get_segregation_index(selected_config)
        arima_order = utils.get_arima_order(selected_config, segregation_index)
        print(f"ARIMA order set to {arima_order}")

        trace_list = []
        traces = traces.cpu().detach().tolist()
        num_traces = len(traces)
        for i, trace in enumerate(traces):
            trace_list.append(
                {
                    "cell_id": i,
                    "trace": trace,
                    "total": num_traces,
                    "arima_order": arima_order,
                }
            )
        with mp.Pool(num_procs) as pool:
            pool_output = list(tqdm(pool.imap(self.arima_processor, trace_list), total=len(trace_list)))
        # ensure ordering
        pool_dict = {}
        for out in pool_output:
            pool_dict[out["cell_id"]] = out["coefs"]
        coefs_list = []
        num_arima_vals = 2 + arima_order[0] + arima_order[2]
        for i in range(num_traces):
            if pool_dict.get(i):
                coefs_list.append(pool_dict[i])
            else:  # we didn't complete that task, was not found
                coefs_list.append([0 for _ in range(num_arima_vals)])

        output_dict = {}
        output_dict["arima_coefs"] = coefs_list

        with open(output_file, "w") as fp:
            json.dump(output_dict, fp, indent=4)

        return coefs_list
    

    def arima_processor(self, trace_dict):
        cell_id = trace_dict["cell_id"]
        trace = trace_dict["trace"]
        total = trace_dict["total"]
        arima_order = trace_dict["arima_order"]

        try:
            coefs = self.get_arima_coefs(trace, arima_order)
        except Exception as e:
            #print(f"problem processing cell {cell_id}: {e} | setting all values to 0.0")
            num_arima_vals = 2 + arima_order[0] + arima_order[2]
            coefs = [0.0 for _ in range(num_arima_vals)]

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
    def get_current_stats(I: torch.Tensor):
        mean = torch.mean(I)
        stddev = torch.std(I)
        features = torch.stack(
            torch.flatten(mean),
            torch.flatten(stddev)
        )

        column_names = []
        column_names.append("I_mean")
        column_names.append("I_stdev")
        return features, column_names
    
    #---------------------------------------------
    #VOLTAGE STATS
    #---------------------------------------------
    def get_spike_stats(self, V: torch.Tensor):
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



    def extract_spike_features(V: torch.Tensor, spike_threshold=0, n_spikes=20):

        threshold_crossings = torch.diff(V > spike_threshold, dim=1)
        num_spikes = torch.round(torch.sum(threshold_crossings, dim=1) * 0.5)
        interspike_times = torch.zeros((V.shape[0], 1))
        for i in range(threshold_crossings.shape[0]):
            interspike_times[i, :] = torch.mean(
                torch.diff(
                    torch.arange(threshold_crossings.shape[1])[threshold_crossings[i, :]]
                ).float()
            )
        interspike_times[torch.isnan(interspike_times)] = 0

        first_n_spikes = torch.zeros((V.shape[0], n_spikes)) * V.shape[1]
        avg_spike_min = torch.zeros((V.shape[0], 1))
        avg_spike_max = torch.zeros((V.shape[0], 1))
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
                    V[i][max(0, ind[0] - 1) : min(end_ind + 5, V.shape[1])].max()
                )
                spike_mins.append(
                    V[i][max(0, ind[0] - 1) : min(end_ind + 5, V.shape[1])].min()
                )
            first_n_spikes[i][: min(n_spikes, len(spike_times))] = torch.tensor(
                spike_times
            ).flatten()[: min(n_spikes, len(spike_times))]
            avg_spike_max[i] = torch.mean(torch.tensor(spike_maxes).flatten())
            avg_spike_min[i] = torch.mean(torch.tensor(spike_mins).flatten())
            first_n_spikes_scaled = (
                first_n_spikes / V.shape[1]
            )  # may be good to return this
        return num_spikes, interspike_times, first_n_spikes_scaled, avg_spike_min, avg_spike_max

    
