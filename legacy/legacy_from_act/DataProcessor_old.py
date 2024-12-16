import os
import json
from io import StringIO

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
from legacy import utils
from typing import Tuple

import warnings

# Data Processor 
# Broken into 3 main sections:
# 1. ARIMA STATS
# 2. FEATURE EXTRACTION
# 3. VOLTAGE TRACE MANIPULATION AND FILTERING


class DataProcessor:
    #---------------------------------------------
    #ARIMA STATS
    #---------------------------------------------
    @staticmethod
    def load_arima_coefs(output_folder, input_file=None):

        if input_file is None:
            input_file= output_folder + "arima_stats.json"

        with open(input_file) as json_file:
            arima_dict = json.load(json_file)
        return torch.tensor(arima_dict["arima_coefs"])
    

    def generate_arima_coefficients(self, selected_config, output_folder=None, num_procs=64):

        if output_folder is None:
            output_folder  = utils.get_sim_output_folder_name(selected_config)

        '''
        arma_stats_already_exists = os.path.exists(f"{output_folder}arima_stats.json")
        want_to_generate_arma = selected_config["optimization_parameters"]["generate_arma"]
        
        if (arma_stats_already_exists):
            print("--------------------------------------------------------------------")
            print(f"ARIMA STATS ALREADY GENERATED - Using stats from: {f"{output_folder}arima_stats.json"}")
            print("--------------------------------------------------------------------")
            return None
        elif (not want_to_generate_arma):
            print("-------------------------------------------------")
            print("ARIMA STATS TURNED OFF IN SIMULATION CONFIGURATION")
            print("-------------------------------------------------")
            return None
        else:
        '''
        
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
    # FEATURE EXTRACTION
    #---------------------------------------------


    def get_summary_features(self, V: torch.Tensor, I: torch.Tensor, spiking_ind, nonsaturated_ind):
        
        # Extract spike summary features
        (   num_spikes_simulated,
            simulated_interspike_times,
            first_n_spikes, 
            avg_spike_min, 
            avg_spike_max
        ) = self.extract_spike_features(V)

        coefs_loaded = False
        if os.path.exists("output/arima_stats.json"):
            coefs_loaded = True
            coefs = self.load_arima_coefs(
                input_file="output/arima_stats.json"
            )  # [subset_target_ind] # TODO REMOVE for testing quickly

            if self.spiking_only:
                coefs = coefs[spiking_ind]
            if self.nonsaturated_only:
                coefs = coefs[nonsaturated_ind]

        def generate_arima_columns(coefs):
            return [f"arima{i}" for i in range(coefs.shape[1])]

        summary_features = None
        summary_feature_columns = []
        if coefs_loaded:
            summary_features = torch.stack(
                (
                    # ampl_next_stage,
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )
            summary_feature_columns.append("Num Spikes")
            summary_feature_columns.append("Interspike Interval")
            summary_feature_columns.append("Avg Min Spike Height")
            summary_feature_columns.append("Avg Max Spike Height")

            if self.use_spike_summary_stats:
                summary_features = torch.cat(
                    (summary_features.T, first_n_spikes, coefs), dim=1
                )
                for i in range(first_n_spikes.shape[1]):
                    summary_feature_columns.append(f"Spike {i+1} time")
                summary_feature_columns = (
                    summary_feature_columns + generate_arima_columns(coefs)
                )
            else:
                summary_features = coefs
                summary_feature_columns = generate_arima_columns(coefs)
        else:
            if self.use_spike_summary_stats:
                summary_features = torch.stack(
                    (
                        # ampl_next_stage,
                        torch.flatten(num_spikes_simulated),
                        torch.flatten(simulated_interspike_times),
                        avg_spike_min.flatten().T,
                        avg_spike_max.flatten().T,
                    )
                )
                summary_features = torch.cat(
                    (summary_features.T, first_n_spikes), dim=1
                )
                summary_feature_columns.append("Num Spikes")
                summary_feature_columns.append("Interspike Interval")
                summary_feature_columns.append("Avg Min Spike Height")
                summary_feature_columns.append("Avg Max Spike Height")
                for i in range(first_n_spikes.shape[1]):
                    summary_feature_columns.append(f"Spike {i+1} time")

        if self.train_amplitude_frequency:
            amplitude, frequency = self.get_amplitude_frequency(
                V.float(), self.inj_dur, self.inj_start, fs=self.fs
            )
            if summary_features is not None:
                summary_features = torch.cat(
                    (
                        summary_features,
                        amplitude.reshape(-1, 1),
                        frequency.reshape(-1, 1),
                    ),
                    dim=1,
                )
            else:
                summary_features = torch.cat(
                    (amplitude.reshape(-1, 1), frequency.reshape(-1, 1)), dim=1
                )
            summary_feature_columns = summary_feature_columns + [
                "amplitude",
                "frequency",
            ]

        if self.train_I_stats:
            amp_mean, amp_std = DataProcessor.extract_I_stats(I)
            print(amp_mean.reshape(-1,1))
            print(summary_features)
            if summary_features is not None:
                summary_features = torch.cat(
                    (
                        summary_features,
                        amp_mean.reshape(-1, 1),
                        amp_std.reshape(-1, 1)
                    ),
                    dim=1,
                )
            else:
                summary_features = torch.cat(
                    (amp_mean.reshape(-1, 1), amp_std.reshape(-1, 1)), dim=1)

            summary_feature_columns = summary_feature_columns + [
                "mean_amp",
                "std_amp",
            ]
        if self.train_mean_potential:
            mean_potential = self.get_mean_potential(
                V.float(), self.inj_dur, self.inj_start
            )
            if summary_features is not None:
                summary_features = torch.cat(
                    (
                        summary_features,
                        mean_potential.reshape(-1, 1),
                    ),
                    dim=1,
                )
            else:
                summary_features = mean_potential.reshape(-1, 1)

            summary_feature_columns = summary_feature_columns + [
                "mean_potential",
            ]



        if summary_features is None:
            print(
                "You have to have some summary feature turned on (use_spike_summary_stats, train_amplitude_frequency, arima stats) or select a model that doesn't use them. Errors will occur"
            )

        return summary_features, summary_feature_columns, coefs_loaded

    @staticmethod
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
        
        threshold_crossings = torch.diff(V > spike_threshold, dim=1)

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

    def get_amplitude_frequency(traces, inj_dur, inj_start, fs=1000):
        amplitudes = []
        frequencies = []
        for idx in range(traces.shape[0]):
            x = traces[idx].cpu().numpy()[inj_start : inj_start + inj_dur]
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
    
    def get_mean_potential(traces, inj_dur, inj_start):
        mean_potential = traces[:,inj_start:inj_start+inj_dur].mean(dim=1)
        return mean_potential

    def extract_I_stats(I: torch.Tensor):
        return torch.mean(I), torch.std(I)
    

    def calculate_passive_properties(
        self, config: SimulationConfig, parameter_names: list, parameter_values: np.ndarray
    ) -> Tuple[PassiveProperties, np.ndarray]:
        """
        Run simulations to determine passive properties for a given cell parameter set
        """
        gleak_var = (
            config.get("cell", {})
            .get("passive_properties", {})
            .get("leak_conductance_variable")
        )
        eleak_var = (
            config.get("cell", {})
            .get("passive_properties", {})
            .get("leak_reversal_variable")
        )
        props: PassiveProperties = {
            "leak_conductance_variable": gleak_var,
            "leak_reversal_variable": eleak_var,
            "r_in": 0,
            "tau": 0,
            "v_rest": 0,
        }
        tstart = 500
        passive_delay = 500
        passive_amp = -100 / 1e3
        passive_duration = 1000

        tstop = tstart + passive_duration

        passive_tensor = self.simulate(
            passive_amp,
            parameter_names,
            parameter_values,
            i_delay=tstart,
            i_dur=passive_duration,
            tstop=tstop,
            no_ramp=True,
        )
        passive_vec = passive_tensor.cpu().detach().numpy()

        index_v_rest = int(((1000 / h.dt) / 1000 * tstart))
        index_v_final = int(((1000 / h.dt) / 1000 * (tstart + passive_delay)))

        # v rest
        v_rest = passive_vec[index_v_rest]
        v_rest_time = index_v_rest / (1 / h.dt)

        # tau/r_in
        passive_v_final = passive_vec[index_v_final]
        v_final_time = index_v_final / (1 / h.dt)

        v_diff = v_rest - passive_v_final

        v_t_const = v_rest - (v_diff * 0.632)
        # Find index of first occurance where the voltage is less than the time constant for tau
        # index_v_tau = list(filter(lambda i: i < v_t_const, cfir_widget.passive_vec))[0]
        index_v_tau = next(
            x
            for x, val in enumerate(list(passive_vec[index_v_rest:]))
            if val < v_t_const
        )
        time_tau = index_v_tau / ((1000 / h.dt) / 1000)
        tau = time_tau  # / 1000 (in ms)
        r_in = (v_diff) / (0 - passive_amp)  # * 1e6  # MegaOhms -> Ohms

        props["v_rest"] = float(v_rest)
        props["tau"] = float(tau)
        props["r_in"] = float(r_in)

        return props, passive_vec
    
    #---------------------------------------------
    # VOLTAGE TRACE MANIPULATION
    #---------------------------------------------
    def resample_voltage(self, V: torch.Tensor, num_obs: int) -> torch.Tensor:
        if not V.shape[-1] == num_obs:
            print(
                f"resampling traces from shape {V.shape[-1]} to {num_obs} to match target shape"
            )
            resampled_data = []
            for i in range(V.shape[0]):
                resampled_data.append(signal.resample(x=V[i].cpu(), num=num_obs))
            resampled_data = torch.tensor(np.array(resampled_data)).float()
            return resampled_data
        else:
            print("resampling of traces not needed, same shape as target")
            return V