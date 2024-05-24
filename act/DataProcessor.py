import os
import json
from io import StringIO

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import resample
import torch
import timeout_decorator
import multiprocessing as mp
from tqdm import tqdm

from neuron import h
from act.act_types import PassiveProperties, SimulationConfig
from act import utils
from typing import Tuple

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
        traces = utils.load_parametric_traces(selected_config)
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

    @staticmethod
    def extract_summary_features(V: torch.Tensor, spike_threshold=0) -> tuple:
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
        return num_spikes, interspike_times
    
    def extract_target_v_summary_features(
            self, target_V, config: SimulationConfig, 
            segregation_arima_order, fs, inj_dur, inj_start,
            use_spike_summary_stats=True, train_amplitude_frequency=False, train_mean_potential=False
            ):
        # Predict and take max across ci to prevent underestimating
        (
            num_spikes_simulated,
            simulated_interspike_times,
        ) = self.extract_summary_features(target_V.float())
        (first_n_spikes, avg_spike_min, avg_spike_max) = utils.spike_stats(
            target_V.float(), n_spikes=self.num_first_spikes
        )
        ampl_target = torch.tensor(self.config["optimization_parameters"]["amps"])
        target_summary_features = None

        coefs_loaded = False
        if os.path.exists("output/arima_stats.json"):
            coefs_loaded = True
            coefs = utils.load_arima_coefs(
                input_file="output/arima_stats.json"
            )  # [subset_target_ind] # TODO REMOVE for testing quickly

        if coefs_loaded:
            arima_order = (10, 0, 10)
            if config.get("summary_features", {}).get("arima_order"):
                arima_order = tuple(config["summary_features"]["arima_order"])
            if segregation_arima_order:
                arima_order = segregation_arima_order
            print(f"ARIMA order set to {arima_order}")
            total_arima_vals = 2 + arima_order[0] + arima_order[1]
            coefs = []
            for data in target_V.cpu().detach().numpy():
                try:
                    c = self.get_arima_coefs(data, order=arima_order)
                except:
                    print("ERROR calculating coefs, setting all to 0")
                    c = np.zeros(total_arima_vals)
                coefs.append(c)
            coefs = torch.tensor(coefs)

            target_summary_features = torch.stack(
                (
                    # ampl_target,
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )
            if use_spike_summary_stats:
                target_summary_features = torch.cat(
                    (target_summary_features.T, first_n_spikes, coefs), dim=1
                )
            else:
                target_summary_features = coefs
        else:
            if use_spike_summary_stats:
                target_summary_features = torch.stack(
                    (
                        # ampl_target,
                        torch.flatten(num_spikes_simulated),
                        torch.flatten(simulated_interspike_times),
                        avg_spike_min.flatten().T,
                        avg_spike_max.flatten().T,
                    )
                )
                target_summary_features = torch.cat(
                    (
                        target_summary_features.T,
                        first_n_spikes,
                    ),
                    dim=1,
                )
        if train_amplitude_frequency:
            target_amplitude, target_frequency = utils.get_amplitude_frequency(
                target_V.float(), inj_dur, inj_start, fs=fs
            )
            if target_summary_features is not None:
                target_summary_features = torch.cat(
                    (
                        target_summary_features,
                        target_amplitude.reshape(-1, 1),
                        target_frequency.reshape(-1, 1),
                    ),
                    dim=1,
                )
            else:
                target_summary_features = torch.cat(
                    (target_amplitude.reshape(-1, 1), target_frequency.reshape(-1, 1)),
                    dim=1,
                )
        if train_mean_potential:
            target_mean_potential = utils.get_mean_potential(
                target_V.float(), inj_dur, inj_start
            )
            if target_summary_features is not None:
                target_summary_features = torch.cat(
                    (
                        target_summary_features,
                        target_mean_potential.reshape(-1, 1),
                    ),
                    dim=1,
                )
            else:
                target_summary_features = target_mean_potential.reshape(-1, 1)

        # remove any remaining nan values
        target_summary_features[torch.isnan(target_summary_features)] = 0

        return target_summary_features
    

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
                resampled_data.append(resample(x=V[i].cpu(), num=num_obs))
            resampled_data = torch.tensor(np.array(resampled_data)).float()
            return resampled_data
        else:
            print("resampling of traces not needed, same shape as target")
            return V

    def cut_voltage_region(
        self, target_V: torch.Tensor, voltage_bounds: list
    ) -> torch.Tensor:
        cut_target_V = torch.zeros_like(target_V)

        for i in range(cut_target_V.shape[0]):
            trace = target_V[i]
            trace = trace[(trace >= voltage_bounds[0]) & (trace < voltage_bounds[1])]
            cut_target_V[i] = torch.tensor(
                resample(x=trace, num=target_V.shape[1])
            ).float()

        return cut_target_V

    def cut_time_region(
        self, target_V: torch.Tensor, time_bounds: tuple
    ) -> torch.Tensor:
        cut_target_V = torch.zeros_like(target_V)

        for i in range(cut_target_V.shape[0]):
            trace = target_V[i, time_bounds[0] : time_bounds[1]]
            cut_target_V[i] = torch.tensor(
                resample(x=trace, num=target_V.shape[1])
            ).float()

        return cut_target_V