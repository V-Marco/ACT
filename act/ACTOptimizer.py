import numpy as np
import torch
from neuron import h
import os

from act.act_types import PassiveProperties, SimulationConfig
from act.cell_model import CellModel
from act.logger import ACTDummyLogger
from act import utils
from act.DataProcessor import DataProcessor
from typing import Tuple

# ACTOptimizer is the base class for all optimization algorithms
# The methods outlined in this class are divided into categories:
# 1. ABSTRACT METHODS
    # fit() and predict() are abstract methods that must be implemented by subclasses
# 2. LOAD MOST RECENT DATA

class ACTOptimizer:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        set_passive_properties=True,
        cell_override=None,
        ignore_segregation=False,
    ):
        self.config = simulation_config
        self.ignore_segregation = ignore_segregation

        # Initialize standard run
        h.load_file("stdrun.hoc")

        # Initialize the cell
        if cell_override:
            self.cell = cell_override
        else:
            self.cell = CellModel(
                hoc_file=self.config["cell"]["hoc_file"],
                cell_name=self.config["cell"]["name"],
            )
        if set_passive_properties:
            self.cell.set_passive_properties(
                simulation_config["cell"].get("passive_properties")
            )

        # For convenience
        self.update_param_vars()

        # Set the logger
        self.logger = logger
        if self.logger is None:
            self.logger = ACTDummyLogger()

    #------------------------------------
    # ABSTRACT METHODS
    #------------------------------------

    def fit(self, x ,y) -> torch.Tensor:
        raise NotImplementedError
    
    def predict(self, x) -> torch.Tensor:
        raise NotImplementedError

    #------------------------------------
    # LOAD MOST RECENT DATA
    #------------------------------------

    def update_param_vars(self) -> None:
        self.preset_params = {}
        self.params = [
            param["channel"]
            for param in self.config["optimization_parameters"]["params"]
        ]

        if not self.ignore_segregation and self.config["run_mode"] == "segregated":
            self.preset_params = utils.load_preset_params(self.config)
            learned_params = utils.load_learned_params(self.config)
            learned_variability = utils.get_learned_variability(self.config)
            learned_variability_params = utils.get_learned_variability_params(
                self.config
            )

            if (
                len(learned_variability_params) > 0
            ):  # if we specify which ones to unlearn, then we want to remove, otherwise everything gets varied
                learned_params = [
                    p for p in learned_params if p in learned_variability_params
                ]

            if learned_variability > 0:
                if (
                    len(learned_variability_params) > 0
                ):  # if we specify which ones to unlearn, then we want to remove, otherwise everything gets varied
                    learned_params = [
                        p for p in learned_params if p in learned_variability_params
                    ]

                self.preset_params = {
                    p: v
                    for p, v in self.preset_params.items()
                    if p not in learned_params
                }  # we're essentially "un-learning"

            self.num_params = len(self.params) - len(
                self.preset_params
            )  # don't count those that will be set
        else:
            self.num_params = len(self.params)

        self.num_ampl = len(self.config["optimization_parameters"]["amps"])

    def get_params(self):
        # extract only traces that have spikes in them
        self.spiking_only = True
        self.nonsaturated_only = True

        self.model_class = None
        self.learning_rate = 0
        self.weight_decay = 0
        self.num_epochs = 0
        self.use_spike_summary_stats = True
        self.train_amplitude_frequency = False
        self.train_mean_potential = False
        self.train_I_stats = True
        self.segregation_arima_order = None
        self.train_test_split = 0.85
        self.summary_feature_columns = []
        self.learned_variability = 0

        self.inj_dur = self.config["simulation_parameters"].get("h_i_dur")
        self.inj_start = self.config["simulation_parameters"].get("h_i_delay")
        self.fs = (
            self.config["simulation_parameters"].get("h_dt")
            * self.config["optimization_parameters"].get("decimate_factor")
            * 1000
        )

        if self.config["run_mode"] == "segregated":
            self.learning_rate = self.config["segregation"][self.segregation_index].get(
                "learning_rate", 0
            )
            self.weight_decay = self.config["segregation"][self.segregation_index].get(
                "weight_decay", 0
            )
            self.model_class = self.config["segregation"][self.segregation_index].get(
                "model_class", None
            )
            self.num_epochs = self.config["segregation"][self.segregation_index].get(
                "num_epochs", 0
            )
            self.spiking_only = self.config["segregation"][self.segregation_index].get(
                "train_spiking_only", True
            )
            self.nonsaturated_only = self.config["segregation"][self.segregation_index].get(
                "nonsaturated_only", True
            )
            self.use_spike_summary_stats = self.config["segregation"][
                self.segregation_index
            ].get("use_spike_summary_stats", True)
            
            self.train_amplitude_frequency = self.config["segregation"][
                self.segregation_index
            ].get("train_amplitude_frequency", False)
            self.train_mean_potential = self.config["segregation"][
                self.segregation_index
            ].get("train_mean_potential", False)

            self.segregation_arima_order = self.config["segregation"][
                self.segregation_index
            ].get("arima_order", None)
            self.train_test_split = self.config["segregation"][self.segregation_index].get(
                "train_test_split", 0.99
            )
            self.learned_variability = self.config["segregation"][
                self.segregation_index
            ].get("learned_variability", 0)
            self.inj_start = self.config["segregation"][self.segregation_index].get(
                "h_i_delay", self.inj_start
            )
            self.inj_dur = self.config["segregation"][self.segregation_index].get(
                "h_i_dur", self.inj_dur
            )

        if not self.num_epochs:
            self.num_epochs = self.config["optimization_parameters"].get("num_epochs")

        self.num_first_spikes = self.config["summary_features"].get("num_first_spikes", 20)
        print(f"Extracting first {self.num_first_spikes} spikes for summary features")

    def get_voltage_traces(self, target_V):
        # Get voltage with characteristics similar to target

        (
            simulated_V_for_next_stage,
            param_samples_for_next_stage,
            ampl_next_stage,
        ) = (
            None,
            None,
            None,
        )

        n_slices = (
            self.config["optimization_parameters"]
            .get("parametric_distribution", {})
            .get("n_slices", 0)
        )
        simulations_per_amp = (
            self.config["optimization_parameters"]
            .get("parametric_distribution", {})
            .get("simulations_per_amp", 1)
        )
        # we should do a parametric sampling of parameters to train network
        # check to see if parametric traces can be loaded
        (
            simulated_V_dist,
            param_samples_dist,
            simulated_amps,
        ) = utils.load_parametric_traces(self.config)
        # REMOVE TODO for testing quickly
        # num_subs = 3000
        # subset_target_ind = np.random.default_rng().choice(len(simulated_V_dist), size=num_subs, replace=False).tolist()
        # simulated_V_dist = simulated_V_dist[subset_target_ind]
        # param_samples_dist = param_samples_dist[subset_target_ind]
        # simulated_amps = simulated_amps[subset_target_ind]

        if simulated_V_dist is None:
            if n_slices > 1:
                print(
                    f"n_slices variable set, but no traces have been generated previously"
                )
                print(f"Generate parametric traces prior to running")
                """
                (
                    simulated_V_dist,
                    param_samples_dist,
                    simulated_amps,
                ) = self.get_parametric_distribution(n_slices, simulations_per_amp)
                """
            else:
                print(
                    f"Parametric distribution parameters 'n_slices' not set, skipping."
                )

        if simulated_V_dist is not None:
            if simulated_V_for_next_stage is not None:
                simulated_V_for_next_stage = torch.cat(
                    (simulated_V_for_next_stage, simulated_V_dist),
                )
                param_samples_for_next_stage = torch.cat(
                    (param_samples_for_next_stage, param_samples_dist),
                )
                ampl_next_stage = torch.cat((ampl_next_stage, simulated_amps))
            else:
                simulated_V_for_next_stage = simulated_V_dist
                param_samples_for_next_stage = param_samples_dist
                ampl_next_stage = simulated_amps
        else:
            print(f"Parametric distribution parameters not applied.")

        if self.spiking_only:
            (
                simulated_V_for_next_stage,
                param_samples_for_next_stage,
                ampl_next_stage,
                spiking_ind,
            ) = utils.extract_spiking_traces(
                simulated_V_for_next_stage,
                param_samples_for_next_stage,
                ampl_next_stage,
            )
        if self.nonsaturated_only:
            drop_dur = 200
            end_of_drop = 750
            start_of_drop = end_of_drop - drop_dur
            threshold_drop = -50

            traces_end = simulated_V_for_next_stage[:, start_of_drop:end_of_drop].mean(
                dim=1
            )
            bad_ind = (traces_end > threshold_drop).nonzero().flatten().tolist()
            nonsaturated_ind = (
                (traces_end <= threshold_drop).nonzero().flatten().tolist()
            )

            print(
                f"Dropping {len(bad_ind)} traces, mean value >{threshold_drop} between {start_of_drop}:{end_of_drop}ms"
            )
            simulated_V_for_next_stage = simulated_V_for_next_stage[nonsaturated_ind]
            param_samples_for_next_stage = param_samples_for_next_stage[
                nonsaturated_ind
            ]
            ampl_next_stage = ampl_next_stage[nonsaturated_ind]

        return (
            simulated_V_for_next_stage,
            ampl_next_stage, 
            spiking_ind,
            nonsaturated_ind
            )
    
    def get_summary_features(self, V: torch.Tensor, I: torch.Tensor, spiking_ind, nonsaturated_ind):
        
        # Extract spike summary features
        (   num_spikes_simulated,
            simulated_interspike_times,
            first_n_spikes, 
            avg_spike_min, 
            avg_spike_max
        ) = DataProcessor.extract_spike_features(V)

        coefs_loaded = False
        if os.path.exists("output/arima_stats.json"):
            coefs_loaded = True
            coefs = DataProcessor.load_arima_coefs(
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
            amplitude, frequency = utils.get_amplitude_frequency(
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
            mean_potential = utils.get_mean_potential(
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
    
    #------------------------------------
    # Temporary simulate (will be replaced with Simulator class)
    #------------------------------------

    def simulate(
        self,
        amp: float,
        parameter_names: list,
        parameter_values: np.ndarray,
        i_dur: float = 0,
        i_delay: float = 0,
        tstop=None,
        no_ramp=False,
        cut_ramp=False,
        ramp_time=0,
        ramp_splits=1, 
    ) -> torch.Tensor:
        h.dt = self.config["simulation_parameters"]["h_dt"]
        h.steps_per_ms = 1 / h.dt
        h.v_init = self.config["simulation_parameters"]["h_v_init"]
        h.celsius = self.config["simulation_parameters"].get("h_celsius", 31.0)
        print(f"h.celsius set to {h.celsius}")

        self.cell.set_parameters(parameter_names, parameter_values)
        segregation_index = utils.get_segregation_index(self.config)
        if not i_dur:
            i_dur = self.config["simulation_parameters"]["h_i_dur"]
            if self.config["run_mode"] == "segregated" and not self.ignore_segregation:
                i_dur = self.config["segregation"][segregation_index].get(
                    "h_i_dur", i_dur
                )
        if not i_delay:
            i_delay = self.config["simulation_parameters"]["h_i_delay"]
            if self.config["run_mode"] == "segregated" and not self.ignore_segregation:
                i_delay = self.config["segregation"][segregation_index].get(
                    "h_i_delay", i_delay
                )

        if self.config["run_mode"] == "segregated" and not no_ramp and not self.ignore_segregation:
            ramp_time = self.config["segregation"][segregation_index].get(
                "ramp_time", 0
            )
            ramp_splits = self.config["segregation"][segregation_index].get(
                "ramp_splits", 1
            )

        tstop_config = self.config["simulation_parameters"]["h_tstop"]
        if (
            self.config["run_mode"] == "segregated" and not self.ignore_segregation
        ):  # sometimes segregated modules have different params
            tstop_config = self.config["segregation"][segregation_index].get(
                "h_tstop", tstop_config
            )

        h.tstop = (tstop or tstop_config) + ramp_time

        self.cell.apply_current_injection(
            amp, i_dur, i_delay, ramp_time=ramp_time, ramp_splits=ramp_splits
        )

        h.run()
        start_idx = int((ramp_time) / h.dt)  # usually 0, this will remove ramp_time
        trace = torch.Tensor(self.cell.Vm.as_numpy())
        if cut_ramp:
            return trace[start_idx:-1]
        else:
            return trace[:-1]  # [start_idx:-1]

    