import numpy as np
import torch
from neuron import h
from scipy.signal import resample
import os
import copy
import tqdm

from act.act_types import PassiveProperties, SimulationConfig
from act.cell_model import CellModel
from act.logger import ACTDummyLogger
from act.models import (
    BranchingNet,
    EmbeddingNet,
    SimpleNet,
    ConvolutionEmbeddingNet,
    SummaryNet,
)
from act import utils
from sklearn.ensemble import RandomForestRegressor


class TorchStandardScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x -= self.mean
            x /= self.std + 1e-7
        return x


class TorchMinMaxColScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.min = x.min(dim=0)[0]
        self.max = x.max(dim=0)[0]
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x = (x - self.min + 1e-7) / (self.max + 1e-7 - self.min)
        return x


class TorchMinMaxScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x = (x - self.min + 1e-7) / (self.max - self.min)
        return x


class ACTOptimizer:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        set_passive_properties=True,
        cell_override=None,
    ):
        self.config = simulation_config

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

    def optimize(self, target_V: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def optimize_with_segregation(
        self, target_V: torch.Tensor, segregate_by: str
    ) -> torch.Tensor:
        raise NotImplementedError

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

    def simulate(
        self,
        amp: float,
        parameter_names: list,
        parameter_values: np.ndarray,
        i_dur: float = 0,
        i_delay: float = 0,
    ) -> torch.Tensor:
        h.dt = self.config["simulation_parameters"]["h_dt"]
        h.steps_per_ms = 1 / h.dt
        h.tstop = self.config["simulation_parameters"]["h_tstop"]
        h.v_init = self.config["simulation_parameters"]["h_v_init"]

        self.cell.set_parameters(parameter_names, parameter_values)
        if not i_dur:
            i_dur = self.config["simulation_parameters"]["h_i_dur"]
        if not i_delay:
            i_delay = self.config["simulation_parameters"]["h_i_delay"]
        self.cell.apply_current_injection(
            amp,
            i_dur,
            i_delay,
        )

        h.run()

        return torch.Tensor(self.cell.Vm.as_numpy()[:-1])

    def calculate_passive_properties(
        self, parameter_names: list, parameter_values: np.ndarray
    ) -> (PassiveProperties, np.ndarray):
        """
        Run simulations to determine passive properties for a given cell parameter set
        """
        gleak_var = (
            self.config.get("cell", {})
            .get("passive_properties", {})
            .get("leak_conductance_variable")
        )
        eleak_var = (
            self.config.get("cell", {})
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

        passive_tensor = self.simulate(
            passive_amp,
            parameter_names,
            parameter_values,
            i_delay=tstart,
            i_dur=passive_duration,
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

    def update_param_vars(self) -> None:
        self.num_ampl = len(self.config["optimization_parameters"]["amps"])
        self.num_params = len(self.config["optimization_parameters"]["params"])


class GeneralACTOptimizer(ACTOptimizer):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
    ):
        super().__init__(
            simulation_config=simulation_config,
            logger=logger,
        )

        self.model = None
        self.model_pool = None
        self.reg = None  # regressor for random forest
        self.init_random_forest()

        self.voltage_data_scaler = TorchMinMaxScaler()
        self.summary_feature_scaler = TorchMinMaxColScaler()
        self.target_param_scaler = TorchStandardScaler()

    def init_random_forest(self):
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "warm_start": True,
            "oob_score": True,
            "random_state": 42,
        }
        self.reg = RandomForestRegressor(**params)

    def train_random_forest(self, X_train, y_train):
        self.reg.fit(X_train, y_train)

    def predict_random_forest(self, X_test):
        y_pred = self.reg.predict(X_test)
        return y_pred

    def optimize(self, target_V: torch.Tensor) -> torch.Tensor:
        # Get voltage with characteristics similar to target
        if not self.config["optimization_parameters"]["skip_match_voltage"]:
            (
                simulated_V_for_next_stage,
                param_samples_for_next_stage,
                ampl_next_stage,
            ) = self.match_voltage(target_V)
            simulated_V_for_next_stage = utils.apply_decimate_factor(
                self.config, simulated_V_for_next_stage
            )
        else:
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

        # extract only traces that have spikes in them
        # (
        #    simulated_V_for_next_stage,
        #    param_samples_for_next_stage,
        #    ampl_next_stage,
        # ) = utils.extract_spiking_traces(
        #    simulated_V_for_next_stage, param_samples_for_next_stage, ampl_next_stage
        # )
        (
            num_spikes_simulated,
            simulated_interspike_times,
        ) = self.extract_summary_features(simulated_V_for_next_stage)
        # spike_stats
        (first_n_spikes, avg_spike_min, avg_spike_max) = utils.spike_stats(
            simulated_V_for_next_stage
        )
        coefs_loaded = False
        if os.path.exists("output/arima_stats.json"):
            coefs_loaded = True
            coefs = utils.load_arima_coefs(
                input_file="output/arima_stats.json"
            )  # [subset_target_ind] # TODO REMOVE for testing quickly

        if coefs_loaded:
            summary_features = torch.stack(
                (
                    ampl_next_stage,
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )
            summary_features = torch.cat(
                (summary_features.T, first_n_spikes, coefs), dim=1
            )
        else:
            summary_features = torch.stack(
                (
                    ampl_next_stage,
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )
            summary_features = torch.cat((summary_features.T, first_n_spikes), dim=1)

        self.model = self.init_nn_model(
            in_channels=target_V.shape[1],
            out_channels=self.num_params,
            summary_features=summary_features,
        )

        # Resample to match the length of target data
        resampled_data = self.resample_voltage(
            simulated_V_for_next_stage, target_V.shape[1]
        )

        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]

        # remove any remaining nan values
        summary_features[torch.isnan(summary_features)] = 0

        # Train model
        train_stats = self.train_model(
            resampled_data.float(),
            param_samples_for_next_stage,
            lows,
            highs,
            summary_features=summary_features,
        )

        # Predict and take max across ci to prevent underestimating
        (
            num_spikes_simulated,
            simulated_interspike_times,
        ) = self.extract_summary_features(target_V.float())
        (first_n_spikes, avg_spike_min, avg_spike_max) = utils.spike_stats(
            target_V.float()
        )
        ampl_target = torch.tensor(self.config["optimization_parameters"]["amps"])

        if coefs_loaded:
            arima_order = (10, 0, 10)
            if self.config.get("summary_features", {}).get("arima_order"):
                arima_order = tuple(self.config["summary_features"]["arima_order"])
            print(f"ARIMA order set to {arima_order}")
            total_arima_vals = 2 + arima_order[0] + arima_order[1]
            coefs = []
            for data in target_V.cpu().detach().numpy():
                try:
                    c = utils.get_arima_coefs(data, order=arima_order)
                except:
                    print("ERROR calculating coefs, setting all to 0")
                    c = np.zeros(total_arima_vals)
                coefs.append(c)
            coefs = torch.tensor(coefs)

            target_summary_features = torch.stack(
                (
                    ampl_target,
                    torch.flatten(num_spikes_simulated),
                    torch.flatten(simulated_interspike_times),
                    avg_spike_min.flatten().T,
                    avg_spike_max.flatten().T,
                )
            )
            target_summary_features = torch.cat(
                (target_summary_features.T, first_n_spikes, coefs), dim=1
            )
        else:
            target_summary_features = torch.stack(
                (
                    ampl_target,
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

        # remove any remaining nan values
        target_summary_features[torch.isnan(target_summary_features)] = 0

        predictions = self.predict_with_model(
            target_V.float(), lows, highs, target_summary_features.float()
        )
        # predictions = torch.max(predictions, dim=0).values

        return predictions, train_stats

    def optimize_with_segregation(
        self, target_V: torch.Tensor, segregate_by: str = "voltage"
    ) -> torch.Tensor:
        if segregate_by == "voltage":
            cut_func = self.cut_voltage_region
        elif segregate_by == "time":
            cut_func = self.cut_time_region
        else:
            raise ValueError("segregate_by must be either 'voltage' or 'time'.")

        # Get voltage with characteristics similar to target
        (
            simulated_V_for_next_stage,
            param_samples_for_next_stage,
            ampl_next_stage,
        ) = self.match_voltage(target_V)

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
        if n_slices > 1:
            (
                simulated_V_dist,
                param_samples_dist,
                simulated_amps,
            ) = self.get_parametric_distribution(n_slices, simulations_per_amp)
            simulated_V_for_next_stage = torch.cat(
                (simulated_V_for_next_stage, simulated_V_dist)
            )
            param_samples_for_next_stage = torch.cat(
                (param_samples_for_next_stage, param_samples_dist)
            )
            ampl_next_stage = torch.cat((ampl_next_stage, simulated_amps))
        else:
            print(f"Parametric distribution parameters 'n_slices' not set, skipping.")

        (
            num_spikes_simulated,
            simulated_interspike_times,
        ) = self.extract_summary_features(simulated_V_for_next_stage)

        summary_features = torch.stack(
            (
                ampl_next_stage,
                torch.flatten(num_spikes_simulated),
                torch.flatten(simulated_interspike_times),
            )
        )
        summary_features = torch.transpose(summary_features, 0, 1).float()

        # Resample to match the length of target data
        resampled_data = self.resample_voltage(
            simulated_V_for_next_stage, target_V.shape[1]
        )

        # Store original params to restore later
        orig_params = {
            p["channel"]: p for p in self.config["optimization_parameters"]["params"]
        }

        self.model_pool = []
        prediction_pool = []

        for segregation in self.config["segregation"]:
            seg_params = segregation["params"]
            bounds = segregation[segregate_by]

            # Set the parameters
            param_ind = [
                i for i, (c, _) in enumerate(orig_params.items()) if c in seg_params
            ]

            lows = [
                p["low"] for channel, p in orig_params.items() if channel in seg_params
            ]
            highs = [
                p["high"] for channel, p in orig_params.items() if channel in seg_params
            ]
            self.num_params = len(seg_params)

            # Cut the required segment
            cut_target_V = cut_func(resampled_data, bounds)

            # Train a model for this segment
            self.model = self.init_nn_model(
                in_channels=cut_target_V.shape[1],
                out_channels=self.num_params,
                summary_features=summary_features,
            )
            self.train_model(
                cut_target_V.float(),
                param_samples_for_next_stage[:, param_ind],
                lows=lows,
                highs=highs,
                summary_features=summary_features,
            )

            # Predict and take max across ci to prevent underestimating
            predictions = self.predict_with_model(
                cut_target_V, lows, highs, summary_features
            )
            predictions = torch.max(predictions, dim=0).values

            # Update cell model
            self.cell.set_parameters(seg_params, predictions)

            # Update the pools
            prediction_pool.append(predictions)
            self.model_pool.append(self.model)

        prediction_pool = torch.cat(prediction_pool, dim=0)

        return prediction_pool

    def init_nn_model(
        self, in_channels: int, out_channels: int, summary_features
    ) -> torch.nn.Sequential:
        # ModelClass = SimpleNet
        # ModelClass = BranchingNet
        # ModelClass = EmbeddingNet
        ModelClass = ConvolutionEmbeddingNet
        # ModelClass = SummaryNet
        model = ModelClass(in_channels, out_channels, summary_features)
        return model

    def train_model(
        self,
        voltage_data: torch.Tensor,
        target_params: torch.Tensor,
        lows,
        highs,
        summary_features,
        train_test_split=0.9,
        batch_size=8,
    ) -> None:
        optim = torch.optim.Adam(self.model.parameters(), lr=2e-5, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()  # torch.nn.functional.l1_loss

        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        self.logger.info("Training a model with SGD optimizer and lr = 1e-8.")
        self.logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # self.model.train(True)

        stats = {
            "train_loss_batches": [],
            "train_loss": [],
            "test_loss": [],
            "train_size": 0,
            "test_size": 0,
        }

        # shuffle the training data
        indexes = torch.randperm(voltage_data.shape[0])
        split_point = int(voltage_data.shape[0] * train_test_split)

        train_ind = indexes[:split_point]
        test_ind = indexes[split_point:]
        stats["train_size"] = len(train_ind)
        stats["test_size"] = len(test_ind)

        voltage_data_train = voltage_data[train_ind]
        voltage_data_test = voltage_data[test_ind]

        summary_features_train = summary_features[train_ind]
        summary_features_test = summary_features[test_ind]

        target_params_train = target_params[train_ind]
        target_params_test = target_params[test_ind]

        # Fit the training data, transform both train and test.
        # The fit is not applied to original dataset due to the possibility of data leakage
        self.voltage_data_scaler.fit(voltage_data_train)
        self.summary_feature_scaler.fit(summary_features_train)

        voltage_data_train = self.voltage_data_scaler.transform(voltage_data_train)
        voltage_data_test = self.voltage_data_scaler.transform(voltage_data_test)
        summary_features_train = self.summary_feature_scaler.transform(
            summary_features_train
        )
        summary_features_test = self.summary_feature_scaler.transform(
            summary_features_test
        )
        batch_start = torch.arange(0, len(voltage_data_train), batch_size)

        # Hold the best model
        best_mse = np.inf  # init to infinity
        best_weights = None

        num_epochs = self.config["optimization_parameters"]["num_epochs"]

        for epoch in range(num_epochs):
            self.model.train()
            with tqdm.tqdm(
                batch_start, unit="batch", mininterval=0, disable=False
            ) as bar:
                bar.set_description(f"Epoch {epoch}/{num_epochs}")
                for start in bar:
                    voltage_data_batch = voltage_data_train[start : start + batch_size]
                    summary_features_batch = summary_features_train[
                        start : start + batch_size
                    ]
                    target_params_batch = target_params_train[
                        start : start + batch_size
                    ]

                    # forward pass

                    pred = (
                        self.model(voltage_data_batch, summary_features_batch)
                        * (sigmoid_maxs - sigmoid_mins)
                        + sigmoid_mins
                    )
                    loss = loss_fn(pred, target_params_batch)
                    stats["train_loss_batches"].append(
                        float(loss.cpu().detach().numpy())
                    )

                    # backward pass
                    optim.zero_grad()  # this line is new, wasn't in last round
                    loss.backward()

                    # update weights
                    optim.step()

                    # print process
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.model.eval()
            y_pred = (
                self.model(voltage_data_train, summary_features_train)
                * (sigmoid_maxs - sigmoid_mins)
                + sigmoid_mins
            )
            mse = loss_fn(y_pred, target_params_train)
            mse = float(mse)
            stats["train_loss"].append(mse)

            y_pred = (
                self.model(voltage_data_test, summary_features_test)
                * (sigmoid_maxs - sigmoid_mins)
                + sigmoid_mins
            )
            mse = loss_fn(y_pred, target_params_test)
            mse = float(mse)
            stats["test_loss"].append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.model.state_dict())

        # restore model and return best accuracy
        self.model.load_state_dict(best_weights)

        return stats

    def predict_with_model(
        self, target_V: torch.Tensor, lows, highs, summary_features
    ) -> torch.Tensor:
        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        self.model.eval()
        outs = []
        target_V_fit = self.voltage_data_scaler.transform(target_V)
        summary_features_fit = self.summary_feature_scaler.transform(summary_features)
        for i in range(target_V.shape[0]):
            out = (
                self.model(
                    target_V_fit[i].reshape(1, -1),
                    summary_features_fit[i].reshape(1, -1),
                )
                * (sigmoid_maxs - sigmoid_mins)
                + sigmoid_mins
            )
            outs.append(out.reshape(1, -1))

        return torch.cat(outs, dim=0)

    def get_parametric_distribution(self, n_slices, simulations_per_amp) -> tuple:
        params = [
            p["channel"] for p in self.config["optimization_parameters"]["params"]
        ]
        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]
        steps = int(
            self.config["simulation_parameters"]["h_tstop"]
            / self.config["simulation_parameters"]["h_dt"]
        )

        param_samples_for_next_stage = torch.zeros(
            (self.num_ampl, simulations_per_amp, self.num_params)
        )
        simulated_V = torch.zeros(
            (
                self.num_ampl,
                simulations_per_amp,
                steps,
            )
        )

        param_dist = np.array(
            [
                np.arange(low, high, (high - low) / n_slices)
                for low, high in zip(lows, highs)
            ]
        ).T

        amps = self.config["optimization_parameters"]["amps"]
        print(
            f"Sampling parameter space... this may take a while. {len(amps)} amps * {simulations_per_amp} simulations per amp = {len(amps) * simulations_per_amp}"
        )
        s_amps = []
        for amp_ind, amp in enumerate(amps):
            print(
                f"    Generating {simulations_per_amp} simulations_per_amp at {amp:.2f} amps."
            )
            for slice_ind in range(simulations_per_amp):
                # For each current injection amplitude, sample random parameters
                param_inds = np.random.randint(
                    0, n_slices, len(params)
                )  # get random indices for our params
                param_sample = param_dist.T[
                    range(len(params)), param_inds
                ]  # select the params
                simulated_V[amp_ind][slice_ind] = self.simulate(
                    amp, params, param_sample
                )
                param_samples_for_next_stage[amp_ind][slice_ind] = torch.Tensor(
                    param_sample
                )

                s_amps.append(amp)

        s_v = torch.flatten(simulated_V).reshape(
            [len(amps) * simulations_per_amp, steps]
        )
        s_param = torch.flatten(param_samples_for_next_stage).reshape(
            [len(amps) * simulations_per_amp, len(params)]
        )
        s_amps = torch.tensor(s_amps)

        return s_v, s_param, s_amps

    def match_voltage(self, target_V: torch.Tensor) -> tuple:
        # Get target voltage summary features
        num_target_spikes, target_interspike_times = self.extract_summary_features(
            target_V
        )  # (N x 1)

        # Set a vector of non-correspondence to target features
        non_corresp = torch.ones(self.num_ampl)

        simulated_V_for_next_stage = torch.zeros(
            (
                self.num_ampl,
                int(
                    self.config["simulation_parameters"]["h_tstop"]
                    / self.config["simulation_parameters"]["h_dt"]
                ),
            )
        )
        param_samples_for_next_stage = torch.zeros((self.num_ampl, self.num_params))

        self.logger.info(
            f"Matching {self.config['optimization_parameters']['num_amps_to_match']} amplitudes."
        )

        params = [
            p["channel"] for p in self.config["optimization_parameters"]["params"]
        ]
        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]

        while torch.sum(non_corresp) > (
            self.num_ampl - self.config["optimization_parameters"]["num_amps_to_match"]
        ):
            # Simulate with these samples
            simulated_V = torch.zeros(
                (
                    self.num_ampl,
                    int(
                        self.config["simulation_parameters"]["h_tstop"]
                        / self.config["simulation_parameters"]["h_dt"]
                    ),
                )
            )
            for ind, amp in enumerate(self.config["optimization_parameters"]["amps"]):
                if non_corresp[ind] == 1:
                    # For each current injection amplitude, sample random parameters
                    param_samples = np.random.uniform(low=lows, high=highs)
                    simulated_V[ind] = self.simulate(amp, params, param_samples)
                    param_samples_for_next_stage[ind] = torch.Tensor(param_samples)

            # Compute summary features for simulated data
            (
                num_spikes_simulated,
                simulated_interspike_times,
            ) = self.extract_summary_features(simulated_V)

            # Save those voltage traces which match the target
            cond_spikes = self.get_match_condition(
                num_target_spikes,
                num_spikes_simulated,
                self.config["summary_features"]["mc_num_spikes"],
            )
            cond_times = self.get_match_condition(
                target_interspike_times,
                simulated_interspike_times,
                self.config["summary_features"]["mc_interspike_time"],
            )
            cond_min = self.get_match_condition(
                torch.min(target_V, dim=1).values,
                torch.min(simulated_V, dim=1).values,
                self.config["summary_features"]["mc_min_v"],
            )
            cond_mean = self.get_match_condition(
                torch.mean(target_V, dim=1),
                torch.mean(simulated_V, dim=1),
                self.config["summary_features"]["mc_mean_v"],
            )
            cond_max = self.get_match_condition(
                torch.max(target_V, dim=1).values,
                torch.max(simulated_V, dim=1).values,
                self.config["summary_features"]["mc_max_v"],
            )

            cond = cond_spikes & cond_times & cond_mean & cond_min & cond_max
            simulated_V_for_next_stage[cond] = simulated_V[cond]
            non_corresp[cond] = 0
            self.logger.info(
                f"Total amplitudes matched: {int(torch.sum(1 - non_corresp))}/{self.config['optimization_parameters']['num_amps_to_match']}."
            )

        simulated_V_for_next_stage = simulated_V_for_next_stage[non_corresp == 0]
        param_samples_for_next_stage = param_samples_for_next_stage[non_corresp == 0]
        inds_of_next_stage = np.where(non_corresp.cpu().detach().numpy() == 0)[
            0
        ].tolist()
        ampl_next_stage = torch.tensor(
            np.array(self.config["optimization_parameters"]["amps"])[inds_of_next_stage]
        )

        self.logger.info(f"Matched amplitudes: {ampl_next_stage}")

        return simulated_V_for_next_stage, param_samples_for_next_stage, ampl_next_stage

    def get_match_condition(
        self, target: torch.Tensor, simulated: torch.Tensor, threshold: float
    ):
        return (torch.abs(target - simulated) < threshold).flatten()

    def extract_summary_features(self, V: torch.Tensor) -> tuple:
        threshold_crossings = torch.diff(
            V > self.config["summary_features"]["spike_threshold"], dim=1
        )
        num_spikes = torch.round(torch.sum(threshold_crossings, dim=1) * 0.5)

        interspike_times = torch.zeros((V.shape[0], 1))
        for i in range(threshold_crossings.shape[0]):
            interspike_times[i, :] = torch.mean(
                torch.diff(
                    torch.arange(threshold_crossings.shape[1])[
                        threshold_crossings[i, :]
                    ]
                ).float()
            )
        interspike_times[torch.isnan(interspike_times)] = 0

        return num_spikes, interspike_times
