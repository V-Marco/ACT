import numpy as np
import torch
from neuron import h
from scipy.signal import resample

from act.act_types import PassiveProperties, SimulationConfig
from act.cell_model import CellModel
from act.logger import ACTDummyLogger

from act.models import EmbeddingNet


class ACTOptimizer:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        reset_cell_params_to_lower_bounds_on_init: bool = True,
        set_passive_properties=True,
    ):
        self.config = simulation_config

        # Initialize standard run
        h.load_file("stdrun.hoc")

        # Initialize the cell
        self.cell = CellModel(
            hoc_file=self.config["cell"]["hoc_file"],
            cell_name=self.config["cell"]["name"],
        )
        if set_passive_properties:
            self.cell.set_passive_properties(
                simulation_config["cell"].get("passive_properties")
            )

        if reset_cell_params_to_lower_bounds_on_init:
            params = [
                p["channel"] for p in self.config["optimization_parameters"]["params"]
            ]
            lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
            self.cell.set_parameters(params, lows)

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
        passive_vec = passive_tensor.detach().numpy()

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
        resampled_data = []
        for i in range(V.shape[0]):
            resampled_data.append(resample(x=V[i], num=num_obs))
        resampled_data = torch.tensor(np.array(resampled_data)).float()
        return resampled_data

    def update_param_vars(self) -> None:
        self.num_ampl = len(self.config["optimization_parameters"]["amps"])
        self.num_params = len(self.config["optimization_parameters"]["params"])


class GeneralACTOptimizer(ACTOptimizer):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        reset_cell_params_to_lower_bounds_on_init: bool = True,
    ):
        super().__init__(
            simulation_config=simulation_config,
            logger=logger,
            reset_cell_params_to_lower_bounds_on_init=reset_cell_params_to_lower_bounds_on_init,
        )

        self.model = None
        self.model_pool = None

    def optimize(self, target_V: torch.Tensor) -> torch.Tensor:
        # Get voltage with characteristics similar to target
        simulated_V_for_next_stage, param_samples_for_next_stage = self.match_voltage(
            target_V
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
        if n_slices > 1:
            simulated_V_dist, param_samples_dist, simulated_amps = self.get_parametric_distribution(
                n_slices, simulations_per_amp
            )
            simulated_V_for_next_stage = torch.cat(
                (simulated_V_for_next_stage, simulated_V_dist), dim=1
            )
            param_samples_for_next_stage = torch.cat(
                (param_samples_for_next_stage, param_samples_dist), dim=1
            )
        else:
            print(f"Parametric distribution parameters 'n_slices' not set, skipping.")

        self.model = self.init_nn_model(
            in_channels=target_V.shape[1], out_channels=self.num_params, num_summary_features = 2,
        )

        # Resample to match the length of target data
        resampled_data = self.resample_voltage(
            simulated_V_for_next_stage, target_V.shape[1]
        )

        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]

        # Train model
        self.train_model(resampled_data, param_samples_for_next_stage, lows, highs)

        # Predict and take max across ci to prevent underestimating
        predictions = self.predict_with_model(target_V, lows, highs)
        predictions = torch.max(predictions, dim=0).values

        return predictions

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
        simulated_V_for_next_stage, param_samples_for_next_stage = self.match_voltage(
            target_V
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
        if n_slices > 1:
            simulated_V_dist, param_samples_dist, simulated_amps = self.get_parametric_distribution(
                n_slices, simulations_per_amp
            )
            simulated_V_for_next_stage = torch.cat(
                (simulated_V_for_next_stage, simulated_V_dist)
            )
            param_samples_for_next_stage = torch.cat(
                (param_samples_for_next_stage, param_samples_dist)
            )
        else:
            print(f"Parametric distribution parameters 'n_slices' not set, skipping.")

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
                in_channels=cut_target_V.shape[1], out_channels=self.num_params
            )
            self.train_model(
                cut_target_V,
                param_samples_for_next_stage[:, param_ind],
                lows=lows,
                highs=highs,
            )

            # Predict and take max across ci to prevent underestimating
            predictions = self.predict_with_model(cut_target_V, lows, highs)
            predictions = torch.max(predictions, dim=0).values

            # Update cell model
            self.cell.set_parameters(seg_params, predictions)

            # Update the pools
            prediction_pool.append(predictions)
            self.model_pool.append(self.model)

        prediction_pool = torch.cat(prediction_pool, dim=0)

        return prediction_pool

    def init_nn_model(self, in_channels: int, out_channels: int, num_summary_features: torch.Tensor = None) -> torch.nn.Sequential:
        # model = torch.nn.Sequential(
        #     torch.nn.Linear(in_channels, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, out_channels),
        #     torch.nn.Sigmoid(),
        # )
        model = EmbeddingNet(in_channels, out_channels, num_summary_features)
        return model

    def train_model(
        self,
        voltage_data: torch.Tensor,
        target_params: torch.Tensor,
        lows,
        highs,
    ) -> None:
        optim = torch.optim.SGD(self.model.parameters(), lr=1e-8)
        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        self.logger.info("Training a model with SGD optimizer and lr = 1e-8.")
        self.logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        self.model.train(True)
        for i in range(voltage_data.shape[0]):
            # Track loss to stop if it starts to go up
            loss0 = np.inf

            for ne in range(self.config["optimization_parameters"]["num_epochs"]):
                extr = self.extract_summary_features(voltage_data[i].reshape(1, -1))
                s_f = []
                for f in extr:
                    s_f.append(f.flatten())
                summary_features = torch.cat(s_f)

                pred = (
                    self.model(voltage_data[i], summary_features) * (sigmoid_maxs - sigmoid_mins)
                    + sigmoid_mins
                )
                loss = torch.nn.functional.l1_loss(pred, target_params[i])

                if ne % 100 == 0:
                    self.logger.epoch(ne, "l1_loss", float(loss.detach().numpy()))

                if loss.detach().numpy() >= loss0:
                    break
                loss0 = loss.detach().numpy()

                loss.backward()
                optim.step()

    def predict_with_model(self, target_V: torch.Tensor, lows, highs) -> torch.Tensor:
        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        self.model.eval()
        outs = []
        for i in range(target_V.shape[0]):
            extr = self.extract_summary_features(target_V[i].reshape(1, -1))
            s_f = []
            for f in extr:
                s_f.append(f.flatten())
            summary_features = torch.cat(s_f)
            out = self.model(target_V[i], summary_features) * (sigmoid_maxs - sigmoid_mins) + sigmoid_mins
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
        inds_of_ampl_next_stage = np.where(non_corresp == 0)[0].tolist()

        self.logger.info(
            f"Matched amplitudes: {np.array(self.config['optimization_parameters']['amps'])[inds_of_ampl_next_stage]}"
        )

        return simulated_V_for_next_stage, param_samples_for_next_stage

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
