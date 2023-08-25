# @DEPRECATED

# This file requires the following dependencies
# "NEURON>=8.2.0",
# "numpy==1.23.5",
# "sbi==0.17.2",
# "matplotlib",
# "torch",
# "scikit-learn"

import json

import numpy as np
import torch
from neuron import h
from sbi import utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from scipy.signal import resample
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from act.cell import Cell


class ACTOptimizer:
    """
    Base class ACT-compatible optimizers should conform to.
    """

    def __init__(self, config_file: str) -> None:
        """
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """

        # "general"
        self.cell = None

        # "run_options"
        self.v_init = None
        self.tstop = None
        self.i_clamp_delay = None
        self.i_clamp_dur = None

        # "optimization_parameters"
        self.current_injections = None
        self.parameters = None
        self.lows = None
        self.highs = None

        # Load the standard hoc file
        h.load_file("stdrun.hoc")

        # Parse config
        self.parse_config(config_file=config_file)

        # For convenience
        self.num_current_injections = len(self.current_injections)
        self.num_parameters = len(self.parameters)

        # Set current clamp session
        self.i_clamp = h.IClamp(self.cell.get_recording_section())

        # Observed (target) data
        self.observed_data = None

    def optimize(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        num_epochs: int,
        num_simulations: int,
    ) -> np.array:
        """
        Optimize parameters. This is the optimizer's main method which should
        (1) train the feature model if applicable,
        (2) perform the optimization,
        (3) return parameters's estimates.

        Additional arguments can be used in specific implementations.

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features.

        num_epochs: int
            Number of epochs (rounds) to run.

        num_simulations: int
            Number of simulations to draw each round.

        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        Returns:
        ----------
        estimates: np.array
            Parameter estimates made for observed_data.

        """
        raise NotImplementedError

    def optimize_with_segregation(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        parameter_inds: list,
        voltage_bounds: list,
        num_epochs: int,
        num_simulations: int,
    ) -> np.array:
        """
        Optimize using the segregation scheme from Alturki2016. Segregates and runs self.optimize(...).

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features.

        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        parameter_inds: list
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        num_epochs: int
            Number of epochs (rounds) to run.

        num_simulations: int
            Number of simulations to draw each round.

        Returns:
        ----------
        estimates: np.array
            Parameter estimates made for observed_data.

        """
        raise NotImplementedError

    def prepare_for_segregation(
        self, observed_data: torch.Tensor, parameter_inds: list, voltage_bounds: list
    ):
        """
        Cuts a specified region from the voltage trace and resamples the cut portions to match the original data's length
        and sets parameter lists to optimize for user-defined parameters.

        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        parameter_inds: list[int]
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        Returns:
        ----------
        cut_observed_data: torch.Tensor
            Data to estimate on.

        original_param_set: list[list]
            List of original values for self.parameters, self.lows, self.highs, self.num_parameters to use for a reset later.
        """

        # Cut observed data and temporarily drop parameters which are not in the region of interest
        # Clunky, but it is a bad side-effect of using config files.
        cut_observed_data = torch.zeros_like(observed_data)
        for i in range(self.num_current_injections):
            trace = observed_data[i]
            trace = trace[(trace >= voltage_bounds[0]) & (trace < voltage_bounds[1])]
            cut_observed_data[i] = torch.tensor(
                resample(x=trace, num=observed_data.shape[1])
            ).float()

        original_param_set = [
            self.parameters.copy(),
            self.lows.copy(),
            self.highs.copy(),
            self.num_parameters,
        ]
        self.parameters = [self.parameters[i] for i in parameter_inds]
        self.lows = [self.lows[i] for i in parameter_inds]
        self.highs = [self.highs[i] for i in parameter_inds]
        self.num_parameters = len(self.parameters)

        return cut_observed_data, original_param_set

    def simulate(self, parameters: np.ndarray) -> torch.Tensor:
        """
        Function to simulate data from a model for optimizer to use.

        Parameters:
        ----------
        parameters: ndarray(shape = num_parameters)
            Parameters optimized for.

        Returns:
        ----------
        out: ndarray(shape = (num_current_injections, tstop * 10 + 1))
            Generated data.
        """
        simulated_data = []

        for inj in self.current_injections:
            # Set simulation parameters
            steps_per_ms = 10
            h.steps_per_ms = steps_per_ms
            h.dt = 1 / steps_per_ms
            h.tstop = self.tstop
            h.v_init = self.v_init

            self.i_clamp.dur = self.i_clamp_dur
            self.i_clamp.amp = inj
            self.i_clamp.delay = self.i_clamp_delay

            self.cell.set_parameters(self.parameters, parameters)

            # Run the simulation with the given parameters
            h.run()

            voltage = self.cell.mem_potential.as_numpy()
            simulated_data.append(voltage)

        out = np.array(simulated_data)

        return out

    def parse_config(self, config_file: str) -> None:
        """
        Parse the configuration files and fill the attributes.

        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """

        with open(config_file) as file:
            config_data = json.load(file)

        self.cell = Cell(
            hoc_file=config_data["general"]["hoc_file"],
            cell_name=config_data["general"]["cell_name"],
        )

        run_options = config_data["run_options"]
        self.v_init = run_options["v_init"]
        self.tstop = run_options["tstop"]
        self.i_clamp_delay = run_options["i_clamp_delay"]
        self.i_clamp_dur = run_options["i_clamp_dur"]

        optimization_parameters = config_data["optimization_parameters"]
        self.current_injections = optimization_parameters["current_injections"]
        self.parameters = optimization_parameters["parameters"]
        self.lows = optimization_parameters["lows"]
        self.highs = optimization_parameters["highs"]


class SBIOptimizer(ACTOptimizer):
    """
    Optimizer based on SBI's SNPE-C (https://www.mackelab.org/sbi/). Trains the feature model at each epoch.
    """

    def __init__(self, config_file: str) -> None:
        """
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """

        super().__init__(config_file=config_file)

        # Set prior
        self.lows = torch.tensor(self.lows, dtype=float)
        self.highs = torch.tensor(self.highs, dtype=float)
        self.prior = utils.BoxUniform(low=self.lows, high=self.highs)

        # Set posterior
        self.posterior = []

    def optimize(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        num_epochs: int = 1,
        num_simulations: int = 100,
        num_samples: int = 10,
        workers: int = 1,
        verbose=False,
    ) -> np.array:
        """
        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features. Accepts tensors of shape (num_current_injections, len_voltage_trace).
            Returns tensors of shape (num_current_injections, num_summary_features).

        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        num_epochs: int = 1
            Number of epochs (rounds) to run.

        num_simulations: int = 100
            Number of simulations to draw each round.

        num_samples: int = 10
            Number of parameter estimates' samples to return.

        workers: int = 1
            Number of threads for SBI to use.

        verbose: bool = False
            Show progress information reported by SBI.

        Returns:
        ----------
        estimates: np.array, shape = (num_current_injections, num_samples, number of parameters)
            Parameter estimates made for observed_data.
        """

        self.observed_data = observed_data

        # Prepare the simulation function and prior for SBI (default method from SBI)
        # Simulation function must return tensors of shape (num_current_injections, voltage_trace_length)
        self.simulator, self.prior = prepare_for_sbi(self.simulate, self.prior)

        # Set up posterior
        # hidden_features and num_transforms are sbi-specific (do not depend on any other params)
        neural_posterior = utils.posterior_nn(
            model="maf",
            embedding_net=feature_model,
            hidden_features=10,
            num_transforms=2,
        )

        # Set up an SBI's optimizer (default = SNPE_C)
        self.inference = SNPE(prior=self.prior, density_estimator=neural_posterior)

        proposal = self.prior
        for _ in range(num_epochs):
            # Train the feature model

            # theta = simulated parameters, shape = (num_simulations, number of optimization params)
            # x = samples obtained with these parameters, shape = (num_simulations, number of CI * 1024)
            # https://www.mackelab.org/sbi/reference/#sbi.inference.base.simulate_for_sbi
            theta, x = simulate_for_sbi(
                self.simulator,
                self.prior,
                num_simulations=num_simulations,
                num_workers=workers,
                show_progress_bar=verbose,
            )
            # Reshape x for convenience
            x = x.reshape((-1, self.num_current_injections, 1024))

            # Train the model for each value of current injections (sbi only works with shapes (1, ...))
            # The model is saved after each current injections, so we basically iterate over a batch
            for i in range(self.num_current_injections):
                density_estimator = self.inference.append_simulations(theta, x[:, i, :])
                # Note: sbi' API has resume_training parameter in .train(...) which defaults to False,
                # and the method breaks if it's set to True. However, we checked the model's weights
                # are saved after each iteration even with resume_training = False, so the code
                # works as intended
                density_estimator.train(show_train_summary=verbose)

            # Build posterior
            proposal = self.inference.build_posterior()
            self.posterior.append(proposal)

            # Set proposal to be the most probable posterior sample
            # x should be of the same shape as current injections, so we again iterate over the batch
            for i in range(self.num_current_injections):
                samples = self.posterior[-1].sample(
                    (1000,), x=self.observed_data[i, :], show_progress_bars=verbose
                )
                log_prob = self.posterior[-1].log_prob(
                    samples, x=self.observed_data[i, :], norm_posterior=False
                )
                proposal = samples[np.argmax(log_prob)]

        # Sample from the posterior
        out = []
        for i in range(self.num_current_injections):
            out.append(
                self.posterior[-1]
                .sample(
                    (num_samples,),
                    x=self.observed_data[i, :],
                    show_progress_bars=verbose,
                )
                .numpy()
            )

        return np.array(out)

    def simulate(self, *args, **kwargs) -> torch.Tensor:
        """
        Simulate data for SBI.

        Returns:
        ----------
        outs: torch.Tensor(shape = (num_current_injections, 1024))
            Simulated data.
        """

        simulated_data = []

        for inj in self.current_injections:
            # Set simulation parameters
            steps_per_ms = 10
            h.steps_per_ms = steps_per_ms
            h.dt = 1 / steps_per_ms
            h.tstop = self.tstop
            h.v_init = self.v_init

            self.i_clamp.dur = self.i_clamp_dur
            self.i_clamp.amp = inj
            self.i_clamp.delay = self.i_clamp_delay

            self.cell.set_parameters(self.parameters, args[0])
            self.cell.set_parameters(list(kwargs.keys()), list(kwargs.values()))

            # Run the simulation with the given parameters
            h.run()

            voltage = self.cell.mem_potential.as_numpy()
            voltage = resample(voltage, self.observed_data.shape[1])
            simulated_data.append(voltage)

        out = torch.tensor(np.array(simulated_data)).float()

        return out

    def optimize_with_segregation(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        parameter_inds: list,
        voltage_bounds: list,
        num_epochs: int = 1,
        num_simulations: int = 100,
        num_samples: int = 10,
        workers: int = 1,
        verbose=False,
    ) -> np.array:
        """
        Optimize using the segregation scheme from Alturki2016.

        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        parameter_inds: list
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        num_groves: int = 100
            Number of groves to average across. Equivalent to the number of epochs.

        num_prediction_rounds: int = 10
            Number of prediction trials.

        tree_max_depth: int = 5
            Max depth of each tree.

        Returns:
        ----------
        estimates: ndarray, shape = (number of parameters, )
            Parameter estimates.
        """
        # Get data to estimate on and set parameters to the user-defined values;
        # Save original parameters for a future reset
        cut_observed_data, original_param_set = self.prepare_for_segregation(
            observed_data, parameter_inds, voltage_bounds
        )

        # Optimize
        out = self.optimize(
            feature_model=feature_model,
            observed_data=cut_observed_data,
            num_epochs=num_epochs,
            num_simulations=num_simulations,
            num_samples=num_samples,
            workers=workers,
            verbose=verbose,
        )

        # Reset parameters back to the full sets
        self.parameters, self.lows, self.highs, self.num_parameters = original_param_set

        return out


class NaiveLinearOptimizer(ACTOptimizer):
    """
    At each epoch, Naive Linear Optimizer generates random parameter samples from the uniform distribution
    on given [lows, highs] bounds and runs NEURON on these samples for each value of current injection amplitude.
    NEURON's voltage output is then used as training data for the predictive model, which tries to find parameters
    that could produce such output.

    The predictive model consists of two linear layers separated by a ReLU activation. The output of the second linear layer
    is passed through a sigmoid function and multiplied by (highs - lows) + lows to ensure predictions lie in the given bounds.
    The network is optimized with the ADAM optimizer to minimize MSE loss.

    The pipeline does not require any external data and uses observed_data only for predictions. It also does not support feature models.

    The optimizer outputs parameter estimates for each current injection amplidtude. Ideally, constant parameter estimates should have
    the same value across all amplitudes, but if that is not the case, the estimates can be treated as different potential solutions.

    The optimizer is known to get stuck in local minima, so multiple re-runs are advised.
    """

    def __init__(self, config_file: str) -> None:
        """
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """
        super().__init__(config_file)

    def optimize(
        self,
        observed_data: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
        return_loss_history: bool = False,
    ) -> np.array:
        """
        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for. They will only be used for predictions.

        num_epochs: int = 100
            Number of epochs (rounds) to run.

        lr: float = 1e-3
            Learning rate for the ADAM optimizer.

        verbose: bool = False
            Whether to print loss at each epoch. The loss is averaged across all current injections.

        return_loss_history: bool = False
            Whether to return loss history together with parameter esitmates.

        Returns:
        ----------
        estimates: ndarray, shape = (num_current_injections, number of parameters)
            Parameter estimates.

        loss_history: list
            If return_loss_history = True, list of loss values at each epoch.
        """

        self.model = torch.nn.Sequential(
            torch.nn.Linear(observed_data.shape[1], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.num_parameters),
            torch.nn.Sigmoid(),
        )

        # Scalers to ensure parameter predictions are within range
        # Ref: https://stackoverflow.com/questions/73071399/how-to-bound-the-output-of-a-layer-in-pytorch
        sigmoid_mins = torch.tensor(self.lows)
        sigmoid_maxs = torch.tensor(self.highs)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train(True)
        loss_history = []
        for ne in range(num_epochs):
            # Generate random parameter samples in the specified bounds
            param_samples = np.random.uniform(low=self.lows, high=self.highs)

            # Simulate a batch with generated parameter samples
            simulated_data = self.simulate(param_samples)

            # Resample to match the length of target data
            resampled_data = []
            for i in range(self.num_current_injections):
                resampled_data.append(
                    resample(x=simulated_data[i], num=observed_data.shape[1])
                )

            resampled_data = torch.tensor(np.array(resampled_data)).float()

            # Do standard optimization
            losses = []
            for i in range(self.num_current_injections):
                optim.zero_grad()
                pred = (
                    self.model(resampled_data[i]) * (sigmoid_maxs - sigmoid_mins)
                    + sigmoid_mins
                )

                loss = torch.nn.functional.mse_loss(
                    pred, torch.tensor(param_samples).float()
                )
                losses.append(loss.detach().numpy())

                loss.backward()
                optim.step()

            if verbose:
                print(f"epoch {ne}: mean_train_loss = {np.mean(losses)}")

            loss_history.append(np.mean(losses))

        # Get predictions
        self.model.eval()
        outs = []
        for i in range(self.num_current_injections):
            out = (
                self.model(observed_data[i]) * (sigmoid_maxs - sigmoid_mins)
                + sigmoid_mins
            )
            outs.append(out.detach().numpy())

        if return_loss_history:
            return (
                np.array(outs).reshape((self.num_current_injections, -1)),
                loss_history,
            )
        else:
            return np.array(outs).reshape((self.num_current_injections, -1))

    def optimize_with_segregation(
        self,
        observed_data: torch.Tensor,
        parameter_inds: list,
        voltage_bounds: list,
        num_epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
        return_loss_history: bool = False,
    ) -> np.array:
        """
        Optimize using the segregation scheme from Alturki2016.

        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for. They will only be used for predictions.

        parameter_inds: list
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        num_epochs: int = 100
            Number of epochs (rounds) to run.

        lr: float = 1e-3
            Learning rate for the ADAM optimizer.

        verbose: bool = False
            Whether to print loss at each epoch. The loss is averaged across all current injections.

        return_loss_history: bool = False
            Whether to return loss history together with parameter esitmates.

        Returns:
        ----------
        estimates: ndarray, shape = (num_current_injections, number of parameters)
            Parameter estimates.

        loss_history: list
            If return_loss_history = True, list of loss values at each epoch.
        """
        # Get data to estimate on and set parameters to the user-defined values;
        # Save original parameters for a future reset
        cut_observed_data, original_param_set = self.prepare_for_segregation(
            observed_data, parameter_inds, voltage_bounds
        )

        # Optimize
        out = self.optimize(
            cut_observed_data, num_epochs, lr, verbose, return_loss_history
        )

        # Reset parameters back to the full sets
        self.parameters, self.lows, self.highs, self.num_parameters = original_param_set

        return out


class RandomSearchLinearOptimizer(ACTOptimizer):
    """
    At each epoch, Random Search Linear Optimizer generates a random parameter sample from the uniform on [lows, highs] distribution
    and runs NEURON with the generated sample. NEURON's voltage output is passed through a feature model and a linear layer with
    ReLU activation to obtain predictions which are then optimized to match the observed data. Both the feature and linear models
    are trained simultaneoulsy with the SGD optimizer and MSE loss.

    Predictions for the observed data are made with random search. At each round, a random parameter sample is generated and passed
    through NEURON and the trained model, and the loss value is saved to the list. The algorithm outputs the parameter sample with
    the lowest loss value across prediction rounds.

    Due to the nature of the algorithm, training loss values on general data will probably be large and unstable.
    """

    def __init__(self, config_file: str) -> None:
        """
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """
        super().__init__(config_file)

    def optimize(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        num_summary_features: int,
        num_epochs: int = 100,
        num_prediction_rounds: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
        return_loss_history: bool = False,
    ) -> np.array:
        """
        Optimize using a linear model.

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features. Accepts tensors of shape (num_current_injections, len_voltage_trace).
            Returns tensors of shape (num_current_injections, num_summary_features).

        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        num_summary_features: int
            Number of summary features which a feature model outputs.

        num_epochs: int = 10
            Number of training epochs to run.

        num_prediction_rounds: int = 10
            Number of prediction trials.

        verbose: bool = False
            Whether to print training loss at each epoch. The loss is averaged across current injections.

        return_loss_history: bool = False
            Whether to return loss history together with parameter esitmates.

        Returns:
        ----------
        estimates: ndarray, shape = (number of parameters, )
            Parameter estimates.

        loss_history: list
            If return_loss_history = True, list of loss values at each epoch.
        """

        self.model = torch.nn.Sequential(
            feature_model,
            torch.nn.Linear(num_summary_features, observed_data.shape[1]),
            torch.nn.ReLU(),
        )

        optim = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.model.train(True)
        loss_history = []
        for ne in range(num_epochs):
            # Generate a random parameter sample in the specified bounds
            param_samples = np.random.uniform(low=self.lows, high=self.highs)

            # Simulate a batch with the generated parameter sample
            simulated_data = self.simulate(param_samples)

            # Resample to match the length of target data
            resampled_data = []
            for i in range(self.num_current_injections):
                resampled_data.append(
                    resample(x=simulated_data[i], num=observed_data.shape[1])
                )

            resampled_data = torch.tensor(np.array(resampled_data)).float()

            # Do standard optimization
            losses = []
            for i in range(self.num_current_injections):
                optim.zero_grad()
                pred = self.model(resampled_data[i])

                loss = torch.nn.functional.mse_loss(
                    pred, observed_data[i].reshape((1, -1))
                )
                losses.append(loss.detach().numpy())

                loss.backward()
                optim.step()

            if verbose:
                print(f"epoch {ne}: mean_train_loss = {np.mean(losses)}")
            if return_loss_history:
                loss_history.append(np.mean(losses))

        # Predict with random search
        predicts = []
        losses = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_prediction_rounds):
                # Generate random parameter samples in the specified bounds
                param_samples = np.random.uniform(low=self.lows, high=self.highs)

                # Simulate a batch with generated parameter samples
                simulated_data = self.simulate(param_samples)

                # Resample to match the length of target data
                resampled_data = []
                for i in range(self.num_current_injections):
                    resampled_data.append(
                        resample(x=simulated_data[i], num=observed_data.shape[1])
                    )

                resampled_data = torch.tensor(np.array(resampled_data)).float()

                candidate_loss = []
                for i in range(self.num_current_injections):
                    pred = self.model(resampled_data[i])
                    loss = torch.nn.functional.mse_loss(
                        pred, observed_data[i].reshape((1, -1))
                    )
                    candidate_loss.append(loss.detach().numpy())

                predicts.append(param_samples)
                losses.append(np.mean(candidate_loss))

        final_prediction = np.array(predicts[np.argmin(losses)]).flatten()

        if return_loss_history:
            return final_prediction, loss_history
        else:
            return final_prediction

    def optimize_with_segregation(
        self,
        feature_model: torch.nn.Module,
        observed_data: torch.Tensor,
        num_summary_features: int,
        parameter_inds: list,
        voltage_bounds: list,
        num_epochs: int = 100,
        num_prediction_rounds: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
        return_loss_history: bool = False,
    ) -> np.array:
        """
        Optimize using the segregation scheme from Alturki2016.

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features. Accepts tensors of shape (num_current_injections, len_voltage_trace).
            Returns tensors of shape (num_current_injections, num_summary_features).

        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        num_summary_features: int
            Number of summary features which a feature model outputs.

        parameter_inds: list
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        num_epochs: int = 10
            Number of training epochs to run.

        num_prediction_rounds: int = 10
            Number of prediction trials.

        verbose: bool = False
            Whether to print training loss at each epoch. The loss is averaged across current injections.

        return_loss_history: bool = False
            Whether to return loss history together with parameter esitmates.

        Returns:
        ----------
        estimates: ndarray, shape = (number of parameters, )
            Parameter estimates.

        loss_history: list
            If return_loss_history = True, list of loss values at each epoch.
        """
        # Get data to estimate on and set parameters to the user-defined values;
        # Save original parameters for a future reset
        cut_observed_data, original_param_set = self.prepare_for_segregation(
            observed_data, parameter_inds, voltage_bounds
        )

        # Optimize
        out = self.optimize(
            feature_model=feature_model,
            observed_data=cut_observed_data,
            num_summary_features=num_summary_features,
            num_epochs=num_epochs,
            num_prediction_rounds=num_prediction_rounds,
            lr=lr,
            verbose=verbose,
            return_loss_history=return_loss_history,
        )

        # Reset parameters back to the full sets
        self.parameters, self.lows, self.highs, self.num_parameters = original_param_set

        return out


class RandomSearchTreeOptimizer(ACTOptimizer):
    """
    At each epoch, Random Search Tree Optimizer generates a random parameter sample from the uniform on [lows, highs] distribution
    and runs NEURON with the generated sample. NEURON's voltage output for each current injection value is then used to train an
    independent decision tree to predict the voltage trace for the same current injection value from the observed data. Trees for
    a single parameter sample are united into groves (lists of fitted decision trees). The final model is a list of groves.

    Predictions for the observed data are made with random search. At each round, a random parameter sample is generated and passed
    through NEURON and then averaged across all groves for each value of current injection. The algorithm outputs the parameter sample with
    the lowest loss value across prediction rounds.

    The tree model is provided by scikit-learn, so all methods and attributes from
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html can be used.
    """

    def __init__(self, config_file: str) -> None:
        """
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        """
        super().__init__(config_file)

    def optimize(
        self,
        observed_data: torch.Tensor,
        num_groves: int = 100,
        num_prediction_rounds: int = 10,
        tree_max_depth: int = 5,
    ) -> np.array:
        """
        Optimize using a decision tree.

        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        num_groves: int = 100
            Number of groves to average across. Equivalent to the number of epochs.

        num_prediction_rounds: int = 10
            Number of prediction trials.

        tree_max_depth: int = 5
            Max depth of each tree.

        Returns:
        ----------
        estimates: ndarray, shape = (number of parameters, )
            Parameter estimates.
        """

        self.model = []

        for _ in range(num_groves):
            # Generate a random parameter sample in the specified bounds
            param_samples = np.random.uniform(low=self.lows, high=self.highs)

            # Simulate a batch with the generated parameter sample
            simulated_data = self.simulate(param_samples)

            # Resample to match the length of target data
            resampled_data = []
            for i in range(self.num_current_injections):
                resampled_data.append(
                    resample(x=simulated_data[i], num=observed_data.shape[1])
                )

            resampled_data = np.array(resampled_data)

            # Train a tree for each value of current injection
            grove = []
            for i in range(self.num_current_injections):
                tree = DecisionTreeRegressor(max_depth=tree_max_depth)
                tree.fit(resampled_data[i].reshape((-1, 1)), observed_data[i])
                grove.append(tree)

            self.model.append(grove)

        # Predict with random search
        predicts = []
        losses = []

        for _ in range(num_prediction_rounds):
            # Generate random parameter samples in the specified bounds
            param_samples = np.random.uniform(low=self.lows, high=self.highs)

            # Simulate a batch with generated parameter samples
            simulated_data = self.simulate(param_samples)

            # Resample to match the length of target data
            resampled_data = []
            for i in range(self.num_current_injections):
                resampled_data.append(
                    resample(x=simulated_data[i], num=observed_data.shape[1])
                )

            resampled_data = np.array(resampled_data)

            # Final prediction is the average prediction across all trees for each value of current injection
            candidate_loss = []
            for i in range(self.num_current_injections):
                pred_for_ci = 0
                for j in range(num_groves):
                    pred_for_ci += self.model[j][i].predict(
                        resampled_data[i].reshape((-1, 1))
                    )
                pred_for_ci = pred_for_ci / num_groves

                loss = mean_squared_error(
                    observed_data[i].detach().numpy(), pred_for_ci
                )
                candidate_loss.append(loss)

            predicts.append(param_samples)
            losses.append(np.mean(candidate_loss))

        final_prediction = np.array(predicts[np.argmin(losses)]).flatten()

        return final_prediction

    def optimize_with_segregation(
        self,
        observed_data: torch.Tensor,
        parameter_inds: list,
        voltage_bounds: list,
        num_groves: int = 100,
        num_prediction_rounds: int = 10,
        tree_max_depth: int = 5,
    ) -> np.array:
        """
        Optimize using the segregation scheme from Alturki2016.

        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, len_voltage_trace))
            Target values to optimize for.

        parameter_inds: list
            Indexes of parameters to be optimized for.

        voltage_bounds: list[int]
            Lower and upper bound of the region which will be cut from the voltage trace.

        num_groves: int = 100
            Number of groves to average across. Equivalent to the number of epochs.

        num_prediction_rounds: int = 10
            Number of prediction trials.

        tree_max_depth: int = 5
            Max depth of each tree.

        Returns:
        ----------
        estimates: ndarray, shape = (number of parameters, )
            Parameter estimates.
        """
        # Get data to estimate on and set parameters to the user-defined values;
        # Save original parameters for a future reset
        cut_observed_data, original_param_set = self.prepare_for_segregation(
            observed_data, parameter_inds, voltage_bounds
        )

        # Optimize
        out = self.optimize(
            observed_data=cut_observed_data,
            num_groves=num_groves,
            num_prediction_rounds=num_prediction_rounds,
            tree_max_depth=tree_max_depth,
        )

        # Reset parameters back to the full sets
        self.parameters, self.lows, self.highs, self.num_parameters = original_param_set

        return out
