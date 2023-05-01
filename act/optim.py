from neuron import h
from sbi import utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

import torch
import numpy as np
import json

from act.cell import Cell

class ACTOptimizer:
    '''
    Base class ACT-compatible optimizers should conform to.
    '''

    def __init__(self, config_file: str) -> None:
        '''
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        '''

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
        h.load_file('stdrun.hoc')

        # Parse config
        self.parse_config(config_file = config_file)

        # For convenience
        self.num_current_injections = len(self.current_injections)
        self.num_parameters = len(self.parameters)

        # Set current clamp session
        self.i_clamp = h.IClamp(self.cell.get_recording_section())

        # Observed (target) data
        self.observed_data = None

    def optimize(self, feature_model: torch.nn.Module, observed_data: torch.Tensor, num_epochs: int, num_simulations: int) -> np.array:
        '''
        Optimize current injections. This is the optimizer's main method which should
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

        observed_data: torch.Tensor(shape = (num_current_injections, 1024))
            Target values to optimize for.

        Returns:
        ----------
        estimates: np.array
            Parameter estimates made for observed_data.

        '''
        raise NotImplementedError
    
    def simulate(self, *args, **kwargs) -> torch.Tensor:
        '''
        Function to simulate data from a model for optimizer to use.

        Returns:
        ----------
        out: torch.Tensor, shape = (num_current_injections, 1024)
            Generated data.
        '''
        raise NotImplementedError

    def parse_config(self, config_file: str) -> None:
        '''
        Parse the configuration files and fill the attributes. 

        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        '''

        with open(config_file) as file:
            config_data = json.load(file)

        self.cell = Cell(hoc_file = config_data["general"]["hoc_file"], 
                         cell_name = config_data["general"]["cell_name"])
        
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
    '''
    Optimizer based on SBI's SNPE-C (https://www.mackelab.org/sbi/).
    '''

    def __init__(self, config_file: str) -> None:
        '''
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        '''

        super().__init__(config_file = config_file)
        
        # Set prior
        self.lows = torch.tensor(self.lows, dtype = float)
        self.highs = torch.tensor(self.highs, dtype = float)
        self.prior = utils.BoxUniform(low = self.lows, high = self.highs)

        # Set posterior
        self.posterior = []

    def optimize(self, feature_model: torch.nn.Module, observed_data: torch.Tensor, 
                 num_epochs: int = 1, num_simulations: int = 100,
                 num_samples: int = 10, workers: int = 1,
                 verbose = False) -> np.array:
        '''
        Optimize using SBI's SNPE-C method. Trains the feature model at each epoch.

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features. Accepts tensors of shape (num_current_injections, 1024).
            Returns tensors of shape (num_current_injections, num_summary_features).

        observed_data: torch.Tensor(shape = (num_current_injections, 1024))
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
            Show progress information.

        Returns:
        ----------
        estimates: np.array, shape = (num_current_injections, num_samples, number of parameters)
            Parameter estimates made for observed_data.
        '''

        self.observed_data = observed_data

        # Prepare the simulation function and prior for SBI (default method from SBI)
        # Simulation function must return tensors of shape (num_current_injections, 1024)
        self.simulator, self.prior = prepare_for_sbi(self.simulate, self.prior)
        
        # Set up posterior
        # hidden_features and num_transforms are sbi-specific (do not depend on any other params)
        neural_posterior = utils.posterior_nn(model = 'maf', embedding_net = feature_model, 
                                              hidden_features = 10, num_transforms = 2)
        
        # Set up an SBI's optimizer (default = SNPE_C)
        self.inference = SNPE(prior = self.prior, density_estimator = neural_posterior)

        proposal = self.prior
        for _ in range(num_epochs):
            # Train the feature model

            # theta = simulated parameters, shape = (num_simulations, number of optimization params)
            # x = samples obtained with these parameters, shape = (num_simulations, number of CI * 1024)
            # https://www.mackelab.org/sbi/reference/#sbi.inference.base.simulate_for_sbi
            theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations = num_simulations, 
                                        num_workers = workers, show_progress_bar = verbose)
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
                density_estimator.train(show_train_summary = verbose)

            # Build posterior
            proposal = self.inference.build_posterior()
            self.posterior.append(proposal)

            # Set proposal to be the most probable posterior sample
            # x should be of the same shape as current injections, so we again iterate over the batch
            for i in range(self.num_current_injections):
                samples = self.posterior[-1].sample((1000,), x = self.observed_data[i, :], show_progress_bars = verbose)
                log_prob = self.posterior[-1].log_prob(samples, x = self.observed_data[i, :], norm_posterior = False)
                proposal = samples[np.argmax(log_prob)]

        # Sample from the posterior
        out = []
        for i in range(self.num_current_injections):
            out.append(self.posterior[-1].sample((num_samples, ), x = self.observed_data[i, :], show_progress_bars = verbose).numpy())

        return np.array(out)

    def simulate(self, *args, **kwargs) -> torch.Tensor:
        '''
        Simulate data for SBI.

        Returns:
        ----------
        outs: torch.Tensor(shape = (num_current_injections, 1024))
            Simulated data.
        '''

        data = []

        for inj in self.current_injections:

            # Set simulation parameters
            steps_per_ms = 5000 / 600
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

            voltage, _ = self.cell.resample()
            data.append(voltage)

        out = torch.tensor(np.array(data)).float() # 5 x 1024

        return out
    
class NaiveLinearOptimizer(ACTOptimizer):
    '''
    At each epoch, naive linear optimizer generates random parameter samples from a uniform distribution 
    on given [lows, highs] bounds and runs NEURON on these samples for each value of current injection amplitude.
    NEURON's voltage output is then used as training data for the predictive model, which tries to find parameters 
    that could produce such output.
    
    The predictive model consists of two linear layers separated by a RELU activation. The output of the second linear layer 
    is passed through a sigmoid function and multiplied by (highs - lows) + lows to ensure predictions lie in the given bounds.
    The network is optimized with the ADAM optimizer to minimize MSE loss.
    
    The pipeline does not require any external data and uses observed_data only for predictions.

    The optimizer outputs parameter estimates for each current injection amplidtude. Ideally, constant parameter estimates should have
    the same value across all amplitudes, but if that is not the case, the estimates can be treated as different potential solutions.

    The optimizer is known to get stuck in local minima, so multiple re-runs are advised. 
    '''

    def __init__(self, config_file: str) -> None:
        '''
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        '''
        super().__init__(config_file)

    def optimize(self, observed_data: torch.Tensor, num_epochs: int = 100, lr: float = 1e-3, 
                 verbose: bool = False, return_loss_history: bool = False) -> np.array:
        '''
        Parameters:
        ----------
        observed_data: torch.Tensor(shape = (num_current_injections, 1024))
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
        '''
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.num_parameters),
            torch.nn.Sigmoid()
        )

        # Scalers to ensure parameter predictions are within range
        # Ref: https://stackoverflow.com/questions/73071399/how-to-bound-the-output-of-a-layer-in-pytorch
        sigmoid_mins = torch.tensor(self.lows)
        sigmoid_maxs = torch.tensor(self.highs)

        optim = torch.optim.Adam(self.model.parameters(), lr = lr)

        self.model.train(True)
        loss_history = []
        for ne in range(num_epochs):
            # Generate random parameter samples in the specified bounds
            param_samples = np.random.uniform(low = self.lows, high = self.highs)

            # Simulate a batch with generated parameter samples
            simulated_data = self.simulate(param_samples, self.parameters)

            # Do standard optimization
            losses = []
            for i in range(self.num_current_injections):
                optim.zero_grad()
                pred = self.model(simulated_data[i]) * (sigmoid_maxs - sigmoid_mins) + sigmoid_mins

                loss = torch.nn.functional.mse_loss(pred, torch.tensor(param_samples).float())
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
            out = self.model(observed_data[i]) * (sigmoid_maxs - sigmoid_mins) + sigmoid_mins
            outs.append(out.detach().numpy())

        if return_loss_history:
            return np.array(outs).reshape((self.num_current_injections, -1)), loss_history
        else:
            return np.array(outs).reshape((self.num_current_injections, -1))
        

    def simulate(self, *args, **kwargs) -> torch.Tensor:
        '''
        Simulate data for the optimizer.

        Returns:
        ----------
        outs: torch.Tensor(shape = (num_current_injections, 1024))
            Simulated data.
        '''

        data = []

        for inj in self.current_injections:

            # Set simulation parameters
            steps_per_ms = 5000 / 600
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

            voltage, _ = self.cell.resample()
            data.append(voltage)

        out = torch.tensor(np.array(data)).float()

        return out



class LinearOptimizer(ACTOptimizer):
    '''
    Optimizes by training one linear layer on top of the provided feature model. 
    Training is done by sampling random parameter estimates within specified bounds, simulating voltage 
    outputs with these estimates, passing the outputs through the model and minimizing MSELoss with the target.

    Predicts by running the model num_prediction_rounds times and selecting the prediction with the smallest loss.
    '''

    def __init__(self, config_file: str) -> None:
        '''
        Parameters:
        ----------
        config_file: str
            Path to the .json configuration file with simulation parameters' values.
        '''
        super().__init__(config_file)

    def optimize(self, feature_model: torch.nn.Module, observed_data: torch.Tensor, num_summary_features: int, 
                 num_epochs: int = 10, num_prediction_rounds: int = 10) -> np.array:
        '''
        Optimize using a linear model.

        Parameters:
        ----------
        feature_model: torch.nn.Module
            Model that generates summary features. Accepts tensors of shape (num_current_injections, 1024).
            Returns tensors of shape (num_current_injections, num_summary_features).

        observed_data: torch.Tensor(shape = (num_current_injections, 1024))
            Target values to optimize for.

        num_summary_features: int
            Number of summary features which a feature model outputs.

        num_epochs: int = 10
            Number of epochs (rounds) to run.

        num_prediction_rounds = 10
            Number of prediction trials.

        Returns:
        ----------
        estimates: np.array, shape = (number of parameters,)
            Parameter estimates made for observed_data.
        '''

        self.model = torch.nn.Sequential(
            feature_model,
            torch.nn.Linear(num_summary_features, 1024),
            torch.nn.ReLU()
        )

        optim = torch.optim.Adam(self.model.parameters(), lr = 1e-3)

        self.model.train(True)
        for _ in range(num_epochs):
            
            # Generate random parameter samples in the specified bounds
            param_samples = np.random.uniform(low = self.lows, high = self.highs)

            # Simulate a batch with generated parameter samples
            simulated_data = self.simulate(param_samples, self.parameters)

            # Do standard optimization
            for i in range(self.num_current_injections):
                optim.zero_grad()
                pred = self.model(simulated_data[i])
                loss = torch.nn.functional.mse_loss(pred, observed_data[i].reshape((1, -1)))
                loss.backward()
                optim.step()

        # Predict
        predicts = []
        losses = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_prediction_rounds):
                # Generate random parameter samples in the specified bounds
                param_samples = np.random.uniform(low = self.lows, high = self.highs)

                # Simulate a batch with generated parameter samples
                simulated_data = self.simulate(param_samples, self.parameters)

                candidate_loss = []
                for i in range(self.num_current_injections):
                    pred = self.model(simulated_data[i])
                    loss = torch.nn.functional.mse_loss(pred, observed_data[i].reshape((1, -1)))
                    candidate_loss.append(loss.detach().numpy())

                predicts.append(param_samples)
                losses.append(np.mean(candidate_loss))

        return predicts[np.argmin(losses)]
        

    def simulate(self, *args, **kwargs) -> torch.Tensor:
        '''
        Simulate data for the optimizer.

        Returns:
        ----------
        outs: torch.Tensor(shape = (num_current_injections, 1024))
            Simulated data.
        '''

        data = []

        for inj in self.current_injections:

            # Set simulation parameters
            steps_per_ms = 5000 / 600
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

            voltage, _ = self.cell.resample()
            data.append(voltage)

        out = torch.tensor(np.array(data)).float() # 5 x 1024

        return out

    
