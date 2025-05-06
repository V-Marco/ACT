import os
import time
from datetime import timedelta
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from itertools import product

from act.types import SimulationParameters, OptimizationParameters, ConductanceOptions, ConstantCurrentInjection, RampCurrentInjection, GaussianCurrentInjection
from act.cell_model import ACTCellModel
from act.simulator import ACTSimulator
from act.data_processing import combine_data, remove_saturated_traces, get_traces_with_spikes, get_summary_features, select_features, clean_g_bars
from act.metrics import summary_features_error

#TODO: maybe put target data reading + sum.f. extraction + feature filtering into a _function, since we call it so much

class ACTModule:

    def __init__(
            self, 
            name: str,
            cell: ACTCellModel, 
            simulation_parameters: SimulationParameters,
            optimization_parameters: OptimizationParameters,
            target_file: str):

        # Module name
        self.name = name
        self.output_folder_path = os.path.join(os.getcwd(), f"module_{self.name}")

        self.cell = cell
        self.simulation_parameters = simulation_parameters
        self.optimization_parameters = optimization_parameters
        self.target_file = target_file

        # Model (assigned after fitting)
        self.model = None
        
        #TODO: check if needed
        self.blocked_channels = []

    def run(self) -> None:
        '''
        The Main method to run the automatic cell tuning process given user settings
        Parameters:
        -----------
        self
        
        Returns:
        -----------
        predicted_g_data_file: str
            Filepath to predicted conductances
        '''
        start_time = time.time()
        print(f"Running Module {self.name}...")
        print("----------")

        print("Simulating train traces...")
        self.simulate_cells(self.cell)

        if self.optimization_parameters.filter_parameters is not None:
            if self.optimization_parameters.filter_parameters.filtered_out_features is not None:
                print("Filtering...")
                self.filter_data(os.path.join(self.output_folder_name, "train", "combined_out.npy"))
        else:
            print("Filtering skipped.")

        print("Training RandomForest...")
        self.train_random_forest()

        # Make and evaluate predictions
        print("Predicting on target data...")
        dataset_target = np.load(self.target_file)
        V_target = dataset_target[:, :, 0]
        I_target = dataset_target[:, :, 1]

        # Compute target SF
        target_df = get_summary_features(
            V = V_target, 
            I = I_target,
            lto_hto = 0, #TODO: fix
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )

        # If train_features is None, use all features
        if self.optimization_parameters.train_features is not None:
            target_df = select_features(target_df, self.optimization_parameters.train_features)

        prediction = self.model.predict(target_df)
        self.simulate_cells(self.cell, prediction)

        print("Evaluating predictions...")
        sf_error, g_pred = self.evaluate_predictions()

        conductance_option_names_list = [conductance_option.variable_name for conductance_option in self.optimization_parameters.conductance_options]
        final_prediction = dict(zip(conductance_option_names_list, g_pred[np.argmin(sf_error)]))
        self.cell.prediction = final_prediction #TODO: do we need this?
        
        #TODO: handle post-processing
        print(self.cell.prediction)
            
        end_time = time.time()
        run_time = end_time - start_time
        runtime_timedelta = timedelta(seconds = run_time)
        formatted_runtime = str(runtime_timedelta)
        print(f"Done. Finished in {formatted_runtime} sec.\n")
    
    def evaluate_predictions(self):
        # Load target data
        dataset_target = np.load(self.target_file)
        V_target = dataset_target[:, :, 0]
        I_target = dataset_target[:, :, 1]
        lto_hto = dataset_target[:, 1, 2]

        # Compute target SF
        target_df = get_summary_features(
            V = V_target, 
            I = I_target,
            lto_hto = 0, #TODO: fix
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )

        # If train_features is None, use all features
        if self.optimization_parameters.train_features is not None:
            target_df = select_features(target_df, self.optimization_parameters.train_features)

        # Load predicted data
        dataset_pred = np.load(os.path.join(self.output_folder_path, "eval", "combined_out.npy"))

        # Construct the dataset
        g_pred = clean_g_bars(dataset_pred)
        V_pred = dataset_pred[:, :, 0]
        I_pred = dataset_pred[:, :, 1]
        lto_hto = dataset_pred[:, 1, 3]
        
        pred_df = get_summary_features(
            V = V_pred, 
            I = I_pred,
            lto_hto = lto_hto, #TODO: check
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )
        
        if self.optimization_parameters.train_features is not None:
            pred_df = select_features(pred_df, self.optim_params.train_features)
        
        sf_error = summary_features_error(target_df.to_numpy(), pred_df.to_numpy())
        
        return sf_error, g_pred

    def generate_g_combinations(self) -> None:

        g = []
        n_slices = []
        for conductance_option in self.optimization_parameters.conductance_options:
            # If blocked, set the range to (0, 0)
            if conductance_option.blocked:
                g.append((0, 0)) # (low, high)
                n_slices.append(1)

            # Or, if was optimized before, set bounds variation
            elif (conductance_option.prediction != None) and (conductance_option.bounds_variation != None):
                g.append((
                    conductance_option.prediction - conductance_option.prediction * conductance_option.bounds_variation,
                    conductance_option.prediction + conductance_option.prediction * conductance_option.bounds_variation
                    ))
                n_slices.append(conductance_option.n_slices)

            # Or, set the range for optimization
            elif (conductance_option.low != None) and (conductance_option.high != None):
                # If only one slice, choose the middle of the range
                if conductance_option.n_slices == 1:
                    g.append((
                    conductance_option.low + (conductance_option.high - conductance_option.low) * 0.5,
                    conductance_option.low + (conductance_option.high - conductance_option.low) * 0.5,
                    ))
                    n_slices.append(1)

                # If multiple slices, just copy the conductance option
                else:
                    g.append((
                    conductance_option.low,
                    conductance_option.high,
                    ))
                    n_slices.append(conductance_option.n_slices)

            else: 
                raise RuntimeError("OptimizationParameters not defined fully. Need either (low & high) or (prediction & variation).")
        
        g_combinations = list(product(*[np.linspace(low, high, num = slices) for (low, high), slices in zip(g, n_slices)]))
        return g_combinations

    def simulate_cells(self, cell: ACTCellModel, g_comb: list = None) -> None:

        if g_comb is None:
            mode = "train"
        else:
            mode = "eval"

        # Set the simulator
        simulator = ACTSimulator(self.output_folder_path)

        # Set self.conductance_combos and self.current_inj_combos
        if mode == "train":
            g_comb = self.generate_g_combinations()
        # else, g_comb is given as RF predictions

        for group_id in range(len(g_comb)):
            
            # Init a cell for every combo
            specific_cell = ACTCellModel(
                cell_name = cell.cell_name, 
                path_to_hoc_file = cell.path_to_hoc_file,
                path_to_mod_files = cell.path_to_mod_files,
                passive = cell.passive,
                active_channels = cell.active_channels,
                prediction = cell.prediction)
            
            # Set conductances
            specific_cell.set_g_bar(specific_cell.active_channels, list(g_comb[group_id]))

            # Submit the job
            for curr_inj in self.optimization_parameters.CI_options:
                simulator.submit_job(
                    specific_cell, 
                    SimulationParameters(
                        sim_name = mode,
                        sim_idx = group_id,
                        h_v_init = self.simulation_parameters.h_v_init,    # (mV)
                        h_tstop = self.simulation_parameters.h_tstop,      # (ms)
                        h_dt = self.simulation_parameters.h_dt,            # (ms)
                        h_celsius = self.simulation_parameters.h_celsius,  # (deg C)
                        CI = [curr_inj]
                    )
                )

        simulator.run_jobs()
        combine_data(os.path.join(self.output_folder_path, mode))
        
    def filter_data(self, path) -> None:

        filtered_out_features = self.optim_params.filter_parameters.filtered_out_features
        data = np.load(path)
                           
        if "saturated" in filtered_out_features:
            window_of_inspection = self.optim_params.filter_parameters.window_of_inspection
            if window_of_inspection is None:
                inspection_start = self.sim_params.CI[0].delay + int(self.sim_params.CI[0].dur * 0.7)
                inspection_end = self.sim_params.CI[0].delay + self.sim_params.CI[0].dur
                window_of_inspection = (inspection_start, inspection_end)
            saturation_threshold = self.optim_params.filter_parameters.saturation_threshold
            data = remove_saturated_traces(data, window_of_inspection, threshold = saturation_threshold)
        
        if "no_spikes" in filtered_out_features:
            spike_threshold = self.optim_params.spike_threshold
            data = get_traces_with_spikes(data, spike_threshold = spike_threshold)
         
        print(f"Dataset size after filtering: {len(data)}. Saving to {os.path.join(path, 'filtered_out.npy')}")
        np.save(os.path.join(path, f"filtered_out.npy"), data)
    

    def train_random_forest(self):
        '''
        Trains a Random Forest Regressor model on the features of the generated simulation data.
        Then gets a prediction for conductance sets that yeild features found in the target data.
        Then calculates an evaluation of the RF model and saves the metric.
        Parameters:
        -----------
        self
        
        Returns:
        -----------
        predictions: np.ndarray
            Conductance Predictions
        '''
        # Load target data
        dataset_target = np.load(self.target_file)
        V_target = dataset_target[:, :, 0]
        I_target = dataset_target[:, :, 1]
        lto_hto = dataset_target[:, 1, 2]

        # Compute target SF
        target_df = get_summary_features(
            V = V_target, 
            I = I_target,
            lto_hto = 0, #TODO: fix
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )

        # If train_features is None, use all features
        if self.optimization_parameters.train_features is not None:
            target_df = select_features(target_df, self.optimization_parameters.train_features)
        
        # Load train data
        file_path = os.path.join(self.output_folder_path, "train", "filtered_out.npy")
        if os.path.exists(file_path):
            dataset_train = np.load(file_path)
        else:
            dataset_train = np.load(os.path.join(self.output_folder_path, "train", "combined_out.npy"))

        # Construct a train dataset
        g_train = clean_g_bars(dataset_train)
        V_train = dataset_train[:, :, 0]
        I_train = dataset_train[:, :, 1]
        lto_hto = dataset_train[:, 1, 3]
        
        train_df = get_summary_features(
            V = V_train, 
            I = I_train,
            lto_hto = lto_hto, #TODO: check
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )
        
        if self.optimization_parameters.train_features is not None:
            train_df = select_features(train_df, self.optim_params.train_features)

        X_train = train_df.to_numpy()
        y_train = g_train

        # Fit RF
        rf = RandomForestRegressor(
            n_estimators = self.optimization_parameters.n_estimators,
            random_state = self.optimization_parameters.random_state,
            max_depth = self.optimization_parameters.max_depth
            )
        rf.fit(X_train, y_train)
        
        # Evaluate performance on the train sample
        print(f"Train MAE: ", mean_absolute_error(y_train, rf.predict(X_train))) #TODO: maybe return and then create a dataframe of metrics?

        # Save the model
        self.model = rf
