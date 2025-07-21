import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from itertools import product

from act.types import SimulationParameters, OptimizationParameters
from act.cell_model import ACTCellModel
from act.simulator import ACTSimulator
from act.data_processing import combine_data, remove_saturated_traces, get_traces_with_spikes, get_summary_features, select_features, clean_g_bars
from act.metrics import summary_features_error

class ACTModule:

    def __init__(
            self, 
            name: str,
            cell: ACTCellModel, 
            simulation_parameters: SimulationParameters,
            optimization_parameters: OptimizationParameters,
            target_file: str):
        """
        Initialize an optimization module.

        Parameters:
        ----------
        name: str
            Module name.

        cell: ACTCellModel
            Cell model to simulate.
        
        simulation_parameters: SimulationParameters
            Simulation parameters.

        optimization_parameters: OptimizationParameters
            Optimization parameters.

        target_file: str
            File with target responses. If .csv, it is assumed that the file contains summary features.
            If .npy, it is assumed that the file contains current-voltage traces, and summary features will be computed by the module.
        """

        # Module name
        self.name = name
        self.output_folder_path = os.path.join(os.getcwd(), f"module_{self.name}")

        self.cell = cell
        self.simulation_parameters = simulation_parameters
        self.optimization_parameters = optimization_parameters
        self.target_file = target_file

        # Model (assigned after fitting)
        self.model = None

    def _read_process_target_data(self) -> pd.DataFrame:
        print("Predicting on target data...")
        
        if self.target_file.endswith(".npy"):
            dataset_target = np.load(self.target_file)
            V_target = dataset_target[:, :, 0]
            I_target = dataset_target[:, :, 1]

            # Compute target SF
            target_df = get_summary_features(
                V = V_target, 
                I = I_target,
                spike_threshold = self.optimization_parameters.spike_threshold,
                max_n_spikes = self.optimization_parameters.max_n_spikes
                )
        else: # .csv
            target_df = pd.read_csv(self.target_file)

        # If train_features is None, use all features
        if self.optimization_parameters.train_features is not None:
            target_df = select_features(target_df, self.optimization_parameters.train_features)
        
        return target_df

    def run(self) -> pd.DataFrame:
        """
        Run the train-predict sequence.

        Returns:
        -----------
        metrics: pd.DataFrame
            Evaluation metrics.
        """
        start_time = time.time()
        print(f"Running Module '{self.name}'...")
        print("----------")

        print("Simulating train traces...")
        self._simulate_cells(self.cell)

        if self.optimization_parameters.filter_parameters is not None:
            if self.optimization_parameters.filter_parameters.filtered_out_features is not None:
                print("Filtering...")
                self._filter_data(os.path.join(self.output_folder_name, "train", "combined_out.npy"))
        else:
            print("Filtering skipped.")

        print("Training RandomForest...")
        train_mae = self._train_random_forest()

        # Make and evaluate predictions
        target_df = self._read_process_target_data()
        prediction = self.model.predict(target_df)
        self._simulate_cells(self.cell, prediction)

        print("Evaluating predictions...")
        sf_error, fi_error, g_pred = self._evaluate_predictions()

        conductance_option_names_list = [conductance_option.variable_name for conductance_option in self.optimization_parameters.conductance_options]
        final_prediction = dict(zip(conductance_option_names_list, g_pred[np.argmin(sf_error)]))
        self.cell.prediction = final_prediction
        
        print(self.cell.prediction)
        runtime = round(time.time() - start_time, 3)

        metric_names = ["Train MAE (g)"] + [f"Test SFE (g{g_id})" for g_id in range(len(g_pred))] + ["Test MAE (FI)", "Runtime (s)"]
        metrics = [train_mae] + sf_error.flatten().tolist() + [fi_error, runtime]
        metrics = pd.DataFrame({"metric" : metric_names, "value": metrics})

        print(f"Done.")

        return metrics
    
    def _evaluate_predictions(self):
        # Load target data
        target_df = self._read_process_target_data()

        # Load predicted data
        dataset_pred = np.load(os.path.join(self.output_folder_path, "eval", "combined_out.npy"))

        # Construct the dataset
        g_pred = clean_g_bars(dataset_pred)
        V_pred = dataset_pred[:, :, 0]
        I_pred = dataset_pred[:, :, 1]
        
        pred_df = get_summary_features(
            V = V_pred, 
            I = I_pred,
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )
        
        if self.optimization_parameters.train_features is not None:
            pred_df = select_features(pred_df, self.optimization_parameters.train_features)
        
        # Compute errors
        sf_error = summary_features_error(target_df.to_numpy(), pred_df.to_numpy())
        fi_error = mean_absolute_error(target_df['spike_frequency'].to_numpy(), pred_df['spike_frequency'].to_numpy())
        
        return sf_error, fi_error, g_pred

    def _generate_g_combinations(self) -> None:

        g = []
        n_slices = []
        for conductance_option in self.optimization_parameters.conductance_options:
            # If blocked, set the range to (0, 0)
            if conductance_option.blocked:
                g.append((0, 0)) # (low, high)
                n_slices.append(1)

            # Or, if was optimized before, set bounds variation
            elif (self.cell.prediction[conductance_option.variable_name] != None) and (conductance_option.bounds_variation != None):
                past_prediction = self.cell.prediction[conductance_option.variable_name]
                g.append((
                    past_prediction - past_prediction * conductance_option.bounds_variation,
                    past_prediction + past_prediction * conductance_option.bounds_variation
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

    def _simulate_cells(self, cell: ACTCellModel, g_comb: list = None) -> None:

        if g_comb is None:
            mode = "train"
        else:
            mode = "eval"

        # Set the simulator
        simulator = ACTSimulator(self.output_folder_path)

        # Set self.conductance_combos and self.current_inj_combos
        if mode == "train":
            print("Generating conductance combinations...")
            g_comb = self._generate_g_combinations()
        elif mode == "eval": # else, g_comb is given as RF predictions
            if len(g_comb) != len(self.optimization_parameters.CI_options):
                raise ValueError("The number of CI options must match the number of target traces.")

        sim_counter = 0
        for group_id in range(len(g_comb)):
            
            # Init a cell for every combo
            specific_cell = ACTCellModel(
                cell_name = cell.cell_name, 
                path_to_hoc_file = cell.path_to_hoc_file,
                path_to_mod_files = cell.path_to_mod_files,
                passive = cell.passive,
                active_channels = cell.active_channels)
            
            # Set cell builder
            if cell._custom_cell_builder is not None:
                specific_cell.set_custom_cell_builder(cell._custom_cell_builder)
            
            # Set conductances
            specific_cell.set_g_bar(specific_cell.active_channels, list(g_comb[group_id]))

            # Submit the job
            for ci_id, curr_inj in enumerate(self.optimization_parameters.CI_options):
                # During prediction, simulate only one I for each g
                if (mode == "eval") and (ci_id != group_id):
                    continue
                
                # During training, simulate all possible I-g combinations
                simulator.submit_job(
                    specific_cell, 
                    SimulationParameters(
                        sim_name = mode,
                        sim_idx = sim_counter,
                        h_v_init = self.simulation_parameters.h_v_init,    # (mV)
                        h_tstop = self.simulation_parameters.h_tstop,      # (ms)
                        h_dt = self.simulation_parameters.h_dt,            # (ms)
                        h_celsius = self.simulation_parameters.h_celsius,  # (deg C)
                        CI = [curr_inj]
                    )
                )
                sim_counter += 1
        
        print("Simulating cells...")
        if not self.optimization_parameters.n_cpus == None:
            simulator.run_jobs(self.optimization_parameters.n_cpus)
        else:
            simulator.run_jobs()
        combine_data(os.path.join(self.output_folder_path, mode))
        
    def _filter_data(self, path) -> None:

        filtered_out_features = self.optimization_parameters.filter_parameters.filtered_out_features
        data = np.load(path)
                           
        if "saturated" in filtered_out_features:
            window_of_inspection = self.optimization_parameters.filter_parameters.window_of_inspection
            if window_of_inspection is None:
                inspection_start = self.simulation_parameters.CI[0].delay + int(self.simulation_parameters.CI[0].dur * 0.7)
                inspection_end = self.simulation_parameters.CI[0].delay + self.simulation_parameters.CI[0].dur
                window_of_inspection = (inspection_start, inspection_end)
            saturation_threshold = self.optimization_parameters.filter_parameters.saturation_threshold
            data = remove_saturated_traces(data, window_of_inspection, threshold = saturation_threshold)
        
        if "no_spikes" in filtered_out_features:
            spike_threshold = self.optimization_parameters.spike_threshold
            data = get_traces_with_spikes(data, spike_threshold = spike_threshold)
         
        print(f"Dataset size after filtering: {len(data)}. Saving to {os.path.join(path, 'filtered_out.npy')}")
        np.save(os.path.join(path, f"filtered_out.npy"), data)
    

    def _train_random_forest(self) -> float:
        """
        Trains a Random Forest Regressor model on the features of the generated simulation data.
        Then gets a prediction for conductance sets that yeild features found in the target data.
        Then calculates an evaluation of the RF model and saves the metric.
        
        Returns:
        -----------
        predictions: np.ndarray
            Conductance Predictions
        """
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
        
        train_df = get_summary_features(
            V = V_train, 
            I = I_train,
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )
        
        if self.optimization_parameters.train_features is not None:
            train_df = select_features(train_df, self.optimization_parameters.train_features)

        X_train = train_df.to_numpy()
        y_train = g_train

        # Fit RF
        rf = RandomForestRegressor(
            n_estimators = self.optimization_parameters.n_estimators,
            random_state = self.optimization_parameters.random_state,
            max_depth = self.optimization_parameters.max_depth
            )
        rf.fit(X_train, y_train)
        
        # Save the model
        self.model = rf

        # Evaluate performance on the train sample
        train_mae = mean_absolute_error(y_train, rf.predict(X_train))
        return train_mae

