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
        
    

    def run(self) -> str:
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
        self.simulate_train_cells(self.cell)

        if self.optimization_parameters.filter_parameters is not None:
            if self.optimization_parameters.filter_parameters.filtered_out_features is not None:
                print("Filtering...")
                self.filter_data(os.path.join(self.output_folder_name, "train", "combined_out.npy"))
        else:
            print("Filtering skipped.")

        print("Training RandomForest...")
        self.train_random_forest(np.load)

        # Make and evaluate predictions
        print("Predicting on target data...")
        dataset_target = np.load(self.target_traces_file)
        V_target = dataset_target[:, :, 0]
        prediction = self.model.predict(V_target)
        self.simulate_cells(self.cell, prediction)

        print("Evaluating predictions...")
        prediction_eval_method = self.optim_params.prediction_eval_method
        save_file = self.optim_params.save_file
        if not save_file == None:
            save_to_json(prediction_eval_method, "prediction_evaluation_method", save_file)
            
        if prediction_eval_method == 'fi_curve':
            predicted_g_data_file, best_prediction = self.evaluate_fi_curves(prediction)
            self.evaluate_v_traces(prediction)
            self.evaluate_features(prediction)
        elif prediction_eval_method == 'voltage':
            predicted_g_data_file, best_prediction = self.evaluate_v_traces(prediction)
            self.evaluate_fi_curves(prediction)
            self.evaluate_features(prediction)
        elif prediction_eval_method == 'features':
            predicted_g_data_file, best_prediction = self.evaluate_features(prediction)
            self.evaluate_fi_curves(prediction)
            self.evaluate_v_traces(prediction)
        else:
            print("prediction_eval_method must be 'fi_curve' or 'voltage'.")

        conductance_option_names_list = [conductance_option.variable_name for conductance_option in self.optim_params.conductance_options]
        final_prediction = dict(zip(conductance_option_names_list, best_prediction))
        self.train_cell.prediction = final_prediction
        
        print(self.train_cell.prediction)
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            save_to_json(final_prediction, "final_g_prediction", save_file)
            
        end_time = time.time()
        run_time = end_time - start_time
        runtime_timedelta = timedelta(seconds = run_time)
        formatted_runtime = str(runtime_timedelta)
        print(f"Done. Finished in {formatted_runtime} sec.\n")
        
        previous_modules = self.optim_params.previous_modules
        total_runtime = run_time
        if not previous_modules == None:
            
            for module in previous_modules:
                base_directory = self.output_folder_name.replace("/module_final", "")
                prev_module_file = base_directory + module + "/results/saved_metrics.json"
                with open(prev_module_file, 'r') as file:
                    data = json.load(file)
                
                module_time = data.get('module_runtime', None)
                total_runtime += module_time
        
        if not save_file == None:
            save_to_json(total_runtime, "module_runtime", save_file)
            save_to_json(predicted_g_data_file, "predicted_g_data_file", save_file)
        
        return predicted_g_data_file
    

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

    def simulate_cells(self, cell: ACTCellModel, g_cobm: list = None) -> None:

        if g_cobm is None:
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
            specific_cell = ACTCellModel(cell_name = cell.cell_name,
                                               path_to_hoc_file = cell.path_to_hoc_file,
                                               path_to_mod_files = cell.path_to_mod_files,
                                               passive = cell.passive,
                                               active_channels = cell.active_channels,
                                               prediction = cell.prediction)
            
            # Set conductances
            specific_cell.set_g_bar(specific_cell.active_channels, list(g_comb[group_id]))

            # Submit the job
            simulator.submit_job(
                specific_cell, 
                SimulationParameters(
                    sim_name = mode,
                    sim_idx = group_id,
                    h_v_init = self.simulation_parameters.h_v_init,    # (mV)
                    h_tstop = self.simulation_parameters.h_tstop,      # (ms)
                    h_dt = self.simulation_parameters.h_dt,            # (ms)
                    h_celsius = self.simulation_parameters.h_celsius,  # (deg C)
                    CI = self.simulation_parameters.CI
                )
            )

        simulator.run_jobs()
        combine_data(os.path.join(self.output_folder_name, mode))
        
        
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
        dataset_target = np.load(self.target_traces_file)
        V_target = dataset_target[:, :, 0]
        lto_hto = dataset_target[:, 1, 2]

        # Load target data
        #TODO: think if there is a better way to do it
        target_df = get_summary_features(
            V = V_target, 
            CI = self.simulation_parameters.CI,
            lto_hto = 0, #TODO: fix
            spike_threshold = self.optimization_parameters.spike_threshold,
            max_n_spikes = self.optimization_parameters.max_n_spikes
            )

        # If train_features is None, use all features
        if self.optimization_parameters.train_features is not None:
            target_df = select_features(target_df, self.optimization_parameters.train_features)
        
        file_path = os.path.join(self.output_folder_path, "train", "filtered_out.npy")
        if os.path.exists(file_path):
            dataset_train = np.load(file_path)
        else:
            dataset_train = np.load(os.path.join(self.output_folder_path, "train", "combined_out.npy"))

        # Construct a train dataset
        g_train = clean_g_bars(dataset_train)
        V_train = dataset_train[:, :, 0]
        lto_hto = dataset_train[:, 1, 3]
        
        train_df = get_summary_features(
            V = V_train, 
            CI = self.simulation_parameters.CI,
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
            n_estimators = self.optim_params.n_estimators,
            random_state = self.sim_params.random_seed,
            max_depth = self.optim_params.max_depth
            )
        rf.fit(X_train, y_train)
        
        # Evaluate performance on the train sample
        print(f"Train MAE: ", mean_absolute_error(y_train, rf.predict(X_train))) #TODO: maybe return and then create a dataframe of metrics?

        # Save the model
        self.model = rf
    

    def evaluate_fi_curves(self, predictions: list) -> tuple:
        '''
        Grades the predicted cells on min FI curve MAE.
        Parameters:
        -----------
        self
        
        predictions: list
            Conductance predictions
            
        Returns:
        ----------
        best_prediction_data_file: str
            Filepath to best (smallest MAE) prediction sim data
        
        predictions: list[float]
            Best (smallest MAE) set of predicted conductances
        '''

        dataset = np.load(self.target_traces_file)

        V_target = dataset[:,:,0]

        target_frequencies = get_fi_curve(V_target, -40, self.sim_params.CI).flatten()
        
        FI_data = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]

            frequencies = get_fi_curve(V_test, -40, self.sim_params.CI)

            FI_data.append(frequencies.flatten())

        list_of_freq = np.array(FI_data)

        fi_mae = []
        for fi in list_of_freq:
            fi_mae.append(mean_absolute_error(target_frequencies, fi))
        
        print(f"FI curve MAE for each prediction: {fi_mae}")

        g_best_idx = fi_mae.index(min(fi_mae))
        
        results_folder = self.output_folder_name + "results/"

        os.makedirs(results_folder, exist_ok=True)
        
        data_to_save = np.array([list_of_freq[g_best_idx], target_frequencies])
        filepath = os.path.join(results_folder, f"frequency_data.npy")
        
        np.save(filepath, data_to_save)
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            save_to_json(min(fi_mae), "final_prediction_fi_mae", save_file)
        
        return best_prediction_data_file, predictions[g_best_idx]
    
    
    def evaluate_v_traces(self, predictions: list) -> tuple:
        '''
        Grades the predicted cells on min voltage trace MAE.
        Parameters:
        -----------
        self
        
        predictions: list
            Conductance predictions
            
        Returns:
        ----------
        best_prediction_data_file: str
            Filepath to best (smallest MAE) prediction sim data
        
        predictions: list[float]
            Best (smallest MAE) set of predicted conductances
        '''
        dataset = np.load(self.target_traces_file)

        V_target = dataset[:,:,0]

        mean_mae_list = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]
            prediction_MAEs = []
            
            for j in range(len(V_test)):
                v_mae = mean_absolute_error(V_test[j],V_target[j])
                prediction_MAEs.append(v_mae)

            mean_mae_list.append(np.mean(prediction_MAEs))
            
        print(f"Mean voltage mae for each prediction: {mean_mae_list}")

        g_best_idx = mean_mae_list.index(min(mean_mae_list))
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            save_to_json(min(mean_mae_list), "final_prediction_voltage_mae", save_file)

        return best_prediction_data_file, predictions[g_best_idx]
    
    
    def evaluate_features(self, predictions: list) -> tuple:
        '''
        Grades the predicted cells on min feature MAE.
        Parameters:
        -----------
        self
        
        predictions: list
            Conductance predictions
            
        Returns:
        ----------
        best_prediction_data_file: str
            Filepath to best (smallest MAE) prediction sim data
        
        predictions: list[float]
            Best (smallest MAE) set of predicted conductances
        '''
        dataset = np.load(self.target_traces_file)

        V_target = dataset[:,:,0]
        I_target = dataset[:,:,1]
        lto_hto = dataset[:,1,2]
        
        train_features = self.optim_params.train_features
        threshold = self.optim_params.spike_threshold
        first_n_spikes = self.optim_params.first_n_spikes
        dt = self.sim_params.h_dt
        
        target_df = get_summary_features(V=V_target,I=I_target, lto_hto=lto_hto, current_inj_combos=self.sim_params.CI, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
        sub_target_df = select_features(target_df, train_features)

        mean_mae_list = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]
            I_test = dataset[:,:,1]
            lto_hto = dataset[:,1,3]
            test_df = get_summary_features(V=V_test,I=I_test, lto_hto=lto_hto, current_inj_combos=self.sim_params.CI, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
            sub_test_df = select_features(test_df, train_features)
            
            prediction_MAEs = []
            
            for j in range(len(V_test)):
                v_mae = mean_absolute_error(sub_target_df.loc[j], sub_test_df.loc[j])
                prediction_MAEs.append(v_mae)

            mean_mae_list.append(np.mean(prediction_MAEs))

        print(f"Mean Feature MAE for each prediction: {mean_mae_list}")
        g_best_idx = mean_mae_list.index(min(mean_mae_list))
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            save_to_json(min(mean_mae_list), "summary_stats_mae_final_prediction", save_file)

        return best_prediction_data_file, predictions[g_best_idx]
