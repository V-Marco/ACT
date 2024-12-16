import os
import time
from datetime import timedelta
import numpy as np
from typing import List
import pickle
import json

from act.act_types import SimulationParameters, OptimizationParameters, ConductanceOptions, CurrentInjection
from act.cell_model import TrainCell
from act.module_parameters import ModuleParameters
from act.simulator import ACTSimulator
from act.DataProcessor import DataProcessor
from act.optimizer import RandomForestOptimizer
from act.Metrics import Metrics

'''
ACTModule is the primary class in the Automatic Cell Tuner project with methods to generate
training data, run a random forest regressor to learn conductance sets to match features, and
additional methods to evaluate predictions.
'''

class ACTModule:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), params.module_folder_name) + "/"
        self.target_traces_file = params.target_traces_file
        self.train_cell: TrainCell = params.cell
        self.sim_params: SimulationParameters = params.sim_params
        self.optim_params: OptimizationParameters = params.optim_params
        
        self.blocked_channels = []
        self.rf_model = self.optim_params.rf_model
    
    '''
    run
    The Main method to run the automatic cell tuning process given user settings
    '''

    def run(self):
        start_time = time.time()
        print("RUNNING THE MODULE")
        print("LOADING TARGET TRACES")
        self.convert_csv_to_npy()

        if self.rf_model == None:
            print("SIMULATING TRAINING DATA")
            self.simulate_train_cells(self.train_cell)
            self.filter_data()


        prediction = self.get_rf_prediction()

        print("SIMULATING PREDICTIONS")
        self.simulate_eval_cells(self.train_cell, prediction)

        print("SELECTING BEST PREDICTION")
        dp = DataProcessor()
        prediction_eval_method = self.optim_params.prediction_eval_method
        save_file = self.optim_params.save_file
        if not save_file == None:
            dp.save_to_json(prediction_eval_method, "prediction_evaluation_method", save_file)
            
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
        self.train_cell.predicted_g = final_prediction
        
        print(self.train_cell.predicted_g)
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            dp.save_to_json(final_prediction, "final_g_prediction", save_file)
            
        end_time = time.time()
        run_time = end_time - start_time
        runtime_timedelta = timedelta(seconds=run_time)

        formatted_runtime = str(runtime_timedelta)
        
        print(f"Module runtime: {formatted_runtime}")
        
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
            dp.save_to_json(total_runtime, "module_runtime", save_file)
            dp.save_to_json(predicted_g_data_file, "predicted_g_data_file", save_file)
        
        return predicted_g_data_file
    
    '''
    convert_csv_to_npy
    Helper function for users with voltage trace data in CSV format
    '''

    def convert_csv_to_npy(self):
        num_traces = len(self.sim_params.CI)
        
        data = np.loadtxt(self.target_traces_file, delimiter=',', skiprows=1)
        
        csv_num_rows = data.shape[0]
        num_samples = csv_num_rows // num_traces
        
        V_I_data = data.reshape(num_traces, num_samples, 2)
        os.makedirs(self.output_folder_name + 'target/', exist_ok=True)
        np.save(self.output_folder_name + 'target/combined_out.npy', V_I_data)

    '''
    get_I_g_combinations
    Takes conductance ranges set by the user along with the number of slices between this range.
    Also takes current injection intensities also set by the user.
    Generates cartesian product of these settings to get combination lists for conductances
    and current injection intensities in a format that can be processed by NEURON
    '''

    def get_I_g_combinations(self):
        print("Getting Conductance Combinations From Preselected Ranges")
        final_g_ranges_slices: List[ConductanceOptions] = []
        for i, conductance_option in enumerate(self.optim_params.conductance_options):
            if conductance_option.blocked:
                final_g_ranges_slices.append(
                        ConductanceOptions(
                            variable_name=conductance_option.variable_name,
                            low=0.0,
                            high=0.0,
                            n_slices=1
                        )
                    )
            elif conductance_option.prediction != None and conductance_option.bounds_variation != None:
                final_g_ranges_slices.append(
                    ConductanceOptions(
                        variable_name=conductance_option.variable_name,
                        low=conductance_option.prediction - (conductance_option.prediction * conductance_option.bounds_variation),
                        high=conductance_option.prediction + (conductance_option.prediction * conductance_option.bounds_variation),
                        n_slices=conductance_option.n_slices
                    )
                )
            elif conductance_option.low != None and conductance_option.high !=None:
                if conductance_option.n_slices == 1:
                  final_g_ranges_slices.append(
                        ConductanceOptions(
                            variable_name=conductance_option.variable_name,
                            low=conductance_option.low + ((conductance_option.high - conductance_option.low)/2),
                            high=conductance_option.low + ((conductance_option.high - conductance_option.low)/2),
                            n_slices=conductance_option.n_slices
                        )
                    )
                else:
                    final_g_ranges_slices.append(
                        ConductanceOptions(
                            variable_name=conductance_option.variable_name,
                            low=conductance_option.low,
                            high=conductance_option.high,
                            n_slices=conductance_option.n_slices
                        )
                    )
            else: 
                raise Exception("OptimizationParameters not defined fully. Need either (low & high) or (prediction & variation).")

        channel_ranges = []
        slices = []
        for conductance_option in final_g_ranges_slices:
            channel_ranges.append((conductance_option.low, conductance_option.high))
            slices.append(conductance_option.n_slices)

        dp = DataProcessor()
        conductance_groups, current_settings = dp.generate_I_g_combinations(channel_ranges, slices, self.sim_params.CI)

        return conductance_groups, current_settings
    
    '''
    simulate_train_cells
    Takes user settings, gets conductance and current injection intensity combinations, and
    simulates these cell settings using ACTSimulator to generate training data for the Model.
    '''

    def simulate_train_cells(self, train_cell: TrainCell):
        simulator = ACTSimulator(self.output_folder_name)
            
        try:
            conductance_groups, current_settings = self.get_I_g_combinations()
        except Exception as e:
            print(e)
            return
    
        self.sim_params.set_g_to = []
        for i in range(len(conductance_groups)):
                train_cell.set_g(train_cell.active_channels, conductance_groups[i], self.sim_params)
                simulator.submit_job(
                    train_cell, 
                    SimulationParameters(
                        sim_name = "train",
                        sim_idx = i,
                        h_v_init = self.sim_params.h_v_init,    # (mV)
                        h_tstop = self.sim_params.h_tstop,      # (ms)
                        h_dt = self.sim_params.h_dt,            # (ms)
                        h_celsius = self.sim_params.h_celsius,  # (deg C)
                        CI = [CurrentInjection(
                            type=self.sim_params.CI[0].type,
                            amp=current_settings[i],
                            dur=self.sim_params.CI[0].dur,
                            delay=self.sim_params.CI[0].delay
                        )],
                        set_g_to=self.sim_params.set_g_to
                    )
                )
        
        simulator.run_jobs()

        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + "train")
        
    '''
    filter_data
    An optional process to filter training data by features such as saturated voltage trace
    or no spiking.
    '''
        
    def filter_data(self):
        dp = DataProcessor()
        
        filtered_out_features = self.optim_params.filter_parameters.filtered_out_features
        if filtered_out_features is None:
            return
        else:
            print("FILTERING DATA")
            data = np.load(self.output_folder_name + "train/combined_out.npy")
            if "saturated" in filtered_out_features:
                window_of_inspection = self.optim_params.filter_parameters.window_of_inspection
                if window_of_inspection is None:
                    inspection_start = self.sim_params.CI[0].delay + int(self.sim_params.CI[0].dur * 0.7)
                    inspection_end = self.sim_params.CI[0].delay + self.sim_params.CI[0].dur
                    window_of_inspection = (inspection_start, inspection_end)
                saturation_threshold = self.optim_params.filter_parameters.saturation_threshold
                dt = self.sim_params.h_dt
                data = dp.get_nonsaturated_traces(data,window_of_inspection, threshold=saturation_threshold,dt=dt)
            
            if "no_spikes" in filtered_out_features:
                spike_threshold = self.optim_params.spike_threshold
                data = dp.get_traces_with_spikes(data,spike_threshold=spike_threshold,dt=dt)
            
            output_path = self.output_folder_name + "train"    
            print(f"Dataset size after filtering: {len(data)}. Saving to {os.path.join(output_path, 'filtered_out.npy')}")
            np.save(os.path.join(output_path, f"filtered_out.npy"), data)
    
    '''
    get_rf_prediction
    Trains a Random Forest Regressor model on the features of the generated simulation data.
    Then gets a prediction for conductance sets that yeild features found in the target data.
    Then calculates an evaluation of the RF model and saves the metric.
    '''

    def get_rf_prediction(self):
        dp = DataProcessor()

        dataset_target = np.load(self.output_folder_name + "target/combined_out.npy")
        V_target = dataset_target[:,:,0]
        I_target = dataset_target[:,:,1]
        
        threshold = self.optim_params.spike_threshold
        first_n_spikes = self.optim_params.first_n_spikes
        dt = self.sim_params.h_dt

        features_target, columns_target = dp.extract_features(train_features=self.optim_params.train_features, V=V_target,I=I_target,threshold=threshold,num_spikes=first_n_spikes,dt=dt)

        if self.rf_model == None:
            print("TRAINING RANDOM FOREST REGRESSOR")
            if os.path.isfile(self.output_folder_name + "train/filtered_out.npy"):
                print("Training Random Forest on Filtered Data")
                dataset_train = np.load(self.output_folder_name + "train/filtered_out.npy")
            else:
                dataset_train = np.load(self.output_folder_name + "train/combined_out.npy")

            g_train = dp.clean_g_bars(dataset_train)
            V_train = dataset_train[:,:,0]
            I_train = dataset_train[:,:,1]

            features_train, columns_train = dp.extract_features(train_features=self.optim_params.train_features,V=V_train,I=I_train,threshold=threshold,num_spikes=first_n_spikes,dt=dt)
            print(f"Extracting features: {columns_train}")

            X_train = features_train
            Y_train = g_train

            rf = RandomForestOptimizer(
                n_estimators= self.optim_params.n_estimators,
                random_state=self.optim_params.random_state,
                max_depth=self.optim_params.max_depth
                )
            rf.fit(X_train, Y_train)
            
            self.rf_model = rf
            
            n_sim_combos = 1
            for conductance_option in self.optim_params.conductance_options:
                n_sim_combos = n_sim_combos * conductance_option.n_slices
                
            n_splits = 10
            if n_sim_combos < 10:
                n_splits = n_sim_combos

            metrics = Metrics()
            save_file = self.optim_params.save_file
            metrics.evaluate_random_forest(rf.model, 
                                        X_train, 
                                        Y_train, 
                                        random_state=self.optim_params.random_state, 
                                        n_repeats=self.optim_params.eval_n_repeats,
                                        n_splits=n_splits,
                                        save_file=save_file)
        else:
            try:
                with open(self.rf_model, 'rb') as file:
                    rf = pickle.load(file)
                print("Model loaded successfully.")
            except FileNotFoundError:
                print("Error: The model file {self.rf_model} was not found.")
                raise
            except pickle.UnpicklingError:
                print("Error: The file is corrupted or is not a valid pickle file.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred while loading the model: {str(e)}")
                raise

        X_test = features_target
        predictions = rf.predict(X_test)
                
        print("Predicted Conductances for each current injection intensity: ")
        print(predictions)

        return predictions
    
    '''
    pickle_rf
    Saves the trained RF model
    '''
    
    def pickle_rf(self, rf_model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(rf_model, file)

    '''
    simulate_eval_cells
    The RF model outputs n predictions where n is the number of current injection intensities.
    This method simulates the predicted conductances.
    '''

    def simulate_eval_cells(self, eval_cell: TrainCell, predictions):
        self.sim_params.set_g_to = []
        
        simulator = ACTSimulator(self.output_folder_name)
        sim_index = 0
        for i in range(len(predictions)):
            for j in range(len(self.sim_params.CI)):
                eval_cell.set_g(eval_cell.active_channels, predictions[i], self.sim_params)
                simulator.submit_job(
                    eval_cell, 
                    SimulationParameters(
                        sim_name = "prediction_eval"+str(i),
                        sim_idx = sim_index,
                        h_v_init = self.sim_params.h_v_init,   # (mV)
                        h_tstop = self.sim_params.h_tstop,     # (ms)
                        h_dt = self.sim_params.h_dt,           # (ms)
                        h_celsius = self.sim_params.h_celsius, # (deg C)
                        CI = [CurrentInjection(type=self.sim_params.CI[0].type, 
                                               amp=self.sim_params.CI[j].amp,
                                               dur=self.sim_params.CI[0].dur,
                                               delay=self.sim_params.CI[0].delay)],
                        set_g_to=self.sim_params.set_g_to
                    )
                )
                sim_index+=1
                
        simulator.run_jobs()

        dp = DataProcessor()
        for i in range(len(predictions)):
            dp.combine_data(self.output_folder_name + "prediction_eval" + str(i))
    
    '''
    evaluate_fi_curves
    Grades the predicted cells on min FI curve MAE.
    '''

    def evaluate_fi_curves(self, predictions):
        dp = DataProcessor()

        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]

        target_frequencies = dp.get_fi_curve(V_target, self.sim_params.CI).flatten()
        
        FI_data = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]

            frequencies = dp.get_fi_curve(V_test, self.sim_params.CI)

            FI_data.append(frequencies.flatten())

        list_of_freq = np.array(FI_data)

        metrics = Metrics()

        fi_mae = []
        for fi in list_of_freq:
            fi_mae.append(metrics.mae_score(target_frequencies, fi))
        
        print(f"FI curve MAE for each prediction: {fi_mae}")

        g_best_idx = fi_mae.index(min(fi_mae))
        
        results_folder = self.output_folder_name + "results/"

        os.makedirs(results_folder, exist_ok=True)
        
        data_to_save = np.array([list_of_freq[g_best_idx], target_frequencies])
        filepath = os.path.join(results_folder, f"frequency_data.npy")
        
        np.save(filepath, data_to_save)
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        dp = DataProcessor()
        save_file = self.optim_params.save_file
        if not save_file == None:
            dp.save_to_json(min(fi_mae), "final_prediction_fi_mae", save_file)
        
        return best_prediction_data_file, predictions[g_best_idx]
    
    '''
    evaluate_v_traces
    Grades the predicted cells on min voltage trace MAE.
    '''
    
    def evaluate_v_traces(self, predictions):
        dp = DataProcessor()

        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]

        metrics = Metrics()
        mean_mae_list = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]
            prediction_MAEs = []
            
            for j in range(len(V_test)):
                v_mae = metrics.mae_score(V_test[j],V_target[j])
                prediction_MAEs.append(v_mae)

            mean_mae_list.append(np.mean(prediction_MAEs))
            
        print(f"Mean voltage mae for each prediction: {mean_mae_list}")

        g_best_idx = mean_mae_list.index(min(mean_mae_list))
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            dp.save_to_json(min(mean_mae_list), "final_prediction_voltage_mae", save_file)

        return best_prediction_data_file, predictions[g_best_idx]
    
    '''
    evaluate_features
    Grades the predicted cells on min feature MAE.
    '''
    
    def evaluate_features(self, predictions):
        dp = DataProcessor()

        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]
        I_target = dataset[:,:,1]
        
        train_features = self.optim_params.train_features
        threshold = self.optim_params.spike_threshold
        first_n_spikes = self.optim_params.first_n_spikes
        dt = self.sim_params.h_dt
        
        target_V_features, _ = dp.extract_features(train_features=train_features, V=V_target,I=I_target, threshold=threshold, num_spikes=first_n_spikes, dt=dt)

        metrics = Metrics()
        mean_mae_list = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]
            I_test = dataset[:,:,1]
            test_V_features, _ = dp.extract_features(train_features=train_features, V=V_test,I=I_test, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
            
            prediction_MAEs = []
            
            for j in range(len(V_test)):
                v_mae = metrics.mae_score(test_V_features[j],target_V_features[j])
                prediction_MAEs.append(v_mae)

            mean_mae_list.append(np.mean(prediction_MAEs))

        print(f"Mean Feature MAE for each prediction: {mean_mae_list}")
        g_best_idx = mean_mae_list.index(min(mean_mae_list))
        
        best_prediction_data_file = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        save_file = self.optim_params.save_file
        if not save_file == None:
            dp.save_to_json(min(mean_mae_list), "summary_stats_mae_final_prediction", save_file)

        return best_prediction_data_file, predictions[g_best_idx]
