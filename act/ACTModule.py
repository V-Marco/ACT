import os

from act.cell_model import TrainCell, TargetCell
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from act.act_types import SimulationParameters, OptimizationParameters, OptimizationParam, PassiveProperties, SimParams
from act.cell_model import ModuleParameters, TargetCell, TrainCell
from act.simulator import Simulator
from act.DataProcessor import DataProcessor

from act.optimizer import RandomForestOptimizer
from act.Metrics import Metrics

class ACTModule:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), "model", params['module_folder_name']) + "/"
        self.target_traces_file = params["target_traces_file"]
        self.train_cell: TrainCell = params["cell"]
        self.sim_params: SimParams = params['sim_params']
        self.optim_params: OptimizationParameters = params['optim_params']
        
        self.blocked_channels = []
        

    def run(self):
        print("RUNNING THE MODULE")
        print("LOADING TARGET TRACES")
        self.convert_csv_to_npy()

        print("SIMULATING TRAINING DATA")
        self.simulate_train_cells(self.train_cell)

        print("TRAINING RANDOM FOREST REGRESSOR")
        prediction = self.get_rf_prediction()

        print("SIMULATING PREDICTIONS")
        self.simulate_eval_cells(self.train_cell, prediction)

        print("SELECTING BEST PREDICTION")
        prediction_eval_method = self.optim_params.get('prediction_eval_method', 'fi_curve')
        if prediction_eval_method == 'fi_curve':
            predicted_g_data_folder, best_prediction = self.evaluate_fi_curves(prediction)
        elif prediction_eval_method == 'voltage':
            predicted_g_data_folder, best_prediction = self.evaluate_v_traces(prediction)
        else:
            print("prediction_eval_method must be 'fi_curve' or 'voltage'.")

        # Save predictions as a conductance dictionary in the train cell
        param_names_list = [param['param'] for param in self.optim_params['g_ranges_slices']]
        self.train_cell.predicted_g = dict(zip(param_names_list, best_prediction))
        
        print(self.train_cell.predicted_g)
        
        return predicted_g_data_folder

    def convert_csv_to_npy(self):
        # Uses the number of traces held in CI_amps to parse a users 2D csv file to 3D npy file
        num_traces = len(self.sim_params['CI_amps'])
        
        data = np.loadtxt(self.output_folder_name + "target/" + self.target_traces_file, delimiter=',', skiprows=1)
        
        # Calculate the number of samples from the number of traces
        csv_num_rows = data.shape[0]
        num_samples = csv_num_rows // num_traces
        
        V_I_data = data.reshape(num_traces, num_samples, 2)
        np.save(self. output_folder_name + 'target/combined_out.npy', V_I_data)


    def get_I_g_combinations(self):
        final_g_ranges_slices: List[OptimizationParam] = []
        for i, optim_param in enumerate(self.optim_params['g_ranges_slices']):
            # Handle Blocked channels
            if optim_param.get('blocked', False):
                #self.blocked_channels.append(optim_param['param'])
                final_g_ranges_slices.append(
                        OptimizationParam(
                            param=optim_param['param'],
                            low=0.0,
                            high=0.0,
                            n_slices=1
                        )
                    )
            elif optim_param.get('prediction', None) != None and optim_param.get('bounds_variation', None) != None:
                final_g_ranges_slices.append(
                    OptimizationParam(
                        param=optim_param['param'],
                        low=optim_param['prediction'] - optim_param['bounds_variation'],
                        high=optim_param['prediction'] + optim_param['bounds_variation'],
                        n_slices=optim_param['n_slices']
                    )
                )
            elif optim_param.get('low', None) != None and optim_param.get('high', None) !=None:
                final_g_ranges_slices.append(
                    OptimizationParam(
                        param=optim_param['param'],
                        low=optim_param['low'],
                        high=optim_param['high'],
                        n_slices=optim_param['n_slices']
                    )
                )
            else: 
                raise Exception("OptimizationParm not defined fully. Need either (low & high) or (prediction & variation).")

        # Now extract the ranges from the final ranges
        channel_ranges = []
        slices = []
        for g_params in final_g_ranges_slices:
            channel_ranges.append((g_params['low'], g_params['high']))
            slices.append(g_params["n_slices"])

        dp = DataProcessor()
        conductance_groups, current_settings = dp.generate_I_g_combinations(channel_ranges, slices, self.sim_params['CI_amps'])

        return conductance_groups, current_settings
    

    def simulate_train_cells(self, train_cell: TrainCell):
        simulator = Simulator(self.output_folder_name)
            
        try:
            conductance_groups, current_settings = self.get_I_g_combinations()
        except Exception as e:
            print(e)
            return
    

        for i in range(len(conductance_groups)):
                # Set parameters from the grid
                train_cell.set_g(train_cell.g_names, conductance_groups[i], self.sim_params)
                simulator.submit_job(
                    train_cell, 
                    SimulationParameters(
                        sim_name = "train",
                        sim_idx = i,
                        h_v_init = self.sim_params['h_v_init'], # (mV)
                        h_tstop = self.sim_params['h_tstop'],  # (ms)
                        h_dt = self.sim_params['h_dt'], # (ms)
                        h_celsius = self.sim_params['h_celsius'], # (deg C)
                        CI = {
                            "type": self.sim_params['CI_type'],
                            "amp": current_settings[i],
                            "dur": self.sim_params['CI_dur'],
                            "delay": self.sim_params['CI_delay']
                        },
                        set_g_to=self.sim_params['set_g_to']
                    )
                )
        
        simulator.run(self.train_cell.mod_folder)

        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + "train")

    def get_rf_prediction(self):
        # Extract Features from TargetCell traces
        dp = DataProcessor()

        dataset_target = np.load(self.output_folder_name + "target/combined_out.npy")
        V_target = dataset_target[:,:,0]
        I_target = dataset_target[:,:,1]

        features_target, columns_target = dp.extract_features(list_of_features=self.optim_params.get('list_of_features',None), V=V_target,I=I_target,inj_dur=self.sim_params['CI_dur'],inj_start=self.sim_params['CI_delay'])

        # Extract Features from TrainCell traces
        dataset_train = np.load(self.output_folder_name + "train/combined_out.npy")

        g_train = dp.clean_g_bars(dataset_train)
        V_train = dataset_train[:,:,0]
        I_train = dataset_train[:,:,1]

        # TODO: ["i_mean_stdev", "v_spike_stats", "v_mean_potential", "v_amplitude_frequency", "v_arima_coefs"]
        features_train, columns_train = dp.extract_features(list_of_features=self.optim_params.get('list_of_features',None),V=V_train,I=I_train,inj_dur=self.sim_params['CI_dur'],inj_start=self.sim_params['CI_delay'])
        print(f"Extracting features: {columns_train}")

        # Train Model on data
        X_train = features_train
        Y_train = g_train

        rf = RandomForestOptimizer(
            n_estimators= self.optim_params.get('n_estimators',1000), 
            random_state=self.optim_params.get('random_state', 42), 
            max_depth=self.optim_params.get('max_depth', None)
            )
        rf.fit(X_train, Y_train)

        # Evaluate the model performance
        n_sim_combos = 1
        for param in self.optim_params['g_ranges_slices']:
            n_sim_combos = n_sim_combos * param.get('n_slices', 1)
            
        n_splits = 10
        if n_sim_combos < 10:
            n_splits = n_sim_combos

        metrics = Metrics()
        metrics.evaluate_random_forest(rf.model, 
                                       X_train, 
                                       Y_train, 
                                       random_state=self.optim_params.get('random_state', 42), 
                                       n_repeats=self.optim_params.get('eval_n_repeats', 3), 
                                       n_splits=n_splits)

        # Predict the conductance values

        X_test = features_target
        predictions = rf.predict(X_test)
        
        # Round off all predictions to 15 decimal places
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                predictions[i][j] = round(predictions[i][j], 15)
                
        print(f"Predicted Conductances for each current injection intensity: {predictions}")

        return predictions


    def simulate_eval_cells(self, eval_cell: TrainCell, predictions):
        simulator = Simulator(self.output_folder_name)
        for i in range(len(predictions)):
            for j in range(len(self.sim_params['CI_amps'])):
                # Set parameters from the grid
                eval_cell.set_g(eval_cell.g_names, predictions[i], self.sim_params)
                simulator.submit_job(
                    eval_cell, 
                    SimulationParameters(
                        sim_name = "prediction_eval"+str(i),
                        sim_idx = i * len(predictions) + j,
                        h_v_init = self.sim_params['h_v_init'], # (mV)
                        h_tstop = self.sim_params['h_tstop'],  # (ms)
                        h_dt = self.sim_params['h_dt'], # (ms)
                        h_celsius = self.sim_params['h_celsius'], # (deg C)
                        CI = {
                            "type": "constant",
                            "amp": self.sim_params['CI_amps'][j],
                            "dur": self.sim_params['CI_dur'],
                            "delay": self.sim_params['CI_delay']
                        },
                        set_g_to=self.sim_params['set_g_to']
                    )
                )
        simulator.run(self.train_cell.mod_folder)

        dp = DataProcessor()
        for i in range(len(predictions)):
            dp.combine_data(self.output_folder_name + "prediction_eval" + str(i))

    def evaluate_fi_curves(self, predictions):

        dp = DataProcessor()

        # Get Target Cell Frequencies
        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]

        target_frequencies = dp.get_fi_curve(V_target, self.sim_params['CI_amps'], inj_dur=self.sim_params['CI_dur']).flatten()

        # Get train2 Cell Frequencies
        FI_data = []
        for i in range(len(predictions)):
            dataset = np.load(self.output_folder_name + "prediction_eval" + str(i) + "/combined_out.npy")
            V_test = dataset[:,:,0]

            #Get FI curve info
            frequencies = dp.get_fi_curve(V_test, self.sim_params['CI_amps'], inj_dur=self.sim_params['CI_dur'])

            FI_data.append(frequencies.flatten())

        list_of_freq = np.array(FI_data)

        # Evaluate FI curves on target FI curve
        metrics = Metrics()

        fi_mae = []
        for fi in list_of_freq:
            fi_mae.append(metrics.mae_score(target_frequencies, fi))

        g_best_idx = fi_mae.index(min(fi_mae))
        
        # Save the predicted and target FI data (for later plotting)
        results_folder = self.output_folder_name + "results/"

        os.makedirs(results_folder, exist_ok=True)
        
        data_to_save = np.array([list_of_freq[g_best_idx], target_frequencies])
        filepath = os.path.join(results_folder, f"frequency_data_{g_best_idx}.npy")
        
        np.save(filepath, data_to_save)
        
        # Get the directory that holds the best prediction simulation data
        best_prediction_data_folder = self.output_folder_name + "prediction_eval" + str(g_best_idx) + "/combined_out.npy"
        
        return best_prediction_data_folder, predictions[g_best_idx]
    
    def evaluate_v_traces(self, predictions):

        dp = DataProcessor()

        # Get Target Cell Frequencies
        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]

        # Get train Cell Frequencies
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

        # Evaluate FI curves on target FI curve
        g_best_idx = mean_mae_list.index(min(mean_mae_list))

        return g_best_idx, predictions[g_best_idx]
