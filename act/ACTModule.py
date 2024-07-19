import os

from act.cell_model import TrainCell, TargetCell
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from act.act_types import ModuleParameters, Cell, SimulationParameters, OptimizationParameters, OptimizationParam, PassiveProperties, SimParams
from act.cell_model import TargetCell, TrainCell
from act.simulator import Simulator
from act.DataProcessor import DataProcessor

from act.optimizer import RandomForestOptimizer
from act.Metrics import Metrics

class ACTModule:

    def __init__(self, params: ModuleParameters):

        self.output_folder_name: str = os.path.join(os.getcwd(), "model", params['module_folder_name']) + "/"
        self.target_traces_file = params["target_traces_file"]
        self.cell: Cell = params["cell"]
        passive_props: PassiveProperties = params['passive_properties']
        self.sim_params: SimParams = params['sim_params']
        self.optim_params: OptimizationParameters = params['optim_params']
        

    def run(self):
        print("RUNNING THE MODULE")

        target_cell = TargetCell(
            hoc_file = self.cell['hoc_file'],
            mod_folder= self.cell['modfiles_folder'],
            cell_name = self.cell['cell_name'],
            g_names = self.cell['g_names']
        )

        print("LOADING TARGET TRACES")
        self.convert_csv_to_npy()

        print("SIMULATING TRAINING DATA")
        train_cell = TrainCell(
            hoc_file = self.cell['hoc_file'],
            mod_folder= self.cell['modfiles_folder'],
            cell_name = self.cell['cell_name'],
            g_names = self.cell['g_names']
            )
        
        self.simulate_train_cells(train_cell)

        print("TRAINING RANDOM FOREST REGRESSOR")
        prediction = self.get_rf_prediction()

        print("SIMULATING PREDICTIONS")
        test_cell = TrainCell(
            hoc_file = self.cell['hoc_file'],
            mod_folder= self.cell['modfiles_folder'],
            cell_name = self.cell['cell_name'],
            g_names = self.cell['g_names']
        )

        self.simulate_test_cells(test_cell, prediction)

        print("SELECTING BEST PREDICTION")
        g_final_idx, final_prediction, FI_predicted, target_frequencies = self.evaluate_fi_curves(prediction)

        return g_final_idx, final_prediction, FI_predicted, target_frequencies

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
            if optim_param['low'] == optim_param['high']:
                #if low == high, this may signal that we have a previous prediction.
                #then we should check if there is a set learned variability for this channel
                if not self.learned_variability[i] == None:
                    final_g_ranges_slices.append(
                        OptimizationParam(
                            param=optim_param['param'],
                            low=optim_param['low'] - self.optim_params['learned_variability'][i],
                            high=optim_param['high'] + self.optim_params['learned_variability'][i],
                            n_slices=optim_param['n_slices']
                        )
                    )
            else:
                final_g_ranges_slices.append(
                    OptimizationParam(
                        param=optim_param['param'],
                        low=optim_param['low'],
                        high=optim_param['high'],
                        n_slices=optim_param['n_slices']
                    )
                )
        # Now extract the ranges from the final ranges
        channel_ranges = []
        slices = []
        for g_params in final_g_ranges_slices:
            channel_ranges.append((g_params['low'], g_params['high']))
            slices.append(g_params["n_slices"])
        #current_intensities = [0.1, 0.2, 0.3] # same as target_cell. comment out if wanting one control spot
        dp = DataProcessor()
        conductance_groups, current_settings = dp.generate_I_g_combinations(channel_ranges, slices, self.sim_params['CI_amps'])

        return conductance_groups, current_settings

    def simulate_train_cells(self, train_cell):
        simulator = Simulator(self.output_folder_name)

        conductance_groups, current_settings = self.get_I_g_combinations()

        for i in range(len(conductance_groups)):
                # Set parameters from the grid
                train_cell.set_g(["gbar_nap", "gkdrbar_kdr", "gbar_na3", "gmbar_im", "glbar_leak"], conductance_groups[i])
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
                        }
                    )
                )
        
        simulator.run(self.cell['modfiles_folder'])

        dp = DataProcessor()
        dp.combine_data(self.output_folder_name + "train")

    def get_rf_prediction(self):
        # Extract Features from TargetCell traces
        dp = DataProcessor()

        dataset_target = np.load(self.output_folder_name + "target/combined_out.npy")
        V_target = dataset_target[:,:,0]
        I_target = dataset_target[:,:,1]

        features_target, columns_target = dp.extract_features(V=V_target,I=I_target,inj_dur=290,inj_start=10)

        # Extract Features from TrainCell traces
        dataset_train = np.load(self.output_folder_name + "train/combined_out.npy")

        g_train = dp.clean_g_bars(dataset_train)
        V_train = dataset_train[:,:,0]
        I_train = dataset_train[:,:,1]

        # TODO: ["i_mean_stdev", "v_spike_stats", "v_mean_potential", "v_amplitude_frequency", "v_arima_coefs"]
        features_train, columns_train = dp.extract_features(V=V_train,I=I_train,inj_dur=self.sim_params['CI_dur'],inj_start=self.sim_params['CI_delay'])


        # Train Model on data
        X_train = features_train
        Y_train = g_train
        
        

        rf = RandomForestOptimizer(random_state=self.optim_params['random_state'])
        rf.fit(X_train, Y_train)

        # Evaluate the model performance

        metrics = Metrics()

        metrics.evaluate_random_forest(rf.model, X_train, Y_train)

        # Predict the conductance values

        X_test = features_target
        prediction = rf.predict(X_test)

        print(f"Predicted Conductances: {prediction}")

        return prediction

        

    def simulate_test_cells(self, test_cell, prediction):
        simulator = Simulator(self.output_folder_name)
        for i in range(len(prediction)):
            for j in range(len(self.sim_params['CI_amps'])):
                # Set parameters from the grid
                test_cell.set_g(["gbar_nap", "gkdrbar_kdr", "gbar_na3", "gmbar_im", "glbar_leak"], prediction[i])
                simulator.submit_job(
                    test_cell, 
                    SimulationParameters(
                        sim_name = "test"+str(i),
                        sim_idx = i * len(prediction) + j,
                        h_v_init = self.sim_params['h_v_init'], # (mV)
                        h_tstop = self.sim_params['h_tstop'],  # (ms)
                        h_dt = self.sim_params['h_dt'], # (ms)
                        h_celsius = self.sim_params['h_celsius'], # (deg C)
                        CI = {
                            "type": "constant",
                            "amp": self.sim_params['CI_amps'][j],
                            "dur": self.sim_params['CI_dur'],
                            "delay": self.sim_params['CI_delay']
                        }
                    )
                )
        simulator.run(self.cell['modfiles_folder'])

        dp = DataProcessor()
        for i in range(len(prediction)):
            dp.combine_data(self.output_folder_name + "test" + str(i))

    def evaluate_fi_curves(self, prediction):

        dp = DataProcessor()

        # Get Target Cell Frequencies
        dataset = np.load(self.output_folder_name + "target/combined_out.npy")

        V_target = dataset[:,:,0]

        target_frequencies = dp.get_fi_curve(V_target, self.sim_params['CI_amps'], inj_dur=self.sim_params['CI_dur']).flatten()

        # Get Test Cell Frequencies
        FI_data = []
        for i in range(len(prediction)):
            dataset = np.load(self.output_folder_name + "test" + str(i) + "/combined_out.npy")
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

        g_final_idx = fi_mae.index(min(fi_mae))

        print(prediction[g_final_idx])
        return g_final_idx, prediction[g_final_idx], FI_data[g_final_idx], target_frequencies

    def create_overlapped_v_plot(self, x, y1, y2, title, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(x, y1, label='Target')
        plt.plot(x, y2, label='Prediction')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(title)
        plt.legend()
        plt.savefig(self.output_folder_name + "results/" + filename)
        plt.close()  # Close the figure to free up memory

    def plot_results(self, g_final_idx, FI_predicted, target_frequencies):
        results_folder = self.output_folder_name + "results/"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # load target traces
        target_traces = np.load(self.output_folder_name + "target/combined_out.npy")
        target_v = target_traces[:,:,0]

        # load final prediction voltage traces
        selected_traces = np.load(self.output_folder_name + "test" + str(g_final_idx) + "/combined_out.npy")
        selected_v = selected_traces[:,:,0]

        time = np.linspace(0, len(target_v[0]), len(target_v[0]))

        # Plot all pairs of traces
        for i in range(len(selected_v)):
            self.create_overlapped_v_plot(time, target_v[i], selected_v[i], f"V Trace Comparison: {self.sim_params['CI_amps'][i]} nA", f"V_trace_{self.sim_params['CI_amps'][i]}nA.png")

        # Plot the FI curves of 

        plt.figure(figsize=(8, 6))
        plt.plot(self.sim_params['CI_amps'], target_frequencies, label='Target FI')
        plt.plot(self.sim_params['CI_amps'], FI_predicted, label='Prediction FI')
        plt.xlabel('Current Injection Intensity (nA)')
        plt.ylabel('Frequency (Hz)')
        plt.title("FI Curve Comparison")
        plt.legend()
        plt.savefig(self.output_folder_name + "results/" + "FI_Curve_Comparison.png")
        plt.close()  # Close the figure to free up memory
