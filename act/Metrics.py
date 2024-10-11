from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from act.DataProcessor import DataProcessor
from datetime import datetime, timedelta
import json

class Metrics:
    def __init__(self):
        pass

    def correlation_score(self, target_data, simulated_data) -> float:
        cov = (target_data - np.mean(target_data, dim=0, keepdim=True)) * (
            simulated_data - np.mean(simulated_data, dim=0, keepdim=True)
        )
        cov = np.sum(cov, dim=0)

        var0 = np.sum(
            np.square(target_data - np.mean(target_data, dim=0, keepdim=True)), dim=0
        )
        var1 = np.sum(
            np.square(simulated_data - np.mean(simulated_data, dim=0, keepdim=True)),
            dim=0,
        )
        corr = cov / (np.sqrt(var0 * var1) + 1e-15)

        return float(np.mean(corr))

    def np_correlation_score(self, target_data, simulated_data) -> float:
        matrix = np.concatenate((target_data,simulated_data),0)
        corr_mat = np.corrcoef(matrix)
        corr_coef = corr_mat[0,1]

        return float(corr_coef)

    def mse_score(self, target_data, simulated_data) -> float:
        return float(
            np.mean(np.square(target_data - simulated_data))
        )

    def mae_score(self, target_data, simulated_data) -> float:
        return float(
            np.mean(np.abs(target_data - simulated_data))
        )
    
    def evaluate_random_forest(self, reg, X_train, Y_train, random_state=42, n_repeats=3, n_splits=10, save_file=None):
        print("Evaluating random forest")
        # evaluate the model
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        n_scores = abs(cross_val_score(
            reg,
            X_train,
            Y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        ))
        # report performance
        print("MAE: %.6f (%.6f)" % (np.mean(n_scores), np.std(n_scores)))
        
        dp = DataProcessor()
        if not save_file == None:
            print(f"Saving rf mean/stdev scores to {save_file}")
            dp.save_to_json(np.mean(n_scores), "rf_mean_g_score_mae", save_file)
            dp.save_to_json(np.std(n_scores), "rf_stdev_g_score_mae", save_file)
    
    
    def save_interspike_interval_comparison(self, module_foldername, prediction_data_filepath, amps, dt, first_n_spikes=5,save_file=None):

        dp = DataProcessor()
        
        # Load target data
        target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
        
        target_V = target_dataset[:,:,0]
        
        # Load prediction data
        pred_dataset = np.load(prediction_data_filepath)
        
        pred_V = pred_dataset[:,:,0]
        
        isi_maes = []
        
        _,_,isi_target,*_ = dp.extract_v_traces_features(target_V, num_spikes=first_n_spikes, dt=dt)
        
        _,_,isi_prediction,*_ = dp.extract_v_traces_features(pred_V, num_spikes=first_n_spikes, dt=dt)
        
        print(f"Interspike times (Target): {isi_target}")
        
        print(f"Interspike times (Prediction): {isi_prediction}")
            
        for i in range(len(amps)):
            # Get mae between the isi (target/pred) for first n spikes
            isi_maes.append(self.mae_score(isi_target[i], isi_prediction[i]))
            
        print(f"MAE for each I injection: {isi_maes}")
            
        # Now get the mean/stdev mae for the 3 I injection intensities
        mean_isi = np.mean(isi_maes)
        stdev_isi = np.std(isi_maes)
        
        print(f"Mean interspike-interval MAE: {mean_isi}")
        print(f"Standard Deviation interspike-interval MAE: {stdev_isi}")
        
        if not save_file == None:
            dp = DataProcessor()
            dp.save_to_json(first_n_spikes, "num_spikes_in_isi_calc", save_file)
            dp.save_to_json(mean_isi, "mean_interspike_interval_mae", save_file)
            dp.save_to_json(stdev_isi, "stdev_interspike_interval_mae", save_file)
            
        return mean_isi, stdev_isi
    
    def save_prediction_g_mae(self, actual_g, save_file):
        with open(save_file, 'r') as file:
            data = json.load(file)
        
        final_g_prediction = data.get('final_g_prediction', None)
        predicted_g = list(final_g_prediction.values())
        
        actual_g = list(actual_g.values())
        
        mae = self.mae_score(np.array(actual_g),np.array(predicted_g))
        
        print(f"MAE of final g prediction: {mae}")
        
        dp = DataProcessor()
        dp.save_to_json(mae, "mae_final_predicted_g", save_file)
        
    def save_feature_mae(self, module_foldername, prediction_data_filepath, train_features, dt, threshold=0, first_n_spikes=5, save_file=None):
        dp = DataProcessor()
        
        # Load target data
        target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
        
        target_V = target_dataset[:,:,0]
        target_I = target_dataset[:,:,1]
        
        # Load prediction data
        pred_dataset = np.load(prediction_data_filepath)
        
        pred_V = pred_dataset[:,:,0]
        pred_I = pred_dataset[:,:,1]
        
        target_V_features, _ = dp.extract_features(train_features=train_features, V=target_V,I=target_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
        
        pred_V_features, _ = dp.extract_features(train_features=train_features, V=pred_V,I=pred_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
        
        # Calculate the MAE between the features
        feature_mae = self.mae_score(target_V_features, pred_V_features)
        print(f"MAE of summary features for final prediction: {feature_mae}")
        
        dp.save_to_json(feature_mae, "summary_stats_mae_final_prediction", save_file)
        

    
    def average_metrics_across_seeds(self, metric_filename_list, save_filename):
        
        dp = DataProcessor()
        
        # Load in as much data as possible from the metrics files
        (
            num_spikes_in_isi_calc_list, 
            mean_interspike_interval_mae_list, 
            stdev_interspike_interval_mae_list, 
            final_g_prediction_list, 
            rf_mean_g_score_mae_list,
            rf_stdev_g_score_mae_list,
            mae_final_predicted_g_list,
            prediction_evaluation_method_list, 
            final_prediction_fi_mae_list, 
            final_prediction_voltage_mae_list, 
            feature_mae_list,
            module_runtime_list
        ) = dp.load_metric_data(metric_filename_list)
        
        # Average/STDEV of the MAE of cross validation of kfold
        avg_rf_mean_g_score_mae = np.mean(rf_mean_g_score_mae_list)
        stdev_rf_mean_g_score_mae = np.std(rf_mean_g_score_mae_list)
        
        avg_rf_stdev_g_score_mae = np.mean(rf_stdev_g_score_mae_list)
        stdev_rf_stdev_g_score_mae = np.std(rf_stdev_g_score_mae_list)
        
        # Average/STDEV of the MAE of the final predicted g vs real g
        avg_final_g_predictions_mae = np.mean(mae_final_predicted_g_list)
        stdev_final_g_predictions_mae = np.std(mae_final_predicted_g_list)
        
        # Average/STDEV of the MAE of the final predicted voltage traces
        avg_voltage_mae = np.mean(final_prediction_voltage_mae_list)
        stdev_voltage_mae = np.std(final_prediction_voltage_mae_list)
        
        # Average/STDEV of the MAE of the final predicted FI values
        avg_fi_mae = np.mean(final_prediction_fi_mae_list)
        stdev_fi_mae = np.std(final_prediction_fi_mae_list)
        
        # Average/STDEV of the MAE of the final predicted interspike intervals
        avg_mean_isi_mae = np.mean(mean_interspike_interval_mae_list)
        stdev_mean_isi_mae = np.std(mean_interspike_interval_mae_list)
        
        avg_stdev_isi_mae = np.mean(stdev_interspike_interval_mae_list)
        stdev_stdev_isi_mae = np.std(stdev_interspike_interval_mae_list)
        
        # Average/STDEV of the MAE of the summary stat features
        avg_feature_mae = np.mean(feature_mae_list)
        stdev_feature_mae = np.std(feature_mae_list)
        
        # Average/STDEV of the Module Runtime
        avg_runtime_s = np.mean(module_runtime_list)
        stdev_runtime_s = np.std(module_runtime_list)
        
        avg_time_obj = f"{timedelta(seconds=avg_runtime_s)}"
        stdev_time_obj = f"{timedelta(seconds=stdev_runtime_s)}"
        
        # Write all of these values to an output json file
        data_to_save = {
            "final_g_predictions": final_g_prediction_list,
            "avg_final_g_predictions_mae": avg_final_g_predictions_mae,
            "stdev_final_g_predictions_mae": stdev_final_g_predictions_mae,
            "avg_rf_mean_g_score_mae": avg_rf_mean_g_score_mae,
            "stdev_rf_mean_g_score_mae": stdev_rf_mean_g_score_mae,
            "avg_rf_stdev_g_score_mae": avg_rf_stdev_g_score_mae,
            "stdev_rf_stdev_g_score_mae": stdev_rf_stdev_g_score_mae,
            "prediction_evaluation_methods": prediction_evaluation_method_list,
            "avg_voltage_mae": avg_voltage_mae,
            "stdev_voltage_mae": stdev_voltage_mae,
            "avg_fi_mae ": avg_fi_mae ,
            "stdev_fi_mae": stdev_fi_mae,
            "num_spikes_in_isi_calcs": num_spikes_in_isi_calc_list,
            "avg_mean_isi_mae": avg_mean_isi_mae,
            "stdev_mean_isi_mae": stdev_mean_isi_mae,
            "avg_stdev_isi_mae": avg_stdev_isi_mae,
            "stdev_stdev_isi_mae": stdev_stdev_isi_mae,
            "avg_feature_mae": avg_feature_mae,
            "stdev_feature_mae": stdev_feature_mae,
            "avg_module_runtime": avg_time_obj,
            "stdev__module_runtime": stdev_time_obj
        }
        
        with open(save_filename, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        
        print(f"Saved metrics to: {save_filename}")