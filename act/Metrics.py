import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from act.DataProcessor import DataProcessor

'''
A class that holds a collection of methods for calculating metrics to evaluate the quality of
the Automatic Cell Tuning process.
'''

class Metrics:
    def __init__(self):
        pass
    
    '''
    np_correlation_score
    Uses np.corrcoef to calculate a correlation coefficient between target and simulated data
    '''

    def np_correlation_score(self, target_data, simulated_data) -> float:
        matrix = np.concatenate((target_data,simulated_data),0)
        corr_mat = np.corrcoef(matrix)
        corr_coef = corr_mat[0,1]

        return float(corr_coef)

    '''
    mse_score
    Calculates Mean Squared Error between Target and Predicted
    '''
    def mse_score(self, target_data, simulated_data) -> float:
        return float(
            np.mean(np.square(target_data - simulated_data))
        )

    '''
    mae_score
    Calculates Mean Absolute Error between Target and Predicted
    '''
    def mae_score(self, target_data, simulated_data) -> float:
        return float(
            np.mean(np.abs(target_data - simulated_data))
        )
    
    '''
    evaluate_random_forest
    Uses RepeatedKFold to calculate the absolute value of the cross_validation_score over the RF model.
    '''
    
    def evaluate_random_forest(self, reg, X_train, Y_train, random_state=42, n_repeats=3, n_splits=10, save_file=None):
        print("Evaluating random forest")
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
        print("MAE: %.6f (%.6f)" % (np.mean(n_scores), np.std(n_scores)))
        
        dp = DataProcessor()
        if not save_file == None:
            print(f"Saving rf mean/stdev scores to {save_file}")
            dp.save_to_json(np.mean(n_scores), "rf_mean_g_score_mae", save_file)
            dp.save_to_json(np.std(n_scores), "rf_stdev_g_score_mae", save_file)
    
    '''
    save_interspike_interval_comparison
    Gets the MAE of the interspike interval (list of 20 spikes - padded with large number)
    between the Target and Prediction. Saves this to the metrics .json file
    '''
    
    def save_interspike_interval_comparison(self, module_foldername, prediction_data_filepath, amps, dt, first_n_spikes=5,save_file=None):
        dp = DataProcessor()
        
        target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
        target_V = target_dataset[:,:,0]
        
        pred_dataset = np.load(prediction_data_filepath)
        pred_V = pred_dataset[:,:,0]
        
        isi_maes = []
        _,_,isi_target,*_ = dp.extract_v_traces_features(target_V, num_spikes=first_n_spikes, dt=dt)
        _,_,isi_prediction,*_ = dp.extract_v_traces_features(pred_V, num_spikes=first_n_spikes, dt=dt)
        
        print(f"Interspike times (Target): {isi_target}")
        print(f"Interspike times (Prediction): {isi_prediction}")
            
        for i in range(len(amps)):
            isi_maes.append(self.mae_score(isi_target[i], isi_prediction[i]))
            
        print(f"MAE for each I injection: {isi_maes}")
            
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
    
    '''
    save_prediction_g_mae
    Gets the MAE between Target and Prediction maximal conductance values. Saves to .json
    '''
    
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
        
    '''
    save_feature_mae
    Gets the MAE between Target and Prediction features (pre-selected by the user).
    Saves to .json
    '''
        
    def save_feature_mae(self, module_foldername, prediction_data_filepath, train_features, dt, threshold=0, first_n_spikes=5, save_file=None):
        dp = DataProcessor()
        
        target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
        
        target_V = target_dataset[:,:,0]
        target_I = target_dataset[:,:,1]
        
        pred_dataset = np.load(prediction_data_filepath)
        
        pred_V = pred_dataset[:,:,0]
        pred_I = pred_dataset[:,:,1]
        
        target_V_features, _ = dp.extract_features(train_features=train_features, V=target_V,I=target_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
        
        pred_V_features, _ = dp.extract_features(train_features=train_features, V=pred_V,I=pred_I, threshold=threshold, num_spikes=first_n_spikes, dt=dt)
        
        feature_mae = self.mae_score(target_V_features, pred_V_features)
        print(f"MAE of summary features for final prediction: {feature_mae}")
        
        dp.save_to_json(feature_mae, "summary_stats_mae_final_prediction", save_file)
    
    '''
    average_metrics_across_seeds
    Takes in a list of metrics files (.json files from different random seeds) and
    averages the scores across those runs. Saves the averaged metrics to .json.
    '''

    def average_metrics_across_seeds(self, metric_filename_list, save_filename):
        dp = DataProcessor()
        
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

        avg_rf_mean_g_score_mae = np.mean(rf_mean_g_score_mae_list)
        stdev_rf_mean_g_score_mae = np.std(rf_mean_g_score_mae_list)
        
        avg_rf_stdev_g_score_mae = np.mean(rf_stdev_g_score_mae_list)
        stdev_rf_stdev_g_score_mae = np.std(rf_stdev_g_score_mae_list)
        
        avg_final_g_predictions_mae = np.mean(mae_final_predicted_g_list)
        stdev_final_g_predictions_mae = np.std(mae_final_predicted_g_list)
        
        avg_voltage_mae = np.mean(final_prediction_voltage_mae_list)
        stdev_voltage_mae = np.std(final_prediction_voltage_mae_list)
        
        avg_fi_mae = np.mean(final_prediction_fi_mae_list)
        stdev_fi_mae = np.std(final_prediction_fi_mae_list)
        
        avg_mean_isi_mae = np.mean(mean_interspike_interval_mae_list)
        stdev_mean_isi_mae = np.std(mean_interspike_interval_mae_list)
        avg_stdev_isi_mae = np.mean(stdev_interspike_interval_mae_list)
        stdev_stdev_isi_mae = np.std(stdev_interspike_interval_mae_list)
        
        avg_feature_mae = np.mean(feature_mae_list)
        stdev_feature_mae = np.std(feature_mae_list)
        
        avg_runtime_s = np.mean(module_runtime_list)
        stdev_runtime_s = np.std(module_runtime_list)
        avg_time_obj = f"{timedelta(seconds=avg_runtime_s)}"
        stdev_time_obj = f"{timedelta(seconds=stdev_runtime_s)}"
        
        data_to_save = {
            "avg_final_g_predictions_mae": avg_final_g_predictions_mae,
            "stdev_final_g_predictions_mae": stdev_final_g_predictions_mae,
            "avg_voltage_mae": avg_voltage_mae,
            "stdev_voltage_mae": stdev_voltage_mae,
            "avg_fi_mae ": avg_fi_mae ,
            "stdev_fi_mae": stdev_fi_mae,
            "avg_mean_isi_mae": avg_mean_isi_mae,
            "stdev_mean_isi_mae": stdev_mean_isi_mae,
            "avg_stdev_isi_mae": avg_stdev_isi_mae,
            "stdev_stdev_isi_mae": stdev_stdev_isi_mae,
            "avg_feature_mae": avg_feature_mae,
            "stdev_feature_mae": stdev_feature_mae,
            "avg_module_runtime": avg_time_obj,
            "stdev__module_runtime": stdev_time_obj,
            "avg_rf_mean_g_score_mae": avg_rf_mean_g_score_mae,
            "stdev_rf_mean_g_score_mae": stdev_rf_mean_g_score_mae,
            "avg_rf_stdev_g_score_mae": avg_rf_stdev_g_score_mae,
            "stdev_rf_stdev_g_score_mae": stdev_rf_stdev_g_score_mae,
            "final_g_predictions": final_g_prediction_list,
            "prediction_evaluation_methods": prediction_evaluation_method_list,
            "num_spikes_in_isi_calcs": num_spikes_in_isi_calc_list,
        }
        
        with open(save_filename, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        
        print(f"Saved metrics to: {save_filename}")