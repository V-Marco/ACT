import json
import numpy as np
from datetime import timedelta
from dataclasses import fields

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

from act.data_processing import *
from act.act_types import GettablePassiveProperties

def pp_error(pp_target: GettablePassiveProperties, pp_pred: GettablePassiveProperties) -> list:
    '''
    Compute absolute error between target and predicted passive properties.

    Parameters:
    ----------
    pp_target: GettablePasssiveProperties
        Target passive properties.

    pp_pred: GettablePasssiveProperties
        Predicted passive properties.

    Returns:
    ----------
    error: list[(property, abs_error)]
        Absolute error for each property.
    '''
    error = []
    for field in fields(pp_target):
        error.append((field.name, np.abs(getattr(pp_target, field.name) - getattr(pp_pred, field.name))))
    return error


def correlation_score(target_data, simulated_data) -> float:
    '''
    Uses np.corrcoef to calculate a correlation coefficient between target and simulated data
    
    Parameters:
    ----------
    target_data: list[float]
        Generic target data
    
    simulated_data: list[float]
        Generic simulation data

    Returns:
    ----------
    corr_coef: float
        correlation coefficient between target and simulated data
    '''
    matrix = np.concatenate((target_data,simulated_data),0)
    corr_mat = np.corrcoef(matrix)
    corr_coef = float(corr_mat[0,1])

    return corr_coef


def evaluate_random_forest(estimator, X_train, Y_train, random_state=42, n_repeats=3, n_splits=10, save_file=None) -> tuple:
    '''
    Uses RepeatedKFold to calculate the absolute value of the cross_validation_score over the RF model.
    
    Parameters:
    ----------
    estimator: BaseEstimator
        Estimator

    X_train: list[float]
        Generic target data
    
    Y_train: list[float]
        Generic simulation data
    
    random_state: int, default = 42
        Random seed
        
    n_repeats: int, default = 3
        Number of repeats
    
    n_splits: int, default = 10
        Number of splits
    
    save_file: str, default = None
        Save File Path

    Returns:
    ----------
    mean_score: float
        Mean model train MAE
    
    std_score: flost
        Standard Deviation model train MAE
    '''
    print("Evaluating random forest")
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    n_scores = abs(cross_val_score(
        estimator,
        X_train,
        Y_train,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    ))
    print("Model Training MAE: %.6f (%.6f)" % (np.mean(n_scores), np.std(n_scores)))
    
    if not save_file == None:
        print(f"Saving training mean/stdev scores to {save_file}")
        save_to_json(np.mean(n_scores), "model_train_mean_mae", save_file)
        save_to_json(np.std(n_scores), "model_train_stdev_mae", save_file)
    
    return np.mean(n_scores), np.std(n_scores)


def save_interspike_interval_comparison(module_foldername: str, prediction_data_filepath: str, CI_settings: list, current_inj_combos: list, dt, threshold = -20, first_n_spikes=5,save_file=None) -> tuple:
    '''
    Gets the MAE of the interspike interval (list of 20 spikes - padded with large number)
    between the Target and Prediction. Saves this to the metrics .json file
    Parameters:
    ----------
    module_foldername: str
        Module foldername path
    
    prediction_data_filepath: str
        prediction data (combined_out.npy) filepath
        
    current_inj_combos: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings in simulation parameters (original settings)
        
    current_inj_combos: list[ConstantCurrentInjection | RampCurrentInjection | GaussianCurrentInjection], default = None
        A list of Current injection settings for all trials
    
    dt: float
        Timestep
    
    first_n_spikes: int, default = 5
        First number of spikes considered
    
    save_file: str, default = None
        Save Filepath
    
    
    Returns:
    ----------
    mean_isi: float
        Mean interspike interval
    
    stdev_isi: float
        Standard deviation interspike interval
    '''
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    target_V = target_dataset[:,:,0]
    target_I = target_dataset[:,:,1]
    target_lto_hto = pred_dataset[:,1,3]
    
    pred_dataset = np.load(prediction_data_filepath)
    pred_V = pred_dataset[:,:,0]
    pred_I = pred_dataset[:,:,1]
    pred_lto_hto = pred_dataset[:,1,3]
    
    isi_maes = []

    target_df = get_summary_features(V=target_V,I=target_I, lto_hto=target_lto_hto, current_inj_combos=current_inj_combos, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
    pred_df = get_summary_features(V=pred_V,I=pred_I, lto_hto=pred_lto_hto, current_inj_combos=current_inj_combos, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
    
    # Getting interspike interval data from extracted features
    isi_target = target_df["isi"]
    isi_prediction = pred_df["isi"]
    
    print(f"Interspike times (Target): {isi_target}")
    print(f"Interspike times (Prediction): {isi_prediction}")
        
    for i in range(len(CI_settings)):
        isi_maes.append(mean_absolute_error(isi_target[i], isi_prediction[i]))
        
    print(f"MAE for each I injection: {isi_maes}")
        
    mean_isi = np.mean(isi_maes)
    stdev_isi = np.std(isi_maes)
    
    print(f"Mean interspike-interval MAE: {mean_isi}")
    print(f"Standard Deviation interspike-interval MAE: {stdev_isi}")
    
    if not save_file == None:
        save_to_json(first_n_spikes, "num_spikes_in_isi_calc", save_file)
        save_to_json(mean_isi, "mean_interspike_interval_mae", save_file)
        save_to_json(stdev_isi, "stdev_interspike_interval_mae", save_file)
        
    return mean_isi, stdev_isi


def save_prediction_g_mae(actual_g, save_file) -> float:
    '''
    Gets the MAE between Target and Prediction maximal conductance values. Saves to .json
    
    Parameters:
    ----------
    actual_g: list[float]
    
    
    save_file: str
        Save file path
    
    Returns:
    ----------
    mae: float
        Final prediction conductance MAE (with target)
    '''
    with open(save_file, 'r') as file:
        data = json.load(file)
    
    final_g_prediction = data.get('final_g_prediction', None)
    predicted_g = list(final_g_prediction.values())
    
    actual_g = list(actual_g.values())
    
    mae = mean_absolute_error(np.array(actual_g),np.array(predicted_g))
    
    print(f"MAE of final g prediction: {mae}")
    save_to_json(mae, "mae_final_predicted_g", save_file)
    return mae
    

def save_feature_mae(module_foldername, prediction_data_filepath, train_features, dt, threshold=0, first_n_spikes=5, CI_settings=None, save_file=None) -> float:
    '''
    Gets the MAE between Target and Prediction features (pre-selected by the user).
    Saves to .json
    
    Parameters:
    ----------
    module_foldername: str
        Module foldername
    
    prediction_data_filepath: str
        Prediction data File path
        
    train_features: list[str]
        List of features being used
        
    dt: float
        Timestep
        
    threshold: float, default = 0
        Spike threshold
        
    first_n_spikes: int, default = 5
        Number of first spikes considered
        
    CI_settings: list[ConstantCurrentInjection| RampCurrentInjection| GaussianCurrentInjection], default = None
        Current Injection settings
    
    save_file: str, default = None
        Save file path
    
    Returns:
    ----------
    feature_mae: float
        MAE for predicted features(after simulating predicted g) and target
    '''
    target_dataset = np.load(f"{module_foldername}/target/combined_out.npy")
    
    target_V = target_dataset[:,:,0]
    target_I = target_dataset[:,:,1]
    target_lto_hto = target_dataset[:,1,2]
    
    pred_dataset = np.load(prediction_data_filepath)
    
    pred_V = pred_dataset[:,:,0]
    pred_I = pred_dataset[:,:,1]
    pred_lto_hto = pred_dataset[:,1,3]
        
    target_features = get_summary_features(V=target_V, I=target_I, lto_hto=target_lto_hto, current_inj_combos=CI_settings, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
    sub_target_features = select_features(target_features, train_features)
    
    pred_features = get_summary_features(V=pred_V,I=pred_I, lto_hto=pred_lto_hto, current_inj_combos=CI_settings, spike_threshold=threshold, max_n_spikes=first_n_spikes, dt=dt)
    sub_pred_features = select_features(pred_features, train_features)
    
    feature_mae = mean_absolute_error(sub_target_features.to_numpy(), sub_pred_features.to_numpy)
    print(f"MAE of summary features for final prediction: {feature_mae}")
    
    save_to_json(feature_mae, "summary_stats_mae_final_prediction", save_file)
    
    return feature_mae


def average_metrics_across_seeds(metric_filename_list, save_filename) -> dict:
    '''
    Takes in a list of metrics files (.json files from different random seeds) and
    averages the scores across those runs. Saves the averaged metrics to .json.
    
    Parameters:
    ----------
    metric_filename_list: list[str]
        List of metric filename paths
        
    save_filename: str
        Save filename path
    
    Returns:
    ----------
    data_to_save: dict
        Averaged metrics across a filename list of saved metrics files
    '''
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
    ) = load_metric_data(metric_filename_list)

    if not rf_mean_g_score_mae_list==None: 
        avg_rf_mean_g_score_mae = np.mean(rf_mean_g_score_mae_list)
        stdev_rf_mean_g_score_mae = np.std(rf_mean_g_score_mae_list)
    else:
        avg_rf_mean_g_score_mae = None
        stdev_rf_mean_g_score_mae = None
    
    if not rf_stdev_g_score_mae_list==None: 
        avg_rf_stdev_g_score_mae = np.mean(rf_stdev_g_score_mae_list)
        stdev_rf_stdev_g_score_mae = np.std(rf_stdev_g_score_mae_list)
    else:
        avg_rf_stdev_g_score_mae = None
        stdev_rf_stdev_g_score_mae = None
    
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
    return data_to_save