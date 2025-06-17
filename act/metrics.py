import numpy as np
from dataclasses import fields

from act.types import GettablePassiveProperties

def pp_error(pp_target: GettablePassiveProperties, pp_pred: GettablePassiveProperties) -> list:
    """
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
    """
    error = []
    for field in fields(pp_target):
        error.append((field.name, np.abs(getattr(pp_target, field.name) - getattr(pp_pred, field.name))))
    return error

def summary_features_error(sf_target: np.ndarray, sf_pred: np.ndarray) -> float:
    """
    Compute mean absolute error between target and predicted summary features.

    Parameters:
    ----------
    sf_target: np.ndarray of shape (n_trials, n_features)
        Target summary features.
    
    sf_pred: np.ndarray of shape (n_trials, n_features)
        Predicted summary features.
    
    Returns:
    ----------
    mae: np.ndarray of shape (n_trials)
        Mean absolute error across z-transformed features.
    """

    if len(sf_target) > 1:
        # Z-transform the predicted features
        z_mean = np.nanmean(sf_pred, axis = 0)
        z_std = np.nanstd(sf_pred, axis = 0)
        sf_pred = (sf_pred - z_mean) / z_std

        # Use sample mean and std for target transformation
        sf_target = (sf_target - z_mean) / z_std

    mae = np.nanmean(np.abs(sf_target - sf_pred), axis = 1)
    return mae