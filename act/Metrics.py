from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from act.DataProcessor import DataProcessor

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
    
    def evaluate_random_forest(self, reg, X_train, Y_train):
        print("Evaluating random forest")
        # evaluate the model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
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

    def get_fi_curve(trace, amps, ignore_negative=True, inj_dur=1000):
        """
        Returns the spike counts per amp.
        inj_dur is the duration of the current injection
        """
        dp = DataProcessor()
        spikes, interspike_times = dp.get_spike_stats(trace)

        if ignore_negative:
            non_neg_idx = (amps >= 0).nonzero().flatten()
            amps = amps[amps >= 0]
            spikes = spikes[non_neg_idx]

        spikes = (1000.0 / inj_dur) * spikes  # Convert to Hz

        return spikes
    
    def final_prediction(self, reg, X_train, Y_train, X_Test):
        pass