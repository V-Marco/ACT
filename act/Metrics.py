import torch
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

class Metrics:
    def __init__(self):
        pass

    def correlation_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
        cov = (target_data - torch.mean(target_data, dim=0, keepdim=True)) * (
            simulated_data - torch.mean(simulated_data, dim=0, keepdim=True)
        )
        cov = torch.sum(cov, dim=0)

        var0 = torch.sum(
            torch.square(target_data - torch.mean(target_data, dim=0, keepdim=True)), dim=0
        )
        var1 = torch.sum(
            torch.square(simulated_data - torch.mean(simulated_data, dim=0, keepdim=True)),
            dim=0,
        )
        corr = cov / (torch.sqrt(var0 * var1) + 1e-15)

        return float(torch.mean(corr).cpu().detach())

    def torch_correlation_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
        matrix = torch.cat((target_data,simulated_data),0)
        corr_mat = torch.corrcoef(matrix)
        corr_coef = corr_mat[0,1]

        return float(corr_coef)

    def mse_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
        return float(
            torch.mean(torch.mean(torch.square(target_data - simulated_data), dim=0))
            .cpu()
            .detach()
        )

    def mae_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
        return float(
            torch.mean(torch.mean(torch.abs(target_data - simulated_data), dim=0))
            .cpu()
            .detach()
        )
    
    def evaluate_random_forest(reg, X_train, Y_train):
        print("Evaluating random forest")
        # evaluate the model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(
            reg,
            X_train,
            Y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        # report performance
        print("MAE: %.6f (%.6f)" % (mean(n_scores), std(n_scores)))