import os

import numpy as np
import pandas as pd

output_folders = [
    "output_Pospischil_sPYr/original_original",
    "output_Pospischil_sPYr/segregated_segregated",
]

target_params = np.array([1e-5, 0.05, 0.005, 3e-5, 0.001])


def main():
    for output_folder in output_folders:
        metrics = pd.read_csv(os.path.join(output_folder, "metrics.csv"))
        preds = np.array(
            pd.read_csv(os.path.join(output_folder, "pred.csv"), index_col=0)
        )
        print(output_folder, ":")
        print(f"Med MSE: {metrics['mse'].median()} ({metrics['mse'].std()})")
        print(f"Med Corr: {metrics['corr'].median()} ({metrics['corr'].std()})")
        print(f"Pred MAE: {np.mean(np.abs(target_params - preds))}")
        print("----------\n")


if __name__ == "__main__":
    main()
