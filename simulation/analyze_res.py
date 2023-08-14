import numpy as np
import pandas as pd
import os

output_folders = [
    "output_Pospischil_sPY/original_original",
    "output_Pospischil_sPY/original_segregated"
]

def main():

    for output_folder in output_folders:
        metrics = pd.read_csv(os.path.join(output_folder, "metrics.csv"))
        print(output_folder, ":")
        print(f"Med MSE: {metrics['mse'].median()} ({metrics['mse'].std()})")
        print(f"Med Corr: {metrics['corr'].median()} ({metrics['corr'].std()})")
        print("----------\n")

if __name__ == "__main__":
    main()