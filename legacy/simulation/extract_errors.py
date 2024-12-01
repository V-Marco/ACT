from pandas import *
import numpy as np
 
# reading CSV file
seed_dir_list = ["42-seed/","43-seed/","44-seed/","45-seed/","46-seed/"]


avg_mse_list = []
min_slice_number = 2
max_slice_number = 7
for i in range(min_slice_number,max_slice_number + 1):
    experiment_dir = f"Burster_orig_{i}-slice/"
    print(f"Experiment - {experiment_dir} MSE:")
    for seed_dir in seed_dir_list:
        output_dir = "output/" + experiment_dir + "model_data/" + seed_dir
        metrics_file = output_dir + "metrics.csv"

        data = read_csv(metrics_file)
        
        # converting column data to list
        amps = data['amp'].tolist()
        mse = data['mse'].tolist()

        avg_mse_list.append(np.mean(mse))

    total_avg_mse = np.mean(avg_mse_list)
    total_std_dev_mse = np.std(avg_mse_list)
    print(f"{total_avg_mse} : {total_std_dev_mse}")