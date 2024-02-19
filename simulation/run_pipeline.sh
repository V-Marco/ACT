#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos=normal
#SBATCH --job-name=act-pipeline
#SBATCH --output=output/act_batch_full.out
#SBATCH --time 0-12:00



START=$(date)
python generate_target_traces.py
python generate_traces.py build
sbatch --wait batch_generate_traces.sh
python generate_arma_stats.py
python run_simulation.py
python analyze_res.py
python plot_fi.py
python plot_learned_parameters.py
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"