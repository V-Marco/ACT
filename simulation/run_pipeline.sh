#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos=normal
#SBATCH --job-name=act-orig-pipeline
#SBATCH --output=output/run_pipeline.out
#SBATCH --time 0-12:00



START=$(date)
python run_full_pipeline.py
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"