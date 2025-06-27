#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=nodes
#SBATCH --job-name=act-orig-pipeline
#SBATCH --output=output/run_pipeline.out
#SBATCH --time 0-48:00



START=$(date)
python original.py
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"