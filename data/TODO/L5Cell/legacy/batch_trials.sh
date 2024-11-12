#!/bin/bash

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=64GB
#SBATCH --job-name=cell_sim

source /home/shared/L5env/bin/activate

JSON_FILE_PATH="constants_to_update.json"

python generate_combinations.py "$JSON_FILE_PATH" | while read -r combination; do
    echo "Updating constants to $combination"
    python update_constants.py "$combination"
    echo "Completed update for $combination"
    python sim_passive.py

done