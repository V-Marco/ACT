#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=exam_PSCs

source /home/shared/L5env/bin/activate

# Define multiple output folders in an array
OUTPUT_FOLDERS=(
'output/PSC_2023-08-23_21-56-54'
)

# List of all exam_something.py scripts to run
scripts=("exam_PSCs.py")

# Loop over each output folder
for OUTPUT_FOLDER_PATH in "${OUTPUT_FOLDERS[@]}"
do
    # Loop over each script for the current output folder
    for script in "${scripts[@]}"
    do
        python $script $OUTPUT_FOLDER_PATH
    done
done
