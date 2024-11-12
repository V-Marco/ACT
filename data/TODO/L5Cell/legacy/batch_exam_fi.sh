#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=exam_sim

source /home/shared/L5env/bin/activate

# Define multiple output folders in an array
OUTPUT_FOLDERS=("output/FI_in_vitro2023-09-08_17-13-23" "output/FI_in_vitro2023-09-08_17-14-55" "output/FI_in_vitro2023-09-08_17-16-27" "output/FI_in_vitro2023-09-08_17-17-59" "output/FI_in_vitro2023-09-08_17-18-06" "output/FI_in_vitro2023-09-08_17-18-02" "output/FI_in_vitro2023-09-08_17-19-38")
# List of all exam_something.py scripts to run
scripts=("exam_fi.py")

# Loop over each output folder
for OUTPUT_FOLDER_PATH in "${OUTPUT_FOLDERS[@]}"
do
    # Loop over each script for the current output folder
    for script in "${scripts[@]}"
    do
        python $script $OUTPUT_FOLDER_PATH
    done
done
