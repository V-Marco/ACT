#!/bin/bash

#SBATCH -W
#SBATCH --qos=normal
#SBATCH --job-name=act_Full_vhalf
#SBATCH --output=./bmtk_sim.out
#SBATCH --time 0-12:00

START=$(date)
python Original.py

#python Full_vhalf.py


END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"