#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -W
#SBATCH --qos=normal
#SBATCH --job-name=act_all_spiker
#SBATCH --output=./bmtk_sim.out
#SBATCH --time 0-12:00

START=$(date)
#python Original.py
#python Qualitative_passive_spike.py
#python Full_vhalf_vcutoff.py
python Full_vhalf_vcutoff_blocked.py


END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"