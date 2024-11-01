#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -W
#SBATCH --qos=normal
#SBATCH --job-name=act_all_spiker
#SBATCH --output=./bmtk_sim.out
#SBATCH --time 0-24:00

START=$(date)

python Original_passmod.py
python Original.py
#python Qualitative_spike_burst.py
#python Qualitative_burst_spike.py
#python Full_vhalf_vcutoff.py

END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"