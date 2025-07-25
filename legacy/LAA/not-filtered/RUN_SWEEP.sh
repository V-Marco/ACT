#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -W
#SBATCH --qos=normal
#SBATCH --job-name=act_LAA_Orig-notfilter
#SBATCH --output=./bmtk_sim.out
#SBATCH --time 5-00:00

START=$(date)

python Original.py
#python Original_passmod.py
#python Qualitative_spike_burst.py
#python Qualitative_burst_spike.py
#python Full_vhalf_vcutoff.py


END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"