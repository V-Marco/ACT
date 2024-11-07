#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=256GB
#SBATCH --job-name=PSCs_sim

source /home/shared/L5env/bin/activate
python sim_kmeans_PSCs_new.py
