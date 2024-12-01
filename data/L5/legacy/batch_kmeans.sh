#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=cell_sim

source netpyne_env/bin/activate
python sim_kmeans.py
