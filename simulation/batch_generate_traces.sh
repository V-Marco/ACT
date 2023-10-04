#!/bin/bash

#SBATCH -N 1
#SBATCH -n 48
#SBATCH --qos=normal
#SBATCH --job-name=act
#SBATCH --output=act_batch.out
#SBATCH --time 0-12:00

START=$(date)
mpiexec nrniv -mpi -python generate_traces.py
#mpiexec ./components_homogenous/mechanisms/x86_64/special -mpi run_network.py simulation_configECP_base_homogenous.json
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"
