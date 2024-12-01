#!/bin/bash

#SBATCH -N 1
#SBATCH -n 48
#SBATCH -W
#SBATCH --qos=normal
#SBATCH --job-name=act
#SBATCH --output=output/bmtk_sim.out
#SBATCH --time 0-12:00

START=$(date)
mpiexec nrniv -mpi -python generate_traces.py --sweep
#mpiexec ./components_homogenous/mechanisms/x86_64/special -mpi run_network.py simulation_configECP_base_homogenous.json
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"