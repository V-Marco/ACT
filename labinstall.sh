#!/bin/bash
set -x #echo on
# Lab server setup

# unload all of your loaded modules
# remove all loading of modules in your bashrc preferably
module purge

# deactivate any previously activated environments
deactivate

# create a new fresh environment in your home directory 
cd ~

# create the new virtual environment
python3.9 -m venv act-venv

# add the load module to your environment activation
sed -i '3s/^/module purge\n/'
sed -i '4s/^/module load mpich-x86_64-nopy\n/'

# OPTIONAL! ADD THIS TO YOUR .BASHRC - Uncomment
# echo "source ${HOME}/act-venv/bin/activate" >> ${HOME}/.bashrc
# or just run `source ~/act-venv/bin/activate` everytime you login

# activate the environment without logging out 
source ${HOME}/act-venv/bin/activate

# clone the repo
git clone https://github.com/V-Marco/ACT

# install dependancies
cd ACT
pip install -e .
pip install mpi4py bmtk
