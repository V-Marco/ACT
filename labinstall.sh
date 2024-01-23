#!/bin/bash
set -x #echo on
# Lab server setup

# set your venv name
ACT_VENV=act-venv
ACT_VENV_LOCATION=${HOME}

# unload all of your loaded modules
# remove all loading of modules in your bashrc preferably
module purge

# deactivate any previously activated environments
deactivate

cd ${ACT_VENV_LOCATION}

# create the new virtual environment
python3.9 -m venv ${ACT_VENV}

# add the load module to your environment activation
sed -i '3s/^/module purge\n/' ${ACT_VENV_LOCATION}/${ACT_VENV}/bin/activate
sed -i '4s/^/module load mpich-x86_64-nopy\n/' ${ACT_VENV_LOCATION}/${ACT_VENV}/bin/activate

# OPTIONAL! ADD THIS TO YOUR .BASHRC - Uncomment
# echo "source ${ACT_VENV_LOCATION}/${ACT_VENV}/bin/activate" >> ${HOME}/.bashrc
# or just run `source act-venv/bin/activate` everytime you login

# activate the environment without logging out 
source ${ACT_VENV_LOCATION}/${ACT_VENV}/bin/activate

# clone the repo
git clone https://github.com/V-Marco/ACT

# install dependancies
cd ACT
pip install -e .
pip install mpi4py bmtk
