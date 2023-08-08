# Automatic Cell Tuner (act)

`act` provides tools for optimization-based parameter selection for biologically realistic cell models developed in [NEURON](https://neuron.yale.edu/neuron/). The project is inspired by the [ASCT](https://github.com/pbcanfield/ASCT) library.

`act` relies on a simulation-based optimization, i.e., for a pipeline

Parameters -> Black-box simulator -> Simulated data

it tries to obtain parameter estimates indirectly by working with simulated data.

## Installation

Currently, `act` can be installed from GitHub using standard `pip` installation process.

```bash
git clone https://github.com/V-Marco/ACT.git
cd ACT
pip install .
```

## Basic Usage

Conceptually, `act` requires three components.

1. A `.hoc` file which declares the cell's properties.
2. Modfiles for this `.hoc` file.
3. Target voltage data to predict on.

Additionally, simulation parameters need to be set in the `constants.py` file.


