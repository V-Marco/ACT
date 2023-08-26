# Automatic Cell Tuner (act)

`act` provides tools for optimization-based parameter selection for biologically realistic cell models developed in [NEURON](https://neuron.yale.edu/neuron/). The project is inspired by the [ASCT](https://github.com/pbcanfield/ASCT) library.

`act` relies on a simulation-based optimization, i.e., for a pipeline

Parameters -> Black-box simulator -> Simulated data

it tries to obtain parameter estimates indirectly by working with simulated data.

## Installation

Currently, `act` can be installed from GitHub using pip or locally with the standard `pip` installation process.

```bash
pip install act-neuron
```

```bash
git clone https://github.com/V-Marco/ACT.git
cd ACT
pip install .
```

## Usage

### Prerequisites

Conceptually, `act` requires three components.

1. A `.hoc` file which declares the cell's properties.
2. Modfiles for this `.hoc` file.
3. Target voltage data of shape (num_cur_inj, ...) to predict on OR parameters to simulate target data with.

### Pipeline
`act` operates in original and segregated modes. Original mode runs in the following steps:
1. Generate a parameter set uniformly randomly from a (lower; upper) interval for each current injection.
2. Simulate a voltage trace for each current injection and respective parameter set.
3. Extract key summary features (e.g., inter-spike time), and keep parameter sets for those voltage traces which match the target voltage trace in these summary features.
4. Repeat steps 1-3 until the specified number of current injections is matched.
5. Train a neural network model to predict conductance values from a voltage trace using saved sets as targets.
6. Predict conductance values by applying the trained model to the target voltage data. Take the maximum of each predicted value across all current injections.

Segregated mode changes step 5 so that the model is trained on regions of a voltage trace. The regions can be specified in terms of time (X-axis) or voltage (Y-axis) bounds.

### Setting up a simulation

Simulations' parameters are defined as `python` classes in `simulation/simulation_constants.py`. 
- Names of parameters to optimize for are defined in the `params` property. The names must match the hoc file. Lower and upper bounds are specified in `lows` and `highs` properties.
- Segregated parameters and respective time/voltage bounds are specified as lists-of-lists in the respective `segr_...` properties.

### Running a simulation

`simulation/run_simulation.py` is an example script of running `act` on Pospichil's cells.

`simulation/analyze_res.py` is an example script which gives a summary of the model's quality.


### Examples (Jupyter Notebook)

`examples/Pospischil_sPYr/main.ipynb` example of running `act` on Pospichil's cells

On Google Colab: [Pospischil_sPYr](https://colab.research.google.com/github/V-Marco/ACT/blob/main/examples/Pospischil_sPYr/main.ipynb)