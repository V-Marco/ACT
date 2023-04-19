# Automatic Cell Tuner (act)

`act` provides tools for optimization-based parameter selection for biologically realistic cell models developed in [NEURON](https://neuron.yale.edu/neuron/). The project is based on the [ASCT](https://github.com/pbcanfield/ASCT) library.

`act` relies on a simulation-based optimization, i.e., for a forward pipeline

Parameters -> Black-box simulator -> Simulated data

it tries to obtain parameter estimates indirectly by working with simulated data.

## Installation

For compatibility reasons, `act` runs on Python 3.8 and uses outdated package versions. So, it is recommended to create a new virtual environment to run `act` in. In the example below, the environment is called `act_test`.

```bash
conda create --name act_test
```

Currently, `act` can be installed from GitHub using standard pip installation process.

```bash
git clone https://github.com/V-Marco/ACT.git
cd ACT
pip install .
```

Although not required, you will probably also want to install `jupyter` and `pandas`.

## Basic Usage

Conceptually, to run `act` you need four components.

1. A `.hoc` file which declares the cell's architecture.
2. Compiled modfiles for this `.hoc` file.
3. A `config.json` file declaring biological parameters of optimization.
4. A data file with observed (target) voltage traces.

The running process consists of three steps.
1. Defining an optimizer and a feature model.
2. Running optimization.
3. Computing goodness-of-fit metrics.

`act.optim` provides several built-in optimizers and a generic class for creating custom optimizers. Some optimizers, such as `NaiveLinearOptimizer`, do not require target data for training, although most optimizers do.

Most optimizers also allow you to specify a feature model, a `torch.nn.Module` which will be used to extract summary features from generated voltage data. `act.feature_model` provides a `DefaultSummaryModel` which can also be used as a reference for creating custom feature models.

`act.metrics` contains standardized goodness-of-fit metrics which can be used as a reference for creating custom metrics. 

A snapshot example is presented below. Please, see `examples/CA3/` for a more detailed overview.

```python
import torch
import numpy as np
import pandas as pd
from act.feature_model import DefaultSummaryModel
from act.optim import LinearOptimizer
from act.metrics import correlation_score

# Load observed data
data = torch.tensor(pd.read_csv("example_data.csv", header = None).to_numpy()).float()

# Reshape to (num_current_injections, 1024) for convenience
data = data.reshape((5, 1024))

# Define optimizer
linopt = LinearOptimizer(config_file = "config.json")

# Define feature model
feature_model = DefaultSummaryModel(num_summary_features = 8, use_statistics = True)

# Optimize
linopt_estimates = linopt.optimize(feature_model = feature_model, observed_data = data, num_summary_features = 11, num_epochs = 100, num_prediction_rounds = 50)

# Check goodness-of-fit
correlation_score(linopt, data, linopt_estimates)
```