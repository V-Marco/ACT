import datetime
import json
import os
import sys
from multiprocessing import Process

import numpy as np
import pandas as pd
import torch
from neuron import h

from act.act_types import SimulationConfig
from act.analysis import save_mse_corr, save_prediction_plots
from act.logger import ACTLogger
from act.metrics import correlation_score, mse_score
from act.optim import GeneralACTOptimizer
from act.target_utils import get_voltage_trace_from_params


def _run(config: SimulationConfig):
    if config["optimization_parameters"]["num_epochs"] < 1000:
        raise ValueError("Number of epochs is expected to be >= 1000.")

    output_folder = config["output"]["folder"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    os.system(f"nrnivmodl {config['cell']['modfiles_folder']}")

    logger = ACTLogger()
    logger.info(
        f"Number of amplitudes: {len(config['optimization_parameters']['amps'])}"
    )

    try:
        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
    except:
        logger.info("Mod files already loaded. Continuing.")

    # Get target voltage
    if config["optimization_parameters"]["target_V"] is not None:
        target_V = config["optimization_parameters"]["target_V"]
    elif config["optimization_parameters"]["target_params"] is not None:
        target_V = get_voltage_trace_from_params(config)
    else:
        raise ValueError(
            "Must specify either target_V or target_params for optimization_parameters"
        )

    logger.info(f"Target voltage shape: {target_V.shape}")

    # Run the optimizer
    pred_pool = []
    err_pool = []
    params = [p["channel"] for p in config["optimization_parameters"]["params"]]
    for _ in range(config["optimization_parameters"]["num_repeats"]):
        if config["run_mode"] == "original":
            optim = GeneralACTOptimizer(simulation_config=config, logger=logger)
            predictions = optim.optimize(target_V)
        elif config["run_mode"] == "segregated":
            optim = GeneralACTOptimizer(simulation_config=config, logger=logger)
            predictions = optim.optimize_with_segregation(target_V, "voltage")
        else:
            raise ValueError(
                "run mode not specified, 'original' or 'segregated' supported."
            )

        pred_pool.append(predictions)

        sims = []
        for amp in config["optimization_parameters"]["amps"]:
            sims.append(optim.simulate(amp, params, predictions).reshape(1, -1))
        sims = torch.cat(sims, dim=0)

        # Compute composite error
        error = mse_score(target_V, sims) + (1 - abs(correlation_score(target_V, sims)))
        err_pool.append(error)

    predictions = pred_pool[np.argmin(err_pool)]

    run_output_folder_name = f"{config['run_mode']}"
    output_folder = os.path.join(config["output"]["folder"], run_output_folder_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(os.path.join(output_folder, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(dict(zip(params, predictions.detach().numpy())), index=[0])
    pred_df.to_csv(os.path.join(output_folder, "pred.csv"))

    save_mse_corr(target_V, config, predictions, output_folder)

    if config["output"]["produce_plots"]:
        i = 0
        while i < len(config["optimization_parameters"]["amps"]):
            save_prediction_plots(
                target_V[i].reshape(1, -1),
                config["optimization_parameters"]["amps"][i],
                config,
                predictions,
                output_folder,
            )
            i += 5  # should be user variable

    if os.path.exists("x86_64"):
        os.system("rm -r x86_64")


def run(config: SimulationConfig):
    _run(config)
    # p = Process(target=_run, args=[config])
    # p.start()
    # p.join()
    # p.terminate()
