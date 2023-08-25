import datetime
import json
import os
import sys
from multiprocessing import Process

import numpy as np
import pandas as pd
import torch
from neuron import h

from act.act_types import SimulationConstants
from act.analysis import save_mse_corr, save_prediction_plots
from act.logger import ACTLogger
from act.metrics import correlation_score, mse_score
from act.optim import GeneralACTOptimizer
from act.target_utils import get_voltage_trace_from_params


def _run(constants: SimulationConstants):
    if constants["optimization_parameters"]["num_epochs"] < 1000:
        raise ValueError("Number of epochs is expected to be >= 1000.")

    output_folder = constants["output"]["folder"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    os.system(f"nrnivmodl {constants['cell']['modfiles_folder']}")

    logger = ACTLogger()
    logger.info(
        f"Number of amplitudes: {len(constants['optimization_parameters']['amps'])}"
    )

    try:
        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
    except:
        logger.info("Mod files already loaded. Continuing.")

    # Get target voltage
    if constants["optimization_parameters"]["target_V"] is not None:
        target_V = constants["optimization_parameters"]["target_V"]
    elif constants["optimization_parameters"]["target_params"] is not None:
        target_V = get_voltage_trace_from_params(constants)
    else:
        raise ValueError(
            "Must specify either target_V or target_params for optimization_parameters"
        )

    logger.info(f"Target voltage shape: {target_V.shape}")

    # Run the optimizer
    pred_pool = []
    err_pool = []
    params = [p["channel"] for p in constants["optimization_parameters"]["params"]]
    for _ in range(constants["optimization_parameters"]["num_repeats"]):
        if constants["run_mode"] == "original":
            optim = GeneralACTOptimizer(simulation_constants=constants, logger=logger)
            predictions = optim.optimize(target_V)
        elif constants["run_mode"] == "segregated":
            optim = GeneralACTOptimizer(simulation_constants=constants, logger=logger)
            predictions = optim.optimize_with_segregation(target_V, "voltage")
        else:
            raise ValueError(
                "run mode not specified, 'original' or 'segregated' supported."
            )

        pred_pool.append(predictions)

        sims = []
        for amp in constants["optimization_parameters"]["amps"]:
            sims.append(optim.simulate(amp, params, predictions).reshape(1, -1))
        sims = torch.cat(sims, dim=0)

        # Compute composite error
        error = mse_score(target_V, sims) + (1 - abs(correlation_score(target_V, sims)))
        err_pool.append(error)

    predictions = pred_pool[np.argmin(err_pool)]

    run_output_folder_name = f"{constants['run_mode']}"
    output_folder = os.path.join(
        constants["output"]["output_folder"], run_output_folder_name
    )
    os.mkdir(output_folder)

    with open(os.path.join(output_folder, "constants.json"), "w") as file:
        json.dump(constants, file, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(dict(zip(params, predictions.detach().numpy())), index=[0])
    pred_df.to_csv(os.path.join(output_folder, "pred.csv"))

    save_mse_corr(target_V, constants, predictions, output_folder)

    if constants["output"]["produce_plots"]:
        i = 0
        while i < len(constants["optimization_parameters"]["amps"]):
            save_prediction_plots(
                target_V[i].reshape(1, -1),
                constants["optimization_parameters"]["amps"][i],
                constants,
                predictions,
                output_folder,
            )
            i += 5  # should be user variable

    if os.path.exists("x86_64"):
        os.system("rm -r x86_64")


def run(constants: SimulationConstants):
    p = Process(target=_run, args=[constants])
    p.start()
    p.join()
    p.terminate()
