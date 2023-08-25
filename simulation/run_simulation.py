import sys

sys.path.append("../")

import datetime
import os
from multiprocessing import Process

import numpy as np
import pandas as pd
import torch
from neuron import h
from simulation_constants import PospischilsPY, PospischilsPYr

from act.analysis import save_mse_corr, save_prediction_plots
from act.logger import ACTLogger
from act.metrics import correlation_score, mse_score
from act.optim import GeneralACTOptimizer
from act.target_utils import get_voltage_trace_from_params


def main(constants: object):
    # Compile modfiles
    os.system(f"nrnivmodl {constants.modfiles_folder}")
    h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")

    logger = ACTLogger()
    logger.info(f"Number of amplitudes: {len(constants.amps)}")

    # Get target voltage
    if constants.target_V is not None:
        target_V = constants.target_V.copy()
    elif constants.target_params is not None:
        target_V = get_voltage_trace_from_params(constants)
    else:
        raise ValueError

    logger.info(f"Target voltage shape: {target_V.shape}")

    # Run the optimizer
    pred_pool = []
    err_pool = []
    for _ in range(constants.num_repeats):
        if constants.run_mode == "original":
            optim = GeneralACTOptimizer(simulation_constants=constants, logger=logger)
            predictions = optim.optimize(target_V)
        elif constants.run_mode == "segregated":
            optim = GeneralACTOptimizer(simulation_constants=constants, logger=logger)
            predictions = optim.optimize_with_segregation(target_V, "voltage")
        else:
            raise ValueError

        pred_pool.append(predictions)

        sims = []
        for amp in constants.amps:
            sims.append(
                optim.simulate(amp, constants.params, predictions).reshape(1, -1)
            )
        sims = torch.cat(sims, dim=0)

        # Compute composite error
        error = mse_score(target_V, sims) + (1 - abs(correlation_score(target_V, sims)))
        err_pool.append(error)

    predictions = pred_pool[np.argmin(err_pool)]

    run_output_folder_name = f"{constants.run_mode}_{constants.modfiles_mode}"
    output_folder = os.path.join(constants.output_folder, run_output_folder_name)
    os.mkdir(output_folder)

    # Save constants
    with open(os.path.join(output_folder, "constants.txt"), "w") as file:
        file.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "\n")
        file.write(
            "\n".join(["%s = %s" % (k, v) for k, v in constants.__dict__.items()])
        )

    # Save predictions
    pred_df = pd.DataFrame(
        dict(zip(constants.params, predictions.detach().numpy())), index=[0]
    )
    pred_df.to_csv(os.path.join(output_folder, "pred.csv"))

    save_mse_corr(target_V, constants, predictions, output_folder)

    if constants.produce_plots:
        i = 0
        while i < len(constants.amps):
            save_prediction_plots(
                target_V[i].reshape(1, -1),
                constants.amps[i],
                constants,
                predictions,
                output_folder,
            )
            i += 5


if __name__ == "__main__":
    constants = PospischilsPYr

    if not os.path.exists(constants.output_folder):
        os.mkdir(constants.output_folder)

    if constants.num_epochs < 1000:
        raise ValueError("Number of epochs is expected to be >= 1000.")

    # Original
    p = Process(target=main, args=[constants])
    p.start()
    p.join()
    p.terminate()

    if os.path.exists("x86_64"):
        os.system("rm -r x86_64")
