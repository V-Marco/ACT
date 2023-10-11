import json
import os
import shutil
from multiprocessing import Process

import h5py
import numpy as np
import pandas as pd
import torch
from neuron import h

from act import utils
from act.act_types import SimulationConfig
from act.analysis import save_mse_corr, save_plot, save_prediction_plots
from act.logger import ACTLogger
from act.metrics import correlation_score, mse_score
from act.optim import GeneralACTOptimizer
from act.target_utils import get_voltage_trace_from_params


def _run(config: SimulationConfig):
    if config["optimization_parameters"]["num_epochs"] < 1000:
        raise ValueError("Number of epochs is expected to be >= 1000.")

    output_folder = utils.create_output_folder(config)

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
        decimate_factor = config["optimization_parameters"].get("decimate_factor")
        if decimate_factor:
            print(
                f"decimate_factor set - reducing generated target voltage by {decimate_factor}x"
            )
            from scipy import signal

            traces = signal.decimate(
                target_V.cpu(), decimate_factor
            ).copy()  # copy per neg index err
            target_V = torch.tensor(traces)

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
        decimate_factor = config["optimization_parameters"].get("decimate_factor")
        if decimate_factor:
            print(f"decimate_factor set - reducing sims voltage by {decimate_factor}x")
            from scipy import signal

            sims = [
                torch.tensor(signal.decimate(sim.cpu(), decimate_factor).copy())
                for sim in sims
            ]
        sims = torch.cat(sims, dim=0)

        # Compute composite error
        error = mse_score(target_V, sims) + (1 - abs(correlation_score(target_V, sims)))
        err_pool.append(error)

    predictions = pred_pool[np.argmin(err_pool)]

    with open(os.path.join(output_folder, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(
        dict(zip(params, predictions.cpu().detach().numpy())), index=[0]
    )

    g_leak_var = optim.cell.gleak_var
    g_bar_leak = optim.cell.g_bar_leak
    if g_leak_var and g_bar_leak:  # if the user set the passive properties
        pred_df.insert(0, g_leak_var, [g_bar_leak])

    pred_df.to_csv(os.path.join(output_folder, "pred.csv"))
    save_mse_corr(target_V, config, predictions, output_folder)

    # save passive properties
    passive_properties, passive_v = optim.calculate_passive_properties(
        params, predictions.cpu().detach().numpy()
    )
    with open(
        os.path.join(output_folder, "pred_passive_properties.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(passive_properties, fp, indent=2)

    if config["output"]["produce_plots"]:
        # output passive amps don't matter
        dt = config["simulation_parameters"]["h_dt"]
        simulated_label = config["output"].get("simulated_label", "Simulated")
        save_plot(
            -0.1,
            output_folder,
            simulated_data=passive_v,
            output_file="passive_-100nA.png",
            dt=dt,
            simulated_label=simulated_label,
        )

        i = 0
        amp_out = []
        simulated_V_out = []
        target_V_out = []
        while i < len(config["optimization_parameters"]["amps"]):
            amp_i = config["optimization_parameters"]["amps"][i]
            target_Vi = target_V[i].reshape(1, -1)
            simulated_Vi = save_prediction_plots(
                target_Vi.cpu().detach().numpy(),
                config["optimization_parameters"]["amps"][i],
                config,
                predictions.cpu().detach().numpy(),
                output_folder,
            )
            simulated_V_out.append(simulated_Vi)
            target_V_out.append(target_Vi)
            amp_out.append(amp_i)

            i += 5  # should be user variable

        # Save the voltage traces for debugging
        target_V_list = [list(t.cpu().detach().numpy()[0]) for t in target_V_out]
        simulated_V_list = [list(t.cpu().detach().numpy()[0]) for t in simulated_V_out]

        f = h5py.File(os.path.join(output_folder, "traces.h5"), "w")
        target_grp = f.create_group("target")
        simulated_grp = f.create_group("simulated")

        target_grp.create_dataset(
            "voltage_trace",
            (len(target_V_list), len(target_V_list[0])),
            dtype="f",
            data=target_V_list,
        )
        simulated_grp.create_dataset(
            "voltage_trace",
            (len(simulated_V_list), len(simulated_V_list[0])),
            dtype="f",
            data=simulated_V_list,
        )
        f.create_dataset("amps", (len(amp_out)), dtype="f", data=amp_out)
        f.close()


def run(config: SimulationConfig, subprocess=True):
    try:
        if subprocess:
            p = Process(target=_run, args=[config])
            p.start()
            p.join()
            p.terminate()
        else:
            _run(config)
    except:
        raise
    finally:  # always remove this folder
        if os.path.exists("x86_64"):
            os.system("rm -r x86_64")
