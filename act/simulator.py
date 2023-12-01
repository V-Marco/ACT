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
from act.target_utils import (
    get_voltage_trace_from_params,
    save_target_traces,
    load_target_traces,
)

temp_modfiles_dir = "temp_modfiles"


def _run_generate_target_traces(config: SimulationConfig, ignore_segregation=False):
    # if there is a target_cell specified then use it too

    if not ignore_segregation:
        segregation_index = utils.get_segregation_index(config)
        segregated_and_lto = config["run_mode"] == "segregated" and config["segregation"][segregation_index].get("use_lto_amps", False)
        segregated_and_hto = config["run_mode"] == "segregated" and config["segregation"][segregation_index].get("use_hto_amps", False)
    else:
        segregated_and_lto = False
        segregated_and_hto = False

    # If we specify a target cell then we should simulate that target
    if config["optimization_parameters"].get("target_cell") and not segregated_and_lto and not segregated_and_hto:
        modfolder = (
            config["optimization_parameters"].get("target_cell").get("modfiles_folder")
        )
    else:
        modfolder = config["cell"]["modfiles_folder"]

    shutil.copytree(modfolder, temp_modfiles_dir, dirs_exist_ok=True)

    os.system(f"nrnivmodl {temp_modfiles_dir}")

    logger = ACTLogger()
    try:
        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
    except:
        logger.info("Mod files already loaded. Continuing.")

    save_target_traces(config, ignore_segregation=ignore_segregation)

    return


def _run(config: SimulationConfig):
    if config["optimization_parameters"]["num_epochs"] < 1:
        raise ValueError("Number of epochs is expected to be >= 1.")

    output_folder = utils.create_output_folder(config)

    # if there is a target_cell specified then use it too
    os.mkdir(temp_modfiles_dir)
    shutil.copytree(
        config["cell"]["modfiles_folder"], temp_modfiles_dir, dirs_exist_ok=True
    )

    os.system(f"nrnivmodl {temp_modfiles_dir}")

    logger = ACTLogger()
    segregation_index = utils.get_segregation_index(config)# if needed
    if config["run_mode"] == "segregated" and config["segregation"][segregation_index].get("use_lto_amps", False):
        print(f"Using LTO Amps for current segregation (use_lto_amps set)")
        amps = config["optimization_parameters"]["lto_amps"]
    elif config["run_mode"] == "segregated" and config["segregation"][segregation_index].get("use_hto_amps", False):
        print(f"Using HTO Amps for current segregation (use_hto_amps set)")
        amps = config["optimization_parameters"]["hto_amps"]
    else:
        amps = config["optimization_parameters"]["amps"]

    logger.info(
        f"Number of amplitudes: {len(amps)}"
    )

    try:
        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
    except:
        logger.info("Mod files already loaded. Continuing.")

    # Get target voltage
    if config["optimization_parameters"]["target_V"] is not None:
        target_V = config["optimization_parameters"]["target_V"]
    elif config["optimization_parameters"]["target_V_file"] is not None:
        target_V = load_target_traces(config)
    elif config["optimization_parameters"]["target_params"] is not None:
        target_V = get_voltage_trace_from_params(config)
    else:
        raise ValueError(
            "Must specify either target_V, target_V_file or target_params for optimization_parameters"
        )

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

    logger.info(f"Target voltage shape: {target_V.shape}")

    # Run the optimizer
    pred_pool = []
    err_pool = []
    fi_err_pool = []
    af_err_pool = []

    params = [p["channel"] for p in config["optimization_parameters"]["params"]]
    for repeat_num in range(config["optimization_parameters"]["num_repeats"]):
        if config["run_mode"] == "original" or config["run_mode"] == "segregated":
            optim = GeneralACTOptimizer(simulation_config=config, logger=logger)
            predictions, train_stats = optim.optimize(target_V)
            predictions_amps = predictions[:,-1].reshape(-1,1)
            predictions = predictions[:,:-1]
        #elif config["run_mode"] == "segregated":
        #    optim = GeneralACTOptimizer(simulation_config=config, logger=logger)
        #    predictions, train_stats = optim.optimize_with_segregation(
        #        target_V, "voltage"
        #    )
        else:
            raise ValueError(
                "run mode not specified, 'original' or 'segregated' supported."
            )
        np_amps = np.array(amps)
        np_predictions_amps = predictions_amps.cpu().detach().flatten().numpy()
        print(f"Amps supplied: {np_amps.tolist()}")
        print(f"Amps predicted: {np_predictions_amps.tolist()}")
        print(f"Amps error: {(np_amps-np_predictions_amps).tolist()}")
        print(f"Amps error sum: {(np_amps-np_predictions_amps).sum()}")

        # output train stats
        print(f"writing training run stats for repeat {repeat_num+1}")
        with open(f"train_stats_repeat_{repeat_num+1}.json", "w") as fp:
            json.dump(train_stats, fp)
        print("done")

        print(f"{repeat_num+1} predictions: {predictions.cpu().detach().tolist()}")
        pred_pool = pred_pool + predictions.cpu().detach().tolist()

        sims = []
        # we want to simulate each amp with each param predicted
        # the one with best overall error is our selection
        print(f"Calculating error...")
        for i, pred in enumerate(predictions.cpu().detach().tolist()):
            sim_list = []
            for j, amp in enumerate(amps):
                sim_list.append(optim.simulate(amp, params, pred).reshape(1, -1))
            sims.append(sim_list)

        decimate_factor = config["optimization_parameters"].get("decimate_factor")
        if decimate_factor:
            print(f"decimate_factor set - reducing sims voltage by {decimate_factor}x")
            from scipy import signal

            sims = [
                [
                    torch.tensor(signal.decimate(sim.cpu(), decimate_factor).copy())
                    for sim in sim_list
                ]
                for sim_list in sims
            ]
        # sims = torch.cat(sims, dim=0)

        # Compute composite error
        # for each prediction
        inj_dur = config["simulation_parameters"]["h_i_dur"]
        inj_start = config["simulation_parameters"]["h_i_delay"]

        for j, pred_sim in enumerate(sims):
            total_error = 0
            # for each amp
            for i, sim in enumerate(pred_sim):
                error = mse_score(target_V[i], sim) + (
                    1 - abs(correlation_score(target_V[i], sim))
                )
                total_error = total_error + error

                amp = amps[i]
                if False: # save prediction plot for debugging
                    save_prediction_plots(
                       target_V[i].reshape(1, len(target_V[i])).cpu().detach(),
                        amp,
                        config,
                        predictions.cpu().detach()[j],
                        output_folder,
                        output_file=f"repeat{repeat_num+1}_pred{j+1}_{(amp * 1000):.0f}nA.png",
                    )
            fi_error = utils.get_fi_curve_error(
                torch.cat(pred_sim), target_V, torch.tensor(amps), inj_dur=inj_dur
            )
            pred_ampl, pred_freq = utils.get_amplitude_frequency(
                torch.cat(pred_sim), inj_dur=inj_dur, inj_start=inj_start
            )
            target_ampl, target_freq = utils.get_amplitude_frequency(
                target_V, inj_dur=inj_dur, inj_start=inj_start
            )
    
            af_error = float((pred_freq - target_freq).abs().sum().cpu())

            err_pool.append(error)
            fi_err_pool.append(fi_error)
            af_err_pool.append(af_error)

        # save prediction values
        mean = np.mean(predictions.cpu().detach().tolist(), axis=0).tolist()
        variance = np.var(predictions.cpu().detach().tolist(), axis=0).tolist()

        p_file = os.path.join(output_folder, f"repeat{repeat_num+1}_predictions.json")
        pred_dict = {
            "predictions": predictions.cpu().detach().tolist(),
            "mean": mean,
            "var": variance,
        }
        with open(p_file, "w") as fp:
            json.dump(pred_dict, fp, indent=4)

        print(f"Repeat {repeat_num+1} mean: {mean} | variance: {variance}")

    print(f"All predictions: {pred_pool}")
    print(f"Err per prediction: {err_pool}")
    print(f"FI Err per prediction: {fi_err_pool}")
    print(f"Amplitude/Frequency err per prediction {af_err_pool}")

    # old way, error was not reliable, a flat line beats spikes offset by a few ms
    
    if config["run_mode"] == "segregated":
        if config["segregation"][segregation_index].get("selection_metric") == "amplitude_frequency_error":
            print("Minimal Amplitude and Frequency error selected for parameter selection")
            predictions = pred_pool[np.argmin(af_err_pool)]
        if config["segregation"][segregation_index].get("selection_metric") == "mse":
            print("MSE error selected for parameter selection")
            predictions = pred_pool[np.argmin(err_pool)]
        else:
            print(f"FI error selected for parameter selection")
            predictions = pred_pool[np.argmin(np.abs(fi_err_pool))]
    else:# by default I want fi error
        predictions = pred_pool[np.argmin(np.abs(fi_err_pool))]

    print(f"Best prediction: {predictions}")

    with open(os.path.join(output_folder, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(dict(zip(params, predictions)), index=[0])

    g_leak_var = optim.cell.gleak_var
    g_bar_leak = optim.cell.g_bar_leak
    if g_leak_var and g_bar_leak:  # if the user set the passive properties
        pred_df.insert(0, g_leak_var, [g_bar_leak])

    pred_df.to_csv(os.path.join(output_folder, "pred.csv"))
    save_mse_corr(target_V, config, predictions, output_folder, amps=amps)

    # save passive properties
    passive_properties, passive_v = optim.calculate_passive_properties(
        params, predictions
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
        while i < len(amps):
            amp_i = amps[i]
            target_Vi = target_V[i].reshape(1, -1)
            simulated_Vi = save_prediction_plots(
                target_Vi.cpu().detach().numpy(),
                amps[i],
                config,
                predictions,
                output_folder,
            )
            simulated_V_out.append(simulated_Vi)
            target_V_out.append(target_Vi)
            amp_out.append(amp_i)

            i += 1  # should be user variable

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

    if config['run_mode'] == "segregated":
        learned_params = {param:predict for param,predict in zip(params, predictions)}
        
        # save a copy of the outputs for future development
        base_output_folder = config["output"]["folder"]
        run_output_folder_name = f"{config['run_mode']}"
        seg_folder = os.path.join(base_output_folder, f"{run_output_folder_name}_seg{segregation_index+1}")
        shutil.copytree(output_folder, seg_folder, dirs_exist_ok=True)

        utils.update_segregation(config, learned_params)

def run_generate_target_traces(config: SimulationConfig, subprocess=True, ignore_segregation=False):
    try:
        if subprocess:
            p = Process(target=_run_generate_target_traces, args=[config, ignore_segregation])
            p.start()
            p.join()
            p.terminate()
        else:
            _run_generate_target_traces(config, ignore_segregation)
    except:
        raise
    finally:  # always remove this folder
        if os.path.exists("x86_64"):
            os.system("rm -r x86_64")
        if os.path.exists(temp_modfiles_dir):
            os.system("rm -r " + temp_modfiles_dir)


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
        if os.path.exists(temp_modfiles_dir):
            os.system("rm -r " + temp_modfiles_dir)
