import argparse
import json
import os
import shutil

from neuron import h
import torch
import sys
sys.path.append("../")
from act.analysis import save_plot, plot_fi_curves
from act.target_utils import load_target_traces
from act.utils import load_learned_params, get_fi_curve, get_fi_curve_error
from act.optim import ACTOptimizer
from act.target_utils import DEFAULT_TARGET_V_LTO_FILE, DEFAULT_TARGET_V_HTO_FILE

from simulation_configs import selected_config

# will have to generate target traces (python generate_target_traces.py --ignore_segregation)

temp_modfiles_dir = "temp_modfiles"

def save_fi(config, simulated_traces, target_traces, amps):
    inj_dur = config["simulation_parameters"]["h_i_dur"]

    output_file = os.path.join(
        config["output"]["folder"], "final" , "final_fi_curve"
    )

    simulated_curve = get_fi_curve(simulated_traces, amps, inj_dur=inj_dur)
    target_curve = get_fi_curve(target_traces, amps, inj_dur=inj_dur)

    simulated_label = config["output"]["simulated_label"]
    target_label = config["output"]["target_label"]

    err1 = get_fi_curve_error(simulated_traces, target_traces, amps, inj_dur=inj_dur)
    simulated_label = simulated_label + f" (err: {err1})"

    curves_list = [
        simulated_curve.cpu().detach().numpy(),
        target_curve.cpu().detach().numpy(),
    ]
    labels = [simulated_label, target_label]

    #if extra_trace is not None:
    #    extra_trace_fi = utils.get_fi_curve(extra_trace, amps, inj_dur=inj_dur)
    #    extra_trace_fi = extra_trace_fi.cpu().detach().numpy()
    #    curves_list.append(extra_trace_fi)
    #    err2 = utils.get_fi_curve_error(extra_trace, target_traces, amps, inj_dur=inj_dur)
    #    extra_trace_label = extra_trace_label + f" (err: {err2})"
    #    labels.append(extra_trace_label) 
    title="FI Curve"
    plot_fi_curves(
        curves_list, amps.cpu().detach().numpy(), labels=labels, title=title, output_file=output_file
    )


def run(simulation_config):

    # if there is a target_cell specified then use it too
    os.mkdir(temp_modfiles_dir)
    shutil.copytree(
        simulation_config["cell"]["modfiles_folder"], temp_modfiles_dir, dirs_exist_ok=True
    )

    os.system(f"nrnivmodl {temp_modfiles_dir}")

    try:
        h.nrn_load_dll("./x86_64/.libs/libnrnmech.so")
    except:
        print("Mod files already loaded. Continuing.")


    target_V = load_target_traces(simulation_config, ignore_segregation=True)
    optim = ACTOptimizer(
        simulation_config=simulation_config,
        set_passive_properties=True,
        ignore_segregation=True
    )
    
    # create output folders
    output_folder = os.path.join(
        simulation_config["output"]["folder"], "final"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
       
    dt = simulation_config["simulation_parameters"]["h_dt"]
    amps = simulation_config["optimization_parameters"]["amps"]
    simulated_label = simulation_config["output"]["simulated_label"]
    target_label = simulation_config["output"]["target_label"]
    learned_params = load_learned_params(simulation_config)
    parameters = [k for k,v in learned_params.items()]
    target_params = [v for k,v in learned_params.items()]
    
    sv_list = []
    # generate data per amp
    for i, amp in enumerate(amps):
        print(f"Generating trace for {float(amp)*1000} pA")
        print(f"Setting params: {learned_params}")
        sv = optim.simulate(amp, parameters, target_params).reshape(1, -1)
        # write to output folder / final
        sv_list.append(sv)
        save_plot(
            amp,
            output_folder,
            simulated_data=sv.cpu().detach().numpy(),
            target_V=target_V[i].cpu().detach().numpy(),
            output_file=f"final_{(amp * 1000):.0f}pA.png",
            dt=dt,
            simulated_label=simulated_label,
            target_label=target_label,
        )
    print("saving fi")
    sv_tensor = torch.cat(sv_list)
    save_fi(simulation_config, sv_tensor, target_V, torch.tensor(amps))

    if os.path.exists(DEFAULT_TARGET_V_LTO_FILE): # we have generated an LTO file before and should simulate at end
        # need to go case by case
        i_dur = 0
        i_delay = 0
        tstop = 0
        ramp_splits = 1
        ramp_time = 0
        module = None
        for m in simulation_config.get("segregation", []):
            if m.get("use_lto_amps"):
                module = m
        if module:
            i_dur = module.get("h_i_dur", 0)
            i_delay = module.get("h_i_delay", 0)
            tstop = module.get("h_tstop", 0)
            ramp_time = module.get("ramp_time", 0)
            ramp_splits = module.get("ramp_splits", 1)
        else:
            raise Exception("NO LTO MODULE FOUND") # comment out for original cases, need to have a param in config or something
            i_dur = 1000
            i_delay = 250
            tstop = 1500
            ramp_time = 1000
            ramp_splits = 20


        target_V = load_target_traces(simulation_config, target_v_file=DEFAULT_TARGET_V_LTO_FILE, ramp_time=ramp_time)
        optim = ACTOptimizer(
            simulation_config=simulation_config,
            set_passive_properties=False,
            ignore_segregation=True
        )

        amps = simulation_config["optimization_parameters"].get("lto_amps")
        if not amps:
            raise Exception(f"No lto_amps specified in config. Either add and generate new target params, or delete {DEFAULT_TARGET_V_LTO_FILE}")
        lto_block_channels = simulation_config["optimization_parameters"]["lto_block_channels"]
        
        # generate data per amp
        for i, amp in enumerate(amps):
            print(f"Generating lto trace for {float(amp)*1000} pA")
            learned_params_lto = dict(learned_params)
            for p in lto_block_channels:
                learned_params_lto[p] = 0

            parameters = [k for k,v in learned_params_lto.items()]
            lto_target_params = [v for k,v in learned_params_lto.items()]
            print(f"{learned_params_lto}")
            sv = optim.simulate(amp, parameters, lto_target_params, cut_ramp=False, ramp_time=ramp_time, ramp_splits=ramp_splits, i_dur=i_dur, i_delay=i_delay, tstop=tstop).reshape(1, -1)
            # write to output folder / final
            save_plot(
                amp,
                output_folder,
                simulated_data=sv.cpu().detach().numpy(),
                target_V=target_V[i].cpu().detach().numpy(),
                output_file=f"final_lto_{(amp * 1000):.0f}pA.png",
                dt=dt,
                simulated_label=simulated_label,
                target_label=target_label,
            )

    if os.path.exists(DEFAULT_TARGET_V_HTO_FILE): # we have generated an HTO file before and should simulate at end
        # need to go case by case
        i_dur = 0
        i_delay = 0
        tstop = 0
        ramp_splits = 1
        ramp_time = 0
        module = None
        for m in simulation_config.get("segregation",[]):
            if m.get("use_hto_amps"):
                module = m
        if module:
            i_dur = module.get("h_i_dur", 0)
            i_delay = module.get("h_i_delay", 0)
            tstop = module.get("h_tstop", 0)
            ramp_time = module.get("ramp_time", 0)
            ramp_splits = module.get("ramp_splits", 1)
        else:
            raise Exception("NO HTO MODULE FOUND")
            i_dur = 1000
            i_delay = 250
            tstop = 1500
            ramp_time = 1000
            ramp_splits = 20

        target_V = load_target_traces(simulation_config, target_v_file=DEFAULT_TARGET_V_HTO_FILE, ramp_time=ramp_time)
        optim = ACTOptimizer(
                simulation_config=simulation_config,
            set_passive_properties=False,
            ignore_segregation=True
        )

        amps = simulation_config["optimization_parameters"].get("hto_amps")
        if not amps:
            raise Exception(f"No hto_amps specified in config. Either add and generate new target params, or delete {DEFAULT_TARGET_V_HTO_FILE}")
        hto_block_channels = simulation_config["optimization_parameters"]["hto_block_channels"]

        # generate data per amp
        for i, amp in enumerate(amps):
            print(f"Generating hto trace for {float(amp)*1000} pA")
            learned_params_hto = dict(learned_params)
            for p in hto_block_channels:
                learned_params_hto[p] = 0

            parameters = [k for k,v in learned_params_hto.items()]
            hto_target_params = [v for k,v in learned_params_hto.items()]
            print(f"{learned_params_hto}")
            sv = optim.simulate(amp, parameters, hto_target_params, cut_ramp=False, ramp_time=ramp_time, ramp_splits=ramp_splits, i_dur=i_dur, i_delay=i_delay, tstop=tstop).reshape(1, -1)
            # write to output folder / final
            save_plot(
                amp,
                output_folder,
                simulated_data=sv.cpu().detach().numpy(),
                target_V=target_V[i].cpu().detach().numpy(),
                output_file=f"final_hto_{(amp * 1000):.0f}pA.png",
                dt=dt,
                simulated_label=simulated_label,
                target_label=target_label,
            )
    
    # print learned params and the difference between target
    print("Learned Params")
    print(f"{learned_params}")
    print()
    params = [p["channel"] for p in simulation_config["optimization_parameters"]["params"]]
    vals = simulation_config["optimization_parameters"]["target_params"]
    target_params = {p:v for p,v in zip(params, vals)}

    print("Target Params")
    print(f"{target_params}")
    print()

    print("Difference")
    diff = {}
    diff_percent = {}
    for p,v in learned_params.items():
        diff[p] = v - target_params[p]
        diff_percent[p] = round(((v - target_params[p])/target_params[p])*100, 2)
    print(f"{diff}")
    print()
    print("Difference (percent)")
    print(f"{diff_percent}")
    print()

    # write to file
    lpd = {}
    lpd["learned_params"] = learned_params
    lpd["target_params"] = target_params
    lpd["difference"] = diff
    lpd["difference_percent"] = diff_percent
    file_path = os.path.join(output_folder, "learned_params.json")
    with open(file_path, "w") as f:
        json.dump(lpd, f, indent=2)
    print("done")


if __name__ == '__main__':
    # will have to generate target traces (python generate_target_traces.py --ignore_segregation)
    try:
        run(selected_config)
    except:
        raise
    finally:  # always remove this folder
        if os.path.exists("x86_64"):
            os.system("rm -r x86_64")
        if os.path.exists(temp_modfiles_dir):
            os.system("rm -r " + temp_modfiles_dir)
