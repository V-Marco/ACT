import os
import shutil

from neuron import h
import torch

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


    target_V = load_target_traces(simulation_config)
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
        target_V = load_target_traces(simulation_config, target_v_file=DEFAULT_TARGET_V_LTO_FILE)
        optim = ACTOptimizer(
            simulation_config=simulation_config,
            set_passive_properties=True,
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
            sv = optim.simulate(amp, parameters, lto_target_params).reshape(1, -1)
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
        target_V = load_target_traces(simulation_config, target_v_file=DEFAULT_TARGET_V_HTO_FILE)
        optim = ACTOptimizer(
            simulation_config=simulation_config,
            set_passive_properties=True,
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
            sv = optim.simulate(amp, parameters, hto_target_params).reshape(1, -1)
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
