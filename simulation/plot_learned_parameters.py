import os
import shutil

from neuron import h

from act.analysis import save_plot
from act.target_utils import load_target_traces
from act.utils import load_learned_params
from act.optim import ACTOptimizer

from simulation_configs import selected_config

# will have to generate target traces (python generate_target_traces.py --ignore_segregation)

temp_modfiles_dir = "temp_modfiles"

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
        simulation_config["output"]["folder"], simulation_config["run_mode"], "final"
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
    
    # generate data per amp
    for i, amp in enumerate(amps):
        print(f"Generating trace for {float(amp)*1000} nA")
        
        sv = optim.simulate(amp, parameters, target_params).reshape(1, -1)
        # write to output folder / mode / final
        save_plot(
            amp,
            output_folder,
            simulated_data=sv.cpu().detach().numpy(),
            target_V=target_V[i].cpu().detach().numpy(),
            output_file=f"final_{(amp * 1000):.0f}nA.png",
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
