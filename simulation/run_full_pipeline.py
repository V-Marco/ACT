import subprocess
from simulation_configs import selected_config
import sys

import meta_sweep

# See how many runs are needed for this configuration. 
# Original: 1 run   -   Segregated: n runs depending on # of modules

def run_single_config():
    run_mode = selected_config["run_mode"]
    if run_mode == "original":
        number_of_runs = 1
    elif run_mode == "segregated":
        number_of_runs = len(selected_config["segregation"])

    for i in range(0,number_of_runs):
        # The whole pipeline is run through these commands
        subprocess.run(["python", "generate_target_traces.py"]) 
        subprocess.run(["python", "generate_traces.py", "build"]) 
        subprocess.run(["sbatch", "--wait", "batch_generate_traces.sh"]) 
        subprocess.run(["python", "generate_arma_stats.py"]) 
        subprocess.run(["python", "run_simulation.py"]) 
        subprocess.run(["python", "analyze_res.py"]) 
        subprocess.run(["python", "plot_fi.py"]) 
        subprocess.run(["python", "plot_learned_parameters.py"]) 

def run_sweep_of_config():
    run_mode = selected_config["run_mode"]
    if run_mode == "original":
        number_of_runs = 1
    elif run_mode == "segregated":
        number_of_runs = len(selected_config["segregation"])

    number_of_configs = meta_sweep.get_number_of_configs()

    print("------------------------------------------------------")
    print(f"REQUESTING {number_of_configs} CONFIGURATIONS TO BE RUN")
    print("------------------------------------------------------")

    for i in range(0,number_of_configs):
        for j in range(0,number_of_runs):
            # The whole pipeline is run through these commands
            subprocess.run(["python", "generate_target_traces.py", "--sweep"]) 
            subprocess.run(["python", "generate_traces.py", "build", "--sweep"]) 
            subprocess.run(["sbatch", "--wait", "batch_generate_traces_sweep.sh"]) 
            subprocess.run(["python", "generate_arma_stats.py", "--sweep"]) 
            subprocess.run(["python", "run_simulation.py", "--sweep"]) 
            subprocess.run(["python", "analyze_res.py", "--sweep"]) 
            subprocess.run(["python", "plot_fi.py", "--sweep"]) 
            subprocess.run(["python", "plot_learned_parameters.py", "--sweep"]) 
        
        meta_sweep.increment_config_sweep_number()

if __name__ == "__main__":
    if '--sweep' in sys.argv:
        run_sweep_of_config()
    else:
        run_single_config()