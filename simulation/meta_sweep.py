import json
import subprocess

def set_meta_params():
    x = 1


def run_meta_sweep():
    meta_data = json.load(open('meta_params.json'))
    print(meta_data)
    run_mode_list = meta_data["run_mode_list"]
    print(run_mode_list)
    print(type(run_mode_list))
    cell_list = meta_data["cell_list"]
    model_type_list = meta_data["model_type_list"]
    parameter_slice_list = meta_data["parameter_slice_list"]
    random_seed_list = meta_data["random_seed_list"]
    generate_arma_stats = meta_data["generate_arma_stats"]

    if "orig" in run_mode_list:
        for cell in cell_list :
            for model_type in model_type_list :
                for slices in parameter_slice_list :
                    for seed in random_seed_list :
                        # Set the parameters
                        print("SETTING PARAMETERS")

                        # Run the Pipeline
                        print(f"PARAMS: orig_{cell}_{model_type}_{slices}_{seed}")
                        #subprocess.run("python generate_target_traces.py") 
                        print(f"Target")
                        #subprocess.run(["python generate_traces.py", "build"]) 
                        print(f"Build")
                        #subprocess.run("sbatch batch_generate_traces.sh") 
                        print(f"Simulate")
                        if (generate_arma_stats == True) and (seed == random_seed_list[0]):
                            print(f"Generating ARMA stats")
                            #subprocess.run("python generate_arma_stats.py") 

                        #subprocess.run("python run_simulation.py") 
                        print(f"Run Model")
                        #subprocess.run("python analyze_res.py") 
                        print(f"Results")
                        #subprocess.run("python plot_fi.py") 
                        print(f"FI Plot")
                        #subprocess.run("python plot_learned_parameters.py") 
                        print(f"Final")



    if "seg" in run_mode_list:
        pass

# python generate_target_traces.py && python generate_traces.py build && sbatch batch_generate_traces.sh -W && python generate_arma_stats.py && python run_simulation.py && python analyze_res.py && python plot_fi.py && python plot_learned_parameters.py
if __name__ == "__main__":
    run_meta_sweep()
