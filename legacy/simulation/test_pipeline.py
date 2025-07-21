import os
import json
import subprocess

# You must set the selected_config in the file simulation_configs.py to Test_Spiker_orig
# Then you must run the pipeline using the command "sbatch run_pipeline.sh"
# before running this script

current_target_file = "./output/Test_Spiker_orig_5-slice/target/target_v.json"
expected_target_file = "../data/TestSpiker/expected_results/exp_target_v.json"

current_volt_file = "./output/Test_Spiker_orig_5-slice/sim_data/output/v_report.h5"
expected_volt_file = "../data/TestSpiker/expected_results/exp_v_report.h5"

current_spike_file = "./output/Test_Spiker_orig_5-slice/sim_data/output/spikes.h5"
expected_spike_file = "../data/TestSpiker/expected_results/exp_spikes.h5"

current_out_file = "./output/Test_Spiker_orig_5-slice/final/53-seed/learned_params.json"
expected_out_file = "../data/TestSpiker/expected_results/exp_learned_params.json"

#________ TARGET DATA _________
if(os.path.exists(current_target_file)):
    current_f = open(current_target_file)
    current_pipeline_output = json.load(current_f)

    if(os.path.exists(expected_target_file)):
        expected_f= open(expected_target_file)
        expected_pipeline_output = json.load(expected_f)

        if(current_pipeline_output == expected_pipeline_output):
            print("The target data matches.")
            print("_____________________________________________________________")
        else:
            print("The target data is NOT expected.")
            print("_____________________________________________________________")
    else:
        print(f"The expected output file could not be found. Looking for {expected_target_file}")
        print("_____________________________________________________________")
else:
    print("Can't find target data.")
    print("Please complete the following tasks before running this script:")
    print("1) Set the selected_config in the file 'simulation_configs.py' to 'Test_Spiker_orig'.")
    print("2) Run the pipeline using the command 'sbatch run_pipeline.sh'.")
    print("_____________________________________________________________")


#________ SIM DATA _________
if(os.path.exists(current_volt_file)):
    if(os.path.exists(expected_volt_file)):
        subprocess.call(["h5diff", "-r", f"{current_volt_file}", f"{expected_volt_file}"])
        print("_____________________________________________________________")
    else:
        print(f"The expected output file could not be found. Looking for {expected_volt_file}")
        print("_____________________________________________________________")
else:
    print("Can't find simulation data: v_report.h5")
    print("Please complete the following tasks before running this script:")
    print("1) Set the selected_config in the file 'simulation_configs.py' to 'Test_Spiker_orig'.")
    print("2) Run the pipeline using the command 'sbatch run_pipeline.sh'.")
    print("_____________________________________________________________")

if(os.path.exists(current_spike_file)):
    if(os.path.exists(expected_spike_file)):
        subprocess.call(["h5diff", "-r", f"{current_spike_file}", f"{expected_spike_file}"])
        print("_____________________________________________________________")
    else:
        print(f"The expected output file could not be found. Looking for {expected_spike_file}")
        print("_____________________________________________________________")
else:
    print("Can't find simulation data: spikes.h5")
    print("Please complete the following tasks before running this script:")
    print("1) Set the selected_config in the file 'simulation_configs.py' to 'Test_Spiker_orig'.")
    print("2) Run the pipeline using the command 'sbatch run_pipeline.sh'.")
    print("_____________________________________________________________")

#________ MODEL PREDICTION _________
if(os.path.exists(current_out_file)):
    current_f = open(current_out_file)
    current_pipeline_output = json.load(current_f)

    if(os.path.exists(expected_out_file)):
        expected_f= open(expected_out_file)
        expected_pipeline_output = json.load(expected_f)

        if(current_pipeline_output == expected_pipeline_output):
            print("The pipeline is running as expected.")
            print("_____________________________________________________________")
        else:
            print("The pipeline is NOT outputting the expected results.")
            print("_____________________________________________________________")
    else:
        print(f"The expected output file could not be found. Looking for {expected_out_file}")
        print("_____________________________________________________________")
else:
    print("Can't find model data.")
    print("Please complete the following tasks before running this script:")
    print("1) Set the selected_config in the file 'simulation_configs.py' to 'Test_Spiker_orig'.")
    print("2) Run the pipeline using the command 'sbatch run_pipeline.sh'.")
    print("_____________________________________________________________")