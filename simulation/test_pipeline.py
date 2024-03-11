import os
import json

# You must set the selected_config in the file simulation_configs.py to Test_Spiker_orig
# Then you must run the pipeline using the command "sbatch run_pipeline.sh"
# before running this script

current_out_file = "./output/Test_Spiker_orig_5-slice/final/53-seed/learned_params.json"
expected_out_file = "../data/TestSpiker/expected_results/expected_learned_params.json"

if(os.path.exists(current_out_file)):
    current_f = open(current_out_file)
    current_pipeline_output = json.load(current_f)

    if(os.path.exists(expected_out_file)):
        expected_f= open(expected_out_file)
        expected_pipeline_output = json.load(expected_f)

        if(current_pipeline_output == expected_pipeline_output):
            print("The pipeline is running as expected.")
        else:
            print("The pipeline is NOT outputting the expected results.")
    else:
        print(f"The expected output file could not be found. Looking for {expected_out_file}")
else:
    print("Please complete the following tasks before running this script:")
    print("1) Set the selected_config in the file 'simulation_configs.py' to 'Test_Spiker_orig'.")
    print("2) run the pipeline using the command 'sbatch run_pipeline.sh'.")