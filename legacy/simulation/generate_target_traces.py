import sys
sys.path.append("../")
import os
from act import simulator
from act import utils
import meta_sweep

from simulation_configs import selected_config



if __name__ == "__main__":

    ignore_segregation = False
    if '--ignore_segregation' in sys.argv:
        ignore_segregation = True
        print('ignoring segregation, typically used for generating final traces')
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()

    # Check if target traces are already saved
    '''
    target_dir = utils.get_output_folder_name(selected_config) + "target/"
    target_file_name = selected_config["optimization_parameters"]["target_V_file"]
    if(os.path.exists(target_dir + target_file_name)):
        print("-----------------------------------------------")
        print(f"TARGET DATA ALREADY GENERATED AT: {target_dir}")
        print("-----------------------------------------------")
    else:
    '''
    print("generating traces...")
    simulator.run_generate_target_traces(selected_config, subprocess=False, ignore_segregation=ignore_segregation)
    print("done")
