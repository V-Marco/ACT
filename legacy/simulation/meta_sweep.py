import json
import numpy as np

from simulation_configs import selected_config

def get_meta_params_for_sweep():
    meta_data = json.load(open('meta_params.json'))
    run_num = meta_data["run_num"]
    
    list_param_lens = []
    parameter_slice_length = len(meta_data["parameter_slice_list"])
    list_param_lens.append(parameter_slice_length)

    random_seed_length = len(meta_data["random_seed_list"])
    list_param_lens.append(random_seed_length)

    index_list = get_index_tuple(run_num,list_param_lens)

    config = selected_config

    config["optimization_parameters"]["parametric_distribution"]["n_slices"] = meta_data["parameter_slice_list"][index_list[0]]
    config["optimization_parameters"]["random_seed"] = meta_data["random_seed_list"][index_list[1]]

    return config


def get_index_tuple(run_num, list_param_lens):
    return np.unravel_index(run_num, list_param_lens)

def increment_config_sweep_number():
    # Increment which configuration we are on
    meta_data = json.load(open('meta_params.json'))
    with open('meta_params.json', 'w') as json_file:
        meta_data["run_num"] = meta_data["run_num"] + 1
        json.dump(meta_data, json_file)

def get_number_of_configs():
    meta_data = json.load(open('meta_params.json'))
    list_param_lens = []
    parameter_slice_length = len(meta_data["parameter_slice_list"])
    list_param_lens.append(parameter_slice_length)

    random_seed_length = len(meta_data["random_seed_list"])
    list_param_lens.append(random_seed_length)

    return np.prod(list_param_lens)