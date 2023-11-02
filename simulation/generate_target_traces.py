from act import target_utils

from simulation_configs import selected_config

if __name__ == "__main__":
    target_utils.save_target_traces(selected_config)
    print("Target traces saved.")