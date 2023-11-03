from act import simulator

from simulation_configs import selected_config

if __name__ == "__main__":
    print("generating traces...")
    simulator.run_generate_target_traces(selected_config, subprocess=False)
    print("done")
