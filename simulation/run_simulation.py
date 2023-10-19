from simulation_configs import selected_config

from act import simulator

if __name__ == "__main__":

    p = simulator.run(selected_config, subprocess=False)
