import sys
sys.path.append("../")
from act import simulator

from simulation_configs import selected_config

if __name__ == "__main__":

    ignore_segregation = False
    if '--ignore_segregation' in sys.argv:
        ignore_segregation = True
        print('ignoring segregation, typically used for generating final traces')

    print("generating traces...")
    simulator.run_generate_target_traces(selected_config, subprocess=False, ignore_segregation=ignore_segregation)
    print("done")
