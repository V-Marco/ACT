from simulation_configs import pospischilsPYr, pospischilsPYr_passive, LA_A_seg

from act import simulator

if __name__ == "__main__":
    config = LA_A_seg

    p = simulator.run(config, subprocess=False)
