from simulation_configs import pospischilsPYr, pospischilsPYr_passive

from act import simulator

if __name__ == "__main__":
    config = pospischilsPYr_passive

    p = simulator.run(config)
