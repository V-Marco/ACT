from simulation_configs import pospischilsPYr

from act import simulator

if __name__ == "__main__":
    config = pospischilsPYr

    p = simulator.run(config)
