from simulation_constants import pospischilsPYr

from act import simulator

if __name__ == "__main__":
    constants = pospischilsPYr

    p = simulator.run(constants)
