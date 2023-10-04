from simulation_configs import (
    LA_A_orig,
    LA_A_seg,
    pospischilsPYr,
    pospischilsPYr_passive,
)

from act import simulator

if __name__ == "__main__":
    config = LA_A_seg

    p = simulator.run(config, subprocess=False)
