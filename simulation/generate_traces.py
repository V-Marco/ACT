from act.utils import generate_parametric_traces

from simulation_configs import (
    LA_A_orig,
    LA_A_seg,
    pospischilsPYr,
    pospischilsPYr_passive,
)

if __name__ == "__main__":
    config = LA_A_seg

    generate_parametric_traces(config)
