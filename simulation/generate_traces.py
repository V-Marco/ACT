from simulation_configs import (
    LA_A_orig,
    LA_A_seg,
    pospischilsPYr,
    pospischilsPYr_passive,
)

import sys

from act.utils import build_parametric_network, generate_parametric_traces

if __name__ == "__main__":
    config = LA_A_seg
    if "build" in sys.argv:
        build_parametric_network(config)
    else:
        generate_parametric_traces(config)
