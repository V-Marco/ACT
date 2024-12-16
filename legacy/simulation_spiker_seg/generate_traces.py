from simulation_configs import selected_config

import sys

from legacy.utils import build_parametric_network, generate_parametric_traces

if __name__ == "__main__":
    if "build" in sys.argv:
        build_parametric_network(selected_config)
    else:
        generate_parametric_traces(selected_config)
