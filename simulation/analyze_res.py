import os
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from simulation_configs import selected_config

from act import analysis
import meta_sweep

def main(config):
    analysis.print_run_stats(selected_config)


if __name__ == "__main__":
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()
    main(selected_config)
