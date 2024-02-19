import os
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from simulation_configs import selected_config

from act import analysis


def main():
    analysis.print_run_stats(selected_config)


if __name__ == "__main__":
    main()
