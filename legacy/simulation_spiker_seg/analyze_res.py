import os

import numpy as np
import pandas as pd
from simulation_configs import selected_config

from legacy import analysis


def main():
    analysis.print_run_stats(selected_config)


if __name__ == "__main__":
    main()
