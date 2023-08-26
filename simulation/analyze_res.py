import os

import numpy as np
import pandas as pd
from simulation_configs import pospischilsPYr

from act import analysis


def main():
    for mode in ["original", "segregated"]:
        pospischilsPYr["run_mode"] = mode
        analysis.print_run_stats(pospischilsPYr)


if __name__ == "__main__":
    main()
