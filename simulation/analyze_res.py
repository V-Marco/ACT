import os

import numpy as np
import pandas as pd
from simulation_configs import pospischilsPYr, pospischilsPYr_passive

from act import analysis


def main():
    # for mode in ["original", "segregated"]:
    #    pospischilsPYr["run_mode"] = mode
    #    analysis.print_run_stats(pospischilsPYr)

    analysis.print_run_stats(pospischilsPYr_passive)


if __name__ == "__main__":
    main()
