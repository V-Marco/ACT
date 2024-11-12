import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    for i in range(1):
        sim.submit_job(
            HayParameters(
                f"sim_{i}", 
                h_tstop = 1000, # (ms)
                h_i_amplitude = 10, # (nA)
                h_i_duration = 900, # (ms)
                h_i_delay = 100 # (ms)
                ))
    sim.run(batch_size = 1)

