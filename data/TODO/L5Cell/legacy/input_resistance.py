'''
Note: segment manager computs axial currents from adjacent segment to the target segment. This code flips that directional relationship by multiplying by -1 when plotting axial currents.
'''

import sys
sys.path.append("../")

import pickle
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss
from mpl_toolkits import mplot3d
import pdb #python debugger

from Modules.logger import Logger

from Modules.plotting_utils import plot_adjacent_segments
from Modules.segment import SegmentManager
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neytomin_Hay/2023-10-12_21-17-43_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_-1000/" #"output/BenModel/"



import importlib
def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module
constants = load_constants_from_folder(output_folder)

if 'BenModel' in output_folder:
  constants.save_every_ms = 3000
  constants.h_tstop = 3000
  transpose =True
else:
  transpose=False
#  constants.save_every_ms = 200
#  constants.h_tstop = 2500
dt=constants.h_dt

#print(constants.h_dt, constants.save_every_ms, constants.h_tstop)

amplitude = -1

def main():
  save_path = os.path.join(output_folder, "Analysis Currents")
  if os.path.exists(save_path):
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Directory already exists: {save_path}')
  else:
    os.mkdir(save_path)
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Creating Directory: {save_path}')
    
  step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
  steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
  #print(step_size, steps)
  t = []
  #for dir in os.listdir(output_folder): # list folders in directory
#  for step in steps:
#      dirname = os.path.join(output_folder, f"saved_at_step_{step}")
#      print(dirname)
#      with h5py.File(os.path.join(dirname, "t.h5")) as file:
#          t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
#  t = np.hstack(t) # (ms)
#  print(t)
#  t=np.append(t,(t[-1]+dt)) # fix for if t vec is one index short of the data # for some reason this fix changes the length of seg data too?
#  print(t)

  #random_state = np.random.RandomState(random_state)
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=300, transpose=transpose)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  
  plt.plot(t,sm.segments[0].v)



if __name__ == "__main__":
    main()