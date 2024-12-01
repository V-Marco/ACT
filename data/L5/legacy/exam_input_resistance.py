'''
Note: Computes input resistance provided a negative current injection. 
Assumes that the membrane voltage will be at steady state in the middle of the current injection, and that the membrane voltage will be at resting membrane voltage halfway between the start of the simulation and the start of the current injection.
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
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neymotin_Hay2/_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_-1000" #"output/BenModel/"

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

def main():
  save_path = os.path.join(output_folder, "Analysis Input Resistance")
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
  #random_state = np.random.RandomState(random_state)
  
  #sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=constants.skip, transpose=transpose)
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=0, transpose=transpose)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  
  #Compute axial currents from each segment toward its adjacent segments.
  #compute axial currents between all segments
  sm.compute_axial_currents()
  
  logger.log(f"soma_spiketimes: {sm.soma_spiketimes}")
  
  logger.log(f'firing_rate: {len(sm.soma_spiketimes) / (len(sm.segments[0].v) * dt / 1000)}') # number of spikes / seconds of simulation
  
  #Find soma segments
  soma_segs = []
  for seg in sm.segments:
    if seg.type == 'soma':
      soma_segs.append(seg)
      
  if len(soma_segs) != 1:
    logger.log(f"Picking 1 out of {len(soma_segs)} Soma segments.")
    #raise(ValueError("There should be only one soma segment."))
    soma_segs=[soma_segs[3]]
    
  with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
      seg_indexes = pickle.load(file)
  axon_seg_index=seg_indexes["axon"]
  
  axon_seg=sm.segments[axon_seg_index]
  soma_seg=soma_segs[0]
  for seg in sm.segments:
    if 'axon' in seg.seg:
      if '[0](0.5)' in seg.seg:
        axon_seg = seg

  def plot_voltage(seg, t, title='Voltage', save_to=None, title_prefix=None):
      fig, ax = plt.subplots()
      
      ax.plot(t, seg.v)
      
      if title_prefix:
          ax.set_title(title_prefix + ' ' + title)
      else:
          ax.set_title(title)
      
      ax.set_xlabel('Time (ms)')
      ax.set_ylabel('Vm (mV)')
      
      if save_to:
          filename = "Voltage.png" if not title_prefix else title_prefix + "Voltage.png"
          fig.savefig(os.path.join(save_to, filename))
          plt.close(fig)
      else:
          plt.show()


  
  # calculate input resistance
  def calc_input_resistance(seg):
      print(seg.seg)
      # get current amplitude
      amp = int(output_folder.split('_')[-1]) / 1000
      print(f"amp: {amp}")
      
      # get resting membrane potential from halway between start of simulation and start of current injection
      rmp_time_index = int((constants.h_i_delay / 2) / constants.h_dt)
      V_steady_during_inj_time_index = int((constants.h_i_delay + (constants.h_i_duration / 2)) / constants.h_dt)
      # with skip below
      #rmp_time_index = int(((constants.h_i_delay / 2) - constants.skip) / constants.h_dt)
      #V_steady_during_inj_time_index = int(((constants.h_i_delay + (constants.h_i_duration / 2)) - constants.skip ) / constants.h_dt)
      rmp = seg.v[rmp_time_index]
      print(f"rmp: {rmp}")
      V_steady_during_inj = seg.v[V_steady_during_inj_time_index]
      print(f"V_steady_during_inj: {V_steady_during_inj}")
      Rin_in_MOhms = (V_steady_during_inj - rmp) / amp
      print(f"Rin_in_MOhms: {Rin_in_MOhms}")
  
  plot_voltage(seg=axon_seg, t=t, save_to=save_path, title_prefix = 'Axon_')
  calc_input_resistance(axon_seg)
  plot_voltage(seg=soma_seg, t=t, save_to=save_path, title_prefix = 'Soma_')
  calc_input_resistance(soma_seg)


if __name__ == "__main__":
    main()