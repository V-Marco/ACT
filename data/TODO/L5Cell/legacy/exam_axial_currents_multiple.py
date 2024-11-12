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
#list_of_output_folders = [sys.argv[i] for i in range(1, len(sys.argv))] if len(sys.argv) > 1 else [["output/FI_in_vitro_ziao_axon/_seeds_130_90CP_Cell[0]_12nseg_0nbranch_0NCs_0nsyn_-1000", "output/FI_in_vitro2023-10-03_15-21-18/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_-1000"],["output/FI_in_vitro_ziao_axon/_seeds_130_90CP_Cell[0]_12nseg_0nbranch_0NCs_0nsyn_1000", "output/FI_in_vitro2023-10-03_15-21-18/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_1000"]]

list_of_output_folders = [sys.argv[i] for i in range(1, len(sys.argv))] if len(sys.argv) > 1 else [["output/FI_in_vitro2023-10-03_16-08-06/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_-1000", "output/FI_in_vitro2023-10-03_16-08-06/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_0"],["output/FI_in_vitro2023-10-03_16-08-06/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_1000", "output/FI_in_vitro2023-10-03_16-08-06/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_600"]]

import importlib
def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module
#constants = load_constants_from_folder(output_folder)
#
#if 'BenModel' in output_folder:
#  constants.save_every_ms = 3000
#  constants.h_tstop = 3000
#  transpose =True
#else:
#  transpose=False
##  constants.save_every_ms = 200
##  constants.h_tstop = 2500
#dt=constants.h_dt
#
##print(constants.h_dt, constants.save_every_ms, constants.h_tstop)

def plot_for_folder(output_folder, axs, prefix):
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
  try:sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=constants.skip, transpose=transpose)
  except: sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=300, transpose=transpose)
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
  
  #Plot segments adjacent to soma
  #plot_adjacent_segments(segs=soma_segs, sm=sm, title_prefix="Soma_", save_to=save_path)
  #Plot segments adjacent to nexus
  with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
      seg_indexes = pickle.load(file)
  if 'BenModel' in output_folder:
    nexus_seg_index = []
    basal_seg_index = []
  else:
#    nexus_seg_index=seg_indexes["nexus"]
    basal_seg_index=seg_indexes["basal"]
    axon_seg_index=seg_indexes["axon"]
#    tuft_seg_index=seg_indexes["tuft"]
#    logger.log(f"NEXUS SEG: {sm.segments[nexus_seg_index].seg}") # to determine matching seg
#  nexus_segs=[sm.segments[nexus_seg_index]]
  basal_segs=[sm.segments[basal_seg_index]]
  axon_segs=[sm.segments[axon_seg_index]]
  found = False
  for seg in sm.segments:
    if 'axon' in seg.seg:
      #print(seg.seg)
      if '[0](0.5)' in seg.seg:
        axon_seg = seg
        found = True
  if not found:
    for seg in sm.segments:
      if 'axon' in seg.seg:
        #print(seg.seg)
        if '(0.5)' in seg.seg:
          axon_seg = seg
        
  print(f"axon seg index: {sm.segments.index(axon_seg)}")
#  tuft_segs=[sm.segments[tuft_seg_index]]
#  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=save_path)
  #plot_adjacent_segments(segs=basal_segs, sm=sm, title_prefix="Basal_", save_to=save_path)
#  plot_adjacent_segments(segs=tuft_segs, sm=sm, title_prefix="Tuft_", save_to=save_path)
       
  
#  #Plot Axial Currents
#  for seg in soma_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix ='Soma_')
#  for seg in [axon_seg]:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Axon_')
#  for seg in nexus_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Nexus_')
#  for seg in basal_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Basal_')
      
  segments_to_plot = {
      "Soma_": soma_segs,
      #"Nexus_": nexus_segs,
      #"Basal_": basal_segs,
      #"Tuft_": tuft_segs
      "Axon_": [axon_seg]
  }

  #plot_whole_data_length=False
  #if plot_whole_data_length:
  #  for prefix, segments in segments_to_plot.items():
  #      for seg in segments:
  #          plot_all(seg, t, save_to=save_path, title_prefix=prefix)
  #print(segments_to_plot[prefix])
  for seg in segments_to_plot[prefix]:
    #axs = axs[i]  # Use the corresponding column of axes
    plot_all(axs, seg, t, save_to=save_path, title_prefix=prefix)
    #print(f"axs passed to plot_all: {axs}")


  def subset_data(t, xlim):
      indices = np.where((t >= xlim[0]) & (t <= xlim[1]))
      return indices[0]
  #print('number of spikes:',len(sm.soma_spiketimes))

def plot_all(axs, segment, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None):
    '''
    Plots axial current from target segment to adjacent segments, unless it the target segment is soma.
    Plots Vm of segment and adjacent segments,
    Plots Currents of segment
    
    Segment: SegmentManager Segment object
    t: time vector
    indices: indices for subsetting data
    xlim: limits for x axis (used to zoom in on AP)
    ylim: limits for y axis (used to zoom in on currents that may be minimized by larger magnitude currents)
    index: Used to label spike index of soma_spiketimes
    '''
    if indices is not None:
        t = t[indices]
        vlines = vlines[np.isin(np.round(vlines,1), np.round(t,1))]
        #print("t:",t)
        #print("vlines:", vlines)
        
    titles = [
        'Axial Current from [{}] to adjacent segments',
        'Vm from [{}] and adjacent segments',
        'Currents from [{}]'
    ]

    if index:
        for i, title in enumerate(titles):
            titles[i] = 'Spike ' + str(int(index)) + ' ' + title
            
    ylabels = ['nA', 'mV', 'nA']
    data_types = ['axial_currents', 'v', ['ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas']]#['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]]

    #fig, folder_axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
    #print(f"length of axs:{len(axs)}")
    for j, ax in enumerate(axs):
        #print(j)
        title = titles[j].format(segment.name)
        ylabel = ylabels[j]
        data_type = data_types[j]

        if type(data_type) == list: # membrane current plots
            for current in data_type:
                if '+' in current:
                    currents_to_sum = current.split('+')
                    max_index = np.max(indices)
                    array_length = len(getattr(segment, currents_to_sum[0]))
#                    print(np.shape(indices))
#                    print(max_index)
#                    print(array_length)
                    if max_index >= array_length:
                        print(f"Error: Trying to access index {max_index} in an array of size {array_length}")
                    indices = [i for i in indices if i < array_length]
                    data = getattr(segment, currents_to_sum[0])[indices] if indices is not None else getattr(segment, currents_to_sum[0])
                    for current_to_sum in currents_to_sum[1:]:
                        data += getattr(segment, current_to_sum)[indices] if indices is not None else getattr(segment, current_to_sum)
                else:
                    data = getattr(segment, current)[indices] if indices is not None else getattr(segment, current)
                
                ax.plot(t, data, label=current)
        elif data_type == 'v': # Voltage plots
            v_data = segment.v[indices] if indices is not None else segment.v
            ax.plot(t, v_data, color=segment.color, label=segment.name)
            
            for adj_seg in segment.adj_segs:
                adj_v_data = adj_seg.v[indices] if indices is not None else adj_seg.v
                
                if adj_seg.color == segment.color:
                    ax.plot(t, adj_v_data, label=adj_seg.name, color='Magenta')
                else:
                    ax.plot(t, adj_v_data, label=adj_seg.name, color=adj_seg.color)
        elif data_type == 'axial_currents':
            # For  axial currents 'Axial Current from [{}]'
            total_AC = np.zeros(len(segment.v))
            total_dend_AC = np.zeros(len(segment.v))
            total_to_soma_AC = np.zeros(len(segment.v))
            total_away_soma_AC = np.zeros(len(segment.v))
            for adj_seg_index, adj_seg in enumerate(segment.adj_segs): # gather axial currents
                total_AC += segment.axial_currents[adj_seg_index] # all dendrites
                if segment.type == 'soma': # plotting soma's ACs
                  if adj_seg.type == 'dend': # sum basal currents
                    total_dend_AC += segment.axial_currents[adj_seg_index] # sum AC from basal dendrites
                  else: # plot axon & apical trunk ACs
                    axial_current = segment.axial_currents[adj_seg_index][indices] if indices is not None else segment.axial_currents[adj_seg_index]
                    ax.plot(t, axial_current, label=adj_seg.name, color=adj_seg.color) # apical, axon
                else: # plotting any other segment's ACs, sum axial currents to or away soma.
                  if adj_seg in segment.parent_segs: # parent segs will be closer to soma with our model.
                    total_to_soma_AC += segment.axial_currents[adj_seg_index]
                  else:
                    total_away_soma_AC += segment.axial_currents[adj_seg_index]
                  
            if segment.type=='soma': # if we are plotting for soma segment, sum basal axial currents
              basal_axial_current = total_dend_AC[indices] if indices is not None else total_dend_AC
              ax.plot(t, basal_axial_current, label = 'Summed axial currents from basal segments to soma', color = 'red')
              #ax.set_ylim([-2,2])
            else: #if not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
              total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
              ax.set_ylim([-0.75,0.75])
            total_AC = total_AC[indices] if indices is not None else total_AC
            ax.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta')
        else:
          raise(ValueError("Cannot analyze {data_type}"))

        if vlines is not None: # indicate action potentials via dashed vertical lines
            if j==0: # only the axial currents plot
              for ap_index,vline in enumerate(vlines):
                if ap_index == 0: # only label one so that legend is not outrageous
                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', label='AP time', linestyle='dashed')
                else:
                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dashed')
        ax.axhline(0, color='grey')
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('time (ms)')
        ax.legend(loc='upper right')
        if title_prefix:
          ax.set_title(title_prefix+title)
        else:
          ax.set_title(title)
            
    plt.tight_layout()

#    if save_to:
#        if title_prefix:
#            fig.savefig(os.path.join(save_to, title_prefix + "AP_" + "_{}".format(index) + ".png"))
#        else:
#            fig.savefig(os.path.join(save_to, "AP_" + "_{}".format(index) + ".png"))
#    plt.close()

def main():
    for i,output_folders in enumerate(list_of_output_folders):
      current_injection_amp = int(output_folders[0].split('_')[-1]) / 1000
      total_folders = len(output_folders)
      num_titles=3
      for prefix in ("Soma_", "Axon_"):
        fig, master_axs = plt.subplots(num_titles, total_folders, figsize=(12.8, 4.8 * total_folders))
        #print(f"master_axs: {master_axs}")
        
        for col, output_folder in enumerate(output_folders):
            plot_for_folder(output_folder, master_axs[:, col], prefix)
    
        plt.tight_layout()
        #plt.show()
        print(f"finish AC_{current_injection_amp}_{prefix}.png")
        fig.savefig(f"AC_{current_injection_amp}_{prefix}.png")
        fig.savefig(f"{output_folders[1]}\AC_{current_injection_amp}_{prefix}.png")

# try to use the same scale for plots in rows (currently think it does columns?)      
#def main():
#    total_folders = len(output_folders)
#    num_titles = 3
#    for prefix in ("Soma_", "Axon_"):
#        fig, master_axs = plt.subplots(num_titles, total_folders, figsize=(12.8, 4.8 * total_folders))
#        
#        for col, output_folder in enumerate(output_folders):
#            plot_for_folder(output_folder, master_axs[:, col], prefix)
#            
#            # Get the ylim of the first axis in the column
#            ylim = master_axs[0, col].get_ylim()
#            # Set this ylim to all axes in this column
#            for row in range(num_titles):
#                master_axs[row, col].set_ylim(ylim)
#        
#        plt.tight_layout()
#        plt.show()



if __name__ == "__main__":
    main()