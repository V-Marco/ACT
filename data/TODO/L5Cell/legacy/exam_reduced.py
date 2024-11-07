import sys
sys.path.append("../")

from collections import defaultdict
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
import importlib

from Modules.logger import Logger
from Modules.plotting_utils import plot_adjacent_segments
from Modules.segment import SegmentManager

output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/2023-08-23_13-06-55_seeds_130_90L5PCtemplate[0]_196nseg_108nbranch_16073NCs_16073nsyn" #"output/BenModel/"

def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module

def subset_data(t, xlim):
    indices = np.where((t >= xlim[0]) & (t <= xlim[1]))
    return indices[0]

def plot_single_axial_current(ax, t, axial_current, label, color):
    ax.plot(t, axial_current, label=label, color=color)

def sum_across_indices(array, indices):
    return array[indices] if indices is not None else array

def plot_axial_currents(ax, t, segments, indices):
    n = len(segment.v)
    for segment in segments:
        total_AC = np.zeros(n)
        total_dend_AC = np.zeros(n) if segment.type == 'soma' else None
        total_to_soma_AC = np.zeros(n) if segment.type != 'soma' else None
        total_away_soma_AC = np.zeros(n) if segment.type != 'soma' else None
        
        for adj_seg_index, adj_seg in enumerate(segment.adj_segs):  # Gather axial currents
            current_ac = segment.axial_currents[adj_seg_index]
            total_AC += current_ac
            
            if segment.type == 'soma':  # Plotting soma's ACs
                if adj_seg.type == 'dend':  # Sum basal currents
                    total_dend_AC += current_ac
                else:  # Plot axon & apical trunk ACs
                    plot_single_axial_current(ax, t, sum_across_indices(current_ac, indices), adj_seg.name, adj_seg.color)
    
            else:  # Plotting any other segment's ACs, sum axial currents to or away from the soma.
                target_array = total_to_soma_AC if adj_seg in segment.parent_segs else total_away_soma_AC
                target_array += current_ac
    
        if segment.type == 'soma':
            plot_single_axial_current(ax, t, sum_across_indices(total_dend_AC, indices), 'Summed axial currents from basal segments to soma', 'red')
            ax.set_ylim([-2, 2])
            
        else:
            plot_single_axial_current(ax, t, sum_across_indices(total_to_soma_AC, indices), 'Summed axial currents to segments toward soma', 'blue')
            plot_single_axial_current(ax, t, sum_across_indices(total_away_soma_AC, indices), 'Summed axial currents to segments away from soma', 'red')
            ax.set_ylim([-0.75, 0.75])

    # Calculate and plot averages
    avg_dend_AC = np.mean([seg.total_dend_AC for seg in segments if seg.type == 'soma'], axis=0)
    avg_to_soma_AC = np.mean([seg.total_to_soma_AC for seg in segments if seg.type != 'soma'], axis=0)
    avg_away_soma_AC = np.mean([seg.total_away_soma_AC for seg in segments if seg.type != 'soma'], axis=0)
    
    if segments[0].type == 'soma':
        plot_single_axial_current(ax, t, sum_across_indices(avg_dend_AC, indices), 'Avg. Axial Currents from basal to soma', 'purple')
    else:
        plot_single_axial_current(ax, t, sum_across_indices(avg_to_soma_AC, indices), 'Avg. Axial Currents to segments toward soma', 'purple')
        plot_single_axial_current(ax, t, sum_across_indices(avg_away_soma_AC, indices), 'Avg. Axial Currents to segments away from soma', 'orange')

def plot_voltages(ax, t, segments, indices):
    avg_voltage = np.mean([seg.v for seg in segments], axis=0)
    plot_single_axial_current(ax, t, sum_across_indices(avg_voltage, indices), 'Average Voltage', 'purple')
    
    for segment in segments:
      v_data = segment.v[indices] if indices is not None else segment.v
      ax.plot(t, v_data, color=segment.color, label=segment.name)
      
      for adj_seg in segment.adj_segs:
          adj_v_data = adj_seg.v[indices] if indices is not None else adj_seg.v
          
          if adj_seg.color == segment.color:
              ax.plot(t, adj_v_data, label=adj_seg.name, color='Magenta')
          else:
              ax.plot(t, adj_v_data, label=adj_seg.name, color=adj_seg.color)

def plot_currents(ax, t, segments, indices, data_type):
    avg_currents = {}
    
    for segment in segments:
       for current in data_type:
            if '+' in current:
                currents_to_sum = current.split('+')
                max_index = np.max(indices)
                array_length = len(getattr(segment, currents_to_sum[0]))
                if max_index >= array_length:
                    print(f"Error: Trying to access index {max_index} in an array of size {array_length}")
                indices = [i for i in indices if i < array_length]
                data = getattr(segment, currents_to_sum[0])[indices] if indices is not None else getattr(segment, currents_to_sum[0])
                for current_to_sum in currents_to_sum[1:]:
                    data += getattr(segment, current_to_sum)[indices] if indices is not None else getattr(segment, current_to_sum)
            else:
                data = getattr(segment, current)[indices] if indices is not None else getattr(segment, current)
            
            ax.plot(t, data, label=current)
    
    # Calculate and plot averages for each current type
    for current_type, current_values in avg_currents.items():
        avg_current = np.mean(current_values, axis=0)
        ax.plot(t, sum_across_indices(avg_current, indices), label=f"Avg. {current_type}")

    
def plot_all(segments, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None):
    '''
    Plots several types of data for a given segment over time.
    :param segments: List of SegmentManager Segment objects
    :param t: Time vector
    :param indices: Indices for subsetting data
    :param xlim: X-axis limits
    :param ylim: Y-axis limits
    :param index: Used to label spike index of soma_spiketimes
    :param save_to: Path to save plot
    :param title_prefix: Title prefix for plot
    :param vlines: Vertical lines for plot
    '''
    if indices is not None:
        t = t[indices]
        vlines = vlines[np.isin(np.round(vlines, 1), np.round(t, 1))]

    titles = [
        'Axial Current from [{}] to adjacent segments',
        'Vm from [{}] and adjacent segments',
        'Currents from [{}]'
    ]
    if index:
        titles = [f'Spike {int(index)} {title}' for title in titles]
        
    ylabels = ['nA', 'mV', 'nA']
    currents_to_plot = ['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]
    
    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))

    for j, (ax, title, ylabel) in enumerate(zip(axs, titles, ylabels)):
        if j == 0:
            plot_axial_currents(ax, t, segments, indices)
        elif j == 1:
            plot_voltages(ax, t, segments, indices)
        elif j == 2:
            plot_currents(ax, t, segments, indices, currents_to_plot)
        
        title = title.format(segment.name)
        if title_prefix:
            title = title_prefix + title
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (ms)')
        
        if xlim:
            ax.set_xlim(xlim)
        
        if vlines is not None:
            for vline in vlines:
                ax.axvline(x=vline, color='k', linestyle='--')
        
        ax.legend()

    plt.tight_layout()

    if save_to:
        filename = f"AP_{index}.png" if title_prefix is None else f"{title_prefix}AP_{index}.png"
        fig.savefig(os.path.join(save_to, filename))
        
    plt.close()

def main():

  constants = load_constants_from_folder(output_folder)
  
  if 'BenModel' in output_folder:
    constants.save_every_ms = 3000
    constants.h_tstop = 3000
    transpose =True
  else:
    transpose=False
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

  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=300, transpose=transpose)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  
  #Compute axial currents from each segment toward its adjacent segments.
  #compute axial currents between all segments
  sm.compute_axial_currents()
  
  logger.log(f"soma_spiketimes: {sm.soma_spiketimes}")
  
  logger.log(f'firing_rate: {len(sm.soma_spiketimes) / (len(sm.segments[0].v) * dt / 1000)}') # number of spikes / seconds of simulation
  
  #Find soma segments and group tuft segments.
  apical_segs_by_distance = defaultdict(list)  # defaultdict will automatically create lists for new keys
  soma_segs = []
  for seg in sm.segments:
    if seg.type == 'soma':
      soma_segs.append(seg)
    elif seg.type == 'apic':
      apical_segs_by_distance[seg.h_distance].append(seg)
      
  if len(soma_segs) != 1:
    logger.log(f"Picking 1 out of {len(soma_segs)} Soma segments.")
    #raise(ValueError("There should be only one soma segment."))
    soma_segs=[soma_segs[3]]
  
  #Plot segments adjacent to soma
  plot_adjacent_segments(segs=soma_segs, sm=sm, title_prefix="Soma_", save_to=save_path)
  #Plot segments adjacent to nexus
  with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
      seg_indexes = pickle.load(file)
  if 'BenModel' in output_folder:
    nexus_seg_index = []
    basal_seg_index = []
  else:
    nexus_seg_index=seg_indexes["nexus"]
    basal_seg_index=seg_indexes["basal"]
    tuft_seg_index=seg_indexes["tuft"]
    logger.log(f"NEXUS SEG: {sm.segments[nexus_seg_index].seg}") # to determine matching seg
  nexus_segs=[sm.segments[nexus_seg_index]]
  basal_segs=[sm.segments[basal_seg_index]]
  tuft_segs=[sm.segments[tuft_seg_index]]
  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=save_path)
  plot_adjacent_segments(segs=basal_segs, sm=sm, title_prefix="Basal_", save_to=save_path)
  plot_adjacent_segments(segs=tuft_segs, sm=sm, title_prefix="Tuft_", save_to=save_path)
       
  
#  #Plot entire trace
#  for seg in soma_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix ='Soma_')
#  for seg in nexus_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Nexus_')
#  for seg in basal_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Basal_')
      
  segments_to_plot = {
      "Soma_": soma_segs,
      "Nexus_": nexus_segs,
      "Basal_": basal_segs,
      "Tuft_": tuft_segs
  }

  plot_whole_data_length=False
  if plot_whole_data_length:
    for prefix, segments in segments_to_plot.items():
        for seg in segments:
            plot_all(seg, t, save_to=save_path, title_prefix=prefix)
  
  # Plot around APs
  for i, AP_time in enumerate(np.array(sm.soma_spiketimes)):  # spike time (ms) 
      before_AP = AP_time - 100  # ms
      after_AP = AP_time + 100  # ms
      xlim = [before_AP, after_AP]  # time range
    
      # Subset the data for the time range
      indices = subset_data(t, xlim)

      for prefix, segments in segments_to_plot.items():
          for seg in segments:
              plot_all(segmenst=[seg], t=t, indices=indices, index=i+1, save_to=save_path, title_prefix=prefix, ylim=[-1, 1] if prefix == "Nexus_" else None, vlines=np.array(sm.soma_spiketimes))
              
              # Plot apical segments grouped by distance
          for h_distance, apical_segs in sorted(apical_segs_by_distance.items()):
              print(f"Plotting for h_distance: {h_distance}")
              title_prefix = f"Apical Segments at h_distance {h_distance} during AP {i+1} - "
              
              plot_all(segments=apical_segs, t=t, indices=indices, index=i+1, title_prefix=title_prefix)


if __name__ == "__main__":
    main()