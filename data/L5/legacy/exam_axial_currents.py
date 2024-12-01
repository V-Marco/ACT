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

PT_Cell=False # true: Neymotin cell; false: Neymotin_Hay
if PT_Cell:
  output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neymotin/2023-10-12_21-10-22_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_300"
else:
  output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neymotin_Hay4/_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_300/"

#FI_Neymotin/2023-10-12_21-10-22_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_300"
#FI_Neymotin_Hay2/_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_300/" 
#"output/BenModel/"

plot_APs = True # Create a zoomed in plot around every AP.
plot_CA_NMDA = False # used to plot the trace from segments that have Ca or NMDA spikes
process_ca_nmda_inds = False # used to reduce segments_for_condition from exam_nmda.py to a list of unique segments
print_steady_state_values = True

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

def create_segment_types(soma):
    """
    Create a dictionary of segment types based on the segments that are adjacent to the soma
    and includes the soma itself.

    Parameters:
    - soma: The soma segment object, expected to have a property 'adj_segs' listing all its adjacent segments.

    Returns:
    A dictionary with keys being segment type names (inferred from segment names) and values being lists of segments.
    """
    segment_types = {"Soma_": [soma]}  # Initialize with the soma
    
    # Iterate over each adjacent segment
    for adj_seg in soma.adj_segs:
        # Use the segment's type as key (assuming there's a 'type' property in the segment object)
        seg_type = adj_seg.type + "_"
        
        # If this type hasn't been added to the dictionary, initialize it with an empty list
        if seg_type not in segment_types:
            segment_types[seg_type] = []
        
        # Append the segment to its corresponding type
        segment_types[seg_type].append(adj_seg)
    
    return segment_types



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
    axon_seg_index=seg_indexes["axon"]
    tuft_seg_index=seg_indexes["tuft"]
#    logger.log(f"NEXUS SEG: {sm.segments[nexus_seg_index].seg}") # to determine matching seg
  nexus_segs=[sm.segments[nexus_seg_index]]
  basal_segs=[sm.segments[basal_seg_index]]
  axon_segs=[sm.segments[axon_seg_index]]
  found = False
# get axon segment
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
# get nexus segment
  found = False
  if constants.build_cell_reports_cell:
    for seg in sm.segments:
      if 'apic' in seg.seg:
        #print(seg.seg)
        if '[24]' in seg.seg:
          nexus_seg = seg
          found = True
#    if not found:
#      for seg in sm.segments:
#        if 'axon' in seg.seg:
#          #print(seg.seg)
#          if '(0.5)' in seg.seg:
#            axon_seg = seg

  # taken from exam_NMDA
  ca_inds=[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 159, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
  nmda_inds= [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
  if process_ca_nmda_inds:
    ca_inds = list(np.unique(ca_inds))
    nmda_inds = list(np.unique(nmda_inds))
    print(f"ca_inds: {ca_inds}")
    print(f"nmda_inds: {nmda_inds}")
  #nexus_segs=[nexus_seg]
  tuft_segs=[sm.segments[tuft_seg_index]]
  ca_segs=[sm.segments[ca_ind] for ca_ind in ca_inds]
  nmda_segs=[sm.segments[nmda_ind] for nmda_ind in nmda_inds]
  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=save_path)
  plot_adjacent_segments(segs=basal_segs, sm=sm, title_prefix="Basal_", save_to=save_path)
  plot_adjacent_segments(segs=tuft_segs, sm=sm, title_prefix="Tuft_", save_to=save_path)
  if plot_CA_NMDA:
    plot_adjacent_segments(segs=ca_segs, sm=sm, title_prefix="CA_", save_to=save_path) # segment with calcium spike
    plot_adjacent_segments(segs=nmda_segs, sm=sm, title_prefix="NMDA_", save_to=save_path) # segment with NMDA spike
       

  #  desired_real_time is the time at which you want to print the current values
  desired_real_time = 8000  # the desired time in ms
  steady_state_index = int(desired_real_time / constants.h_dt)
  
#  # Loop over all segment types and call print_steady_state_values
#  segment_types = {
#      'Soma_': soma_segs,
#      'Axon_': [axon_seg],  # Making it a list to be consistent with the loop
#      'Nexus_': nexus_segs,
#      'Basal_': basal_segs
#  }
  segment_types = create_segment_types(soma_segs[0])
  if print_steady_state_values:
        # Filter segment_types to only include Soma and Axon
    filtered_segment_types = {k: v for k, v in segment_types.items() if k in ["Soma_", "axon_"]}
    
    # Loop over the filtered segment types and call print_steady_state_values
    for title_prefix, segments in filtered_segment_types.items():
        for seg in segments:
            print_steady_state_values(seg, t, steady_state_index, title_prefix=title_prefix)
  return
  
#  #Plot Axial Currents
  for seg in soma_segs:
      plot_all(seg, t, save_to=save_path, title_prefix ='Soma_')
  for seg in [axon_seg]:
      plot_all(seg, t, save_to=save_path, title_prefix = 'Axon_')
  for seg in nexus_segs:
      plot_all(seg, t, save_to=save_path, title_prefix = 'Nexus_')
  for seg in basal_segs:
      plot_all(seg, t, save_to=save_path, title_prefix = 'Basal_')
  if plot_CA_NMDA:
      for seg in ca_segs:
          plot_all(seg, t, save_to=save_path, title_prefix = 'CA_')
      for seg in nmda_segs:
          plot_all(seg, t, save_to=save_path, title_prefix = 'NMDA_')
      
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


  def subset_data(t, xlim):
      indices = np.where((t >= xlim[0]) & (t <= xlim[1]))
      return indices[0]
  print('number of spikes:',len(sm.soma_spiketimes))
  # Plot around APs
  if plot_APs:
      for i, AP_time in enumerate(np.array(sm.soma_spiketimes)):  # spike time (ms) 
          before_AP = AP_time - 100  # ms
          after_AP = AP_time + 100  # ms
          xlim = [before_AP, after_AP]  # time range
        
          # Subset the data for the time range
          indices = subset_data(t, xlim)
    
          for prefix, segments in segments_to_plot.items():
              for seg in segments:
                  plot_all(segment=seg, t=t, indices=indices, index=i+1, save_to=save_path, title_prefix=prefix, ylim=[-1, 1] if prefix == "Nexus_" else None, vlines=np.array(sm.soma_spiketimes))
              
def plot_all(segment, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None, plot_adj_Vm=True):
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
    data_types = ['axial_currents', 'v', ['ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas', 'ica', 'iampa','inmda','igaba']]#['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]]

    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
    
    for j, ax in enumerate(axs):
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
                if np.shape(t) != np.shape(data):
                  print(np.shape(t), np.shape(data))
                  ax.plot(t[:-1], data, label=current)
                else:
                  ax.plot(t, data, label=current)
                #ax.set_ylim([-0.1,0.1])
        elif data_type == 'v': # Voltage plots
            v_data = segment.v[indices] if indices is not None else segment.v
            ax.plot(t, v_data, color=segment.color, label=segment.name)
            if plot_adj_Vm:
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
              #ax.set_ylim([-0.2,0.1])
            else: #if not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
              total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
              #ax.set_ylim([-0.75,0.75])
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

    if save_to:
        if title_prefix:
            fig.savefig(os.path.join(save_to, title_prefix + "AP_" + "_{}".format(index) + ".png"))
        else:
            fig.savefig(os.path.join(save_to, "AP_" + "_{}".format(index) + ".png"))
    plt.close()

def print_steady_state_values(segment, t, steady_state_time_index, title_prefix=None):
    '''
    Print the steady state values of currents and axial currents for a given segment at a specific time index.
    
    Segment: SegmentManager Segment object
    t: time vector
    steady_state_time_index: Index at which the steady state values should be printed
    title_prefix: Prefix for the title (typically denotes the segment)
    '''

    # Current types present in the segment
    data_types = ['v','ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas', 'ica', 'iampa','inmda','igaba']

    # Print title
    if title_prefix:
        print(f"{title_prefix} - Steady State Values at time {t[steady_state_time_index]}ms:")
    else:
        print(f"Steady State Values at time {t[steady_state_time_index]}ms:")

    for current in data_types:
        data = getattr(segment, current)
        if current == 'v':
          units = 'mV'
        else:
          units = 'nA'
        print(f"{current}: {data[steady_state_time_index]} {units}")

    # If there are axial currents
    if hasattr(segment, 'axial_currents'):
        axial_current_by_type = {}  # Store summed axial currents by type
        for idx, adj_seg in enumerate(segment.adj_segs):
            axial_current_value = segment.axial_currents[idx][steady_state_time_index]
            print(f"Axial current from {segment.name} to {adj_seg.name} (Type: {adj_seg.type}): {axial_current_value} nA")

            # Sum the axial currents by type
            if adj_seg.type in axial_current_by_type:
                axial_current_by_type[adj_seg.type] += axial_current_value
            else:
                axial_current_by_type[adj_seg.type] = axial_current_value

        for seg_type, axial_current_sum in axial_current_by_type.items():
            print(f"Total axial current to {seg_type} type segments: {axial_current_sum} nA")

        total_AC = sum(axial_current_by_type.values())
        print(f"Total summed axial currents: {total_AC} nA")

    print("\n")  # For readability

# Example usage
# print_steady_state_values(segment_object, t, 100, title_prefix="Segment 1")

  

#def plot_all(segment, t, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines = None):
#    '''
#    Plots axial current from segment to adjacent segments,
#    Plots Vm of segment and adjacent segments,
#    Plots Currents of segment
#    
#    Segment: SegmentManager Segment object
#    t: time vector
#    xlim: limits for x axis (used to zoom in on AP)
#    ylim: limits for y axis (used to zoom in on currents that may be minimized by larger magnitude currents)
#    index: Used to label spike index of soma_spiketimes
#    '''
#    titles = [
#        'Axial Current from [{}] to adjacent segments',
#        'Vm from [{}] and adjacent segments',
#        'Currents from [{}]'
#    ]
#    if index:
#      for i,title in enumerate(titles):
#        titles[i] = 'Spike ' + str(int(index)) + ' ' + title
#    ylabels = ['nA', 'mV', 'nA']
#    data_types = ['axial_currents', 'v', ['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]] # can adjust this list to visualize a specific current
#
#    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
#
#    for j, ax in enumerate(axs):
#        title = titles[j].format(segment.name)
#        ylabel = ylabels[j]
#        data_type = data_types[j]
#
#        if type(data_type) == list: # For 'Currents from [{}]'
#            for current in data_type:
#                if '+' in current:
#                  currents_to_sum = current.split('+')
#                  data=getattr(segment,currents_to_sum[0])
#                  for current_to_sum in currents_to_sum[1:]:
#                    data+=getattr(segment,current_to_sum)
#                else:
#                  data = getattr(segment, current)
#                t=np.arange(0,len(data)*dt,dt)
#                ax.plot(t, data, label = current)
#            #if ylim is None:
#            #    ax.set_ylim([min(data), max(data)])
#        elif data_type == 'v': # For 'Vm from [{}]'
#            t=np.arange(0,len(segment.v)*dt,dt)
#            ax.plot(t, segment.v, color = segment.color, label = segment.name)
#            for i, adj_seg in enumerate(segment.adj_segs):
#                if adj_seg.color == segment.color:
#                    ax.plot(t, adj_seg.v, label = adj_seg.name, color = 'Magenta')
#                else:
#                    ax.plot(t, adj_seg.v, label = adj_seg.name, color = adj_seg.color)
#            if ylim is None:
#                ax.set_ylim([min(segment.v), max(segment.v)])
#        else: # For 'Axial Current from [{}]'
#            total_AC = np.zeros(len(segment.v))
#            total_dend_AC = np.zeros(len(segment.v))
#            total_to_soma_AC = np.zeros(len(segment.v))
#            total_away_soma_AC = np.zeros(len(segment.v))
#            for i, adj_seg in enumerate(segment.adj_segs):
#                total_AC += segment.axial_currents[i] # all dendrites
#                if adj_seg in segment.parent_segs: # if the adjacent segment is closer to soma
#                  total_to_soma_AC += segment.axial_currents[i]
#                else:
#                  total_away_soma_AC += segment.axial_currents[i]
#                if adj_seg.type == 'dend':
#                  basals=True
#                  total_dend_AC += segment.axial_currents[i] # basal dendrites
#                elif segment.type == 'soma':
#                  ax.plot(t, segment.axial_currents[i], label = adj_seg.name, color = adj_seg.color) # apical, axon
#            if segment.type=='soma':
#              ax.plot(t, total_dend_AC, label = 'Summed basal axial currents', color = 'red')
#              ax.set_ylim([-2,2])
#            else:
#              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
#              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
#              ax.set_ylim([-0.75,0.75])
#            ax.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta')
#
#        if vlines is not None:
#            if j==0: # only the axial currents plot
#              for ap_index,vline in enumerate(vlines):
#                if ap_index == 0: # only label one so that legend is not outrageous
#                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', label='AP time', linestyle='dashed')
#                else:
#                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dashed')
#        ax.axhline(0, color='grey')
#        if xlim:
#            ax.set_xlim(xlim)
#        ax.set_ylabel(ylabel)
#        ax.set_xlabel('time (ms)')
#        ax.legend(loc='upper right')
#        if title_prefix:
#          ax.set_title(title_prefix+title)
#        else:
#          ax.set_title(title)
#
#    plt.tight_layout()
#
#    if save_to:
#        if title_prefix:
#          fig.savefig(os.path.join(save_to, title_prefix + "AP_" + "_{}".format(index) + ".png"))
#        else:
#          fig.savefig(os.path.join(save_to, "AP_" + "_{}".format(index) + ".png"))
#          
#    plt.close()



if __name__ == "__main__":
    main()