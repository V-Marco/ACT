# Recently updated from reading csv to reading pickles
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from cell_inference.config import params
from Modules.segment import Segment

from Modules.logger import Logger

output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/default_path"

import importlib
def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module
constants = load_constants_from_folder(output_folder)

constants.show_electrodes = False
if constants.show_electrodes:
  elec_pos = params.ELECTRODE_POSITION
else:
  elec_pos = None

# Default view
elev, azim = 90, -90#
  
# Set up the plotting_modes and cluster_types to iterate over
plotting_modes = ['functional_groups', 'presynaptic_cells']
cluster_types = ['exc', 'inh_distributed', 'inh_soma']  

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def plot_spike_rasters(functional_groups, title_prefix=None, save_to=None): # can be moved to analysis script
    fig = plt.figure(figsize=(15, 10))

    # Initialize counter for y-axis label (for each neuron)
    y_count = 0

    # Generate a list of distinct colors based on the number of functional groups
    colors = plt.cm.viridis(np.linspace(0, 1, len(functional_groups)))

    for color, functional_group in zip(colors, functional_groups):
        for cell_index, presynaptic_cell in enumerate(functional_group.presynaptic_cells):
            y_count += 1
            spike_train = presynaptic_cell.spike_train
            
            # Generate a shade of the group color based on the cell index
            cell_color = mcolors.to_rgba(color, alpha=(cell_index + 1) / len(functional_group.presynaptic_cells))

            if len(spike_train) > 0:  # Ensure that there are spikes to plot
                for spikes in spike_train:
                    plt.scatter(spikes, [y_count] * len(spikes), color=cell_color, marker='|')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Spike Raster Plot')
    if save_to:
        filename = f"Spikes.png" if title_prefix is None else f"{title_prefix}_spikes.png"
        fig.savefig(os.path.join(save_to, filename))    
    plt.close()

def exam_inc_spikes(functional_groups, tstop, prefix=None, save_to=None): # can be moved to analysis script
    '''
    Calculates the presynaptic cell firing rates distribution from generated spike trains.
    tstop (ms)
    '''
    spike_trains = []
    firing_rates = []
    
    for func_grp in functional_groups:
        for pre_cell in func_grp.presynaptic_cells:
            spike_train = pre_cell.spike_train
            for syn in pre_cell.synapses:
                spike_trains.append(spike_train)
                
    for spike_train in spike_trains:
        firing_rate = len(spike_train) / (tstop / 1000)
        firing_rates.append(firing_rate)
        
    mean_fr = np.mean(firing_rates)
    std_fr = np.std(firing_rates)
    
    # Plot and save histogram
    plt.hist(firing_rates, bins=30, alpha=0.75, label='Firing Rate Distribution')
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Frequency')
    plt.title('Firing Rate Distribution')
    plt.legend()
    
    if save_to:
        # Make sure the directory exists
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        
        # Save the histogram
        plt.savefig(os.path.join(save_to, f"{prefix}_firing_rate_histogram.png"))
        
        # Save firing rates to a numpy binary file
        #np.save(os.path.join(save_to, f"{prefix}_firing_rates.npy"), np.array(firing_rates))
        
        # Save mean and std of firing rates to a text file
        with open(os.path.join(save_to, f"{prefix}_firing_rates_stats.txt"), "w") as f:
            f.write(f"Mean Firing Rate: {mean_fr}\n")
            f.write(f"Standard Deviation of Firing Rate: {std_fr}\n")
            
        # Save firing rates to a CSV file
        #df = pd.DataFrame({'Firing Rates': firing_rates})
        #df.to_csv(os.path.join(save_to, f"{prefix}_firing_rates.csv"), index=False)
        
    #plt.show()  # Show the plot


def load_data(cluster_types, output_folder):
  data = {}
  data["detailed_seg_info"] = pd.read_csv(os.path.join(output_folder, "detailed_seg_info.csv"))
  data["functional_groups"] = {}
  data["presynaptic_cells"] = {}
  
  for cluster_type in cluster_types:
      data["functional_groups"][cluster_type] = pd.read_pickle(os.path.join(output_folder, cluster_type + "_functional_groups.pkl"))
      data["presynaptic_cells"][cluster_type] = pd.read_pickle(os.path.join(output_folder, cluster_type + "_presynaptic_cells.pkl"))
  
  return data

def reset_segment_assignments(segments):
    """Reset the assignment attributes of all segments."""
    for seg in segments:
        seg.functional_group_names = []
        seg.functional_group_indices = []
        seg.presynaptic_cell_names = []
        seg.presynaptic_cell_indices = []

def assign_funcgroups_and_precells_to_segments(cluster_type, plotting_mode, segments, data, logger):
    """Assign FuncGroups and PreCells to segments based on the given cluster_type."""
    max_dists = [] # list for cluster spans
    pc_mean_firing_rates = None
    all_num_synapses = []
    if plotting_mode == 'functional_groups':
      # Iterate over functional groups and assign them to segments
      logger.log_section_start("Iterating through Functional Groups")
      for _, row in data["functional_groups"][cluster_type].iterrows():
          fg_name = row["name"]
          fg_index = row["functional_group_index"]
          num_synapses=row['num_synapses']
          cleaned_strings = row["target_segment_indices"].replace('[', '').replace(']', '').split(',')
          target_indices = [int(x) for x in cleaned_strings]
          #max_dists.append(max_distance_for_segments(segments))
          all_num_synapses.append(num_synapses)
          for target_seg_index in target_indices:
              seg = segments[target_seg_index]
              seg.functional_group_names.append(fg_name)
              seg.functional_group_indices.append(fg_index)
      logger.log_section_end("Iterating through Functional Groups")
    # For presynaptic cells
    elif plotting_mode == 'presynaptic_cells':
      # Iterate over presynaptic cells and assign them to segments
      pc_mean_firing_rates = []
      logger.log_section_start("Iterating through presynaptic cells")
      for _, row in data["presynaptic_cells"][cluster_type].iterrows():
          pc_name = row["name"]
          pc_index = row["presynaptic_cell_index"]
          pc_mean_firing_rate = float(row["mean_firing_rate"])
          num_synapses=row['num_synapses']
          cleaned_strings = row["target_segment_indices"].replace('[', '').replace(']', '').split(',')
          target_indices = [int(x) for x in cleaned_strings]
          #max_dists.append(max_distance_for_segments(segments))
          all_num_synapses.append(num_synapses)
          pc_mean_firing_rates.append(pc_mean_firing_rate)
          for target_seg_index in target_indices:
              seg = segments[target_seg_index]
              seg.presynaptic_cell_names.append(pc_name)
              seg.presynaptic_cell_indices.append(pc_index)
              seg.presynaptic_cell_mean_firing_rate = pc_mean_firing_rate
      logger.log_section_end("Iterating through presynaptic cells")
            
    #mean_distance = np.mean(max_dists)
    #std_distance = np.std(max_dists)
    mean_num_synapses = np.mean(all_num_synapses)
    std_num_synapses = np.std(all_num_synapses)
    if pc_mean_firing_rates is not None: # only for presynaptic cells
      mean_mean_fr = np.mean(pc_mean_firing_rates)
      std_mean_fr = np.std(pc_mean_firing_rates)
    else:
      mean_mean_fr = None
      std_mean_fr = None
    
    return mean_mean_fr, std_mean_fr, mean_num_synapses, std_num_synapses #, mean_distance, std_distance


  # Function to compute pairwise distance between segment endpoints
def pairwise_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + 
                   (point1[1] - point2[1])**2 + 
                   (point1[2] - point2[2])**2)

# Function to compute the maximum distance between two segments
def max_distance_between_segments(seg1, seg2):
    # updated to only check middle of segments because it takes so long and is just to measure cluster span.
    seg1_points = [
        #(seg1.p0_x3d, seg1.p0_y3d, seg1.p0_z3d),
        (seg1.p0_5_x3d, seg1.p0_5_y3d, seg1.p0_5_z3d)
        #(seg1.p1_x3d, seg1.p1_y3d, seg1.p1_z3d)
    ]
    
    seg2_points = [
        #(seg2.p0_x3d, seg2.p0_y3d, seg2.p0_z3d),
        (seg2.p0_5_x3d, seg2.p0_5_y3d, seg2.p0_5_z3d)
        #(seg2.p1_x3d, seg2.p1_y3d, seg2.p1_z3d)
    ]
    
    # compute all pairwise distances and return the maximum
    return max(pairwise_distance(p1, p2) for p1 in seg1_points for p2 in seg2_points)

# find the max distance between clustered segments
def max_distance_for_segments(segments):
    n = len(segments)
    if n <= 1:
        return 0
    max_distance = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = max_distance_between_segments(segments[i], segments[j])
            max_distance = max(max_distance, dist)
    return max_distance
      
def plot_segments(segments, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for segment in segments:
        p0 = (segment.p0_x3d, segment.p0_y3d, segment.p0_z3d)
        p0_5 = (segment.p0_5_x3d, segment.p0_5_y3d, segment.p0_5_z3d)
        p1 = (segment.p1_x3d, segment.p1_y3d, segment.p1_z3d)
        color = segment.color
        ax.plot([p0[0], p0_5[0]], [p0[1], p0_5[1]], [p0[2], p0_5[2]], color=color)
        ax.plot([p0_5[0], p1[0]], [p0_5[1], p1[1]], [p0_5[2], p1[2]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set the y-axis to be vertical
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect([1, 3, 1])  # x, y, z
    plt.savefig(save_name)
    plt.close()


def plot_segments_mean_firing_rate(segments, save_name, mean_value=None, std_value=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Find min and max mean_firing_rate for normalization purposes
    min_rate = min(segment.presynaptic_cell_mean_firing_rate for segment in segments)
    max_rate = max(segment.presynaptic_cell_mean_firing_rate for segment in segments)

    # Create colormap
    cmap = plt.cm.viridis

    for segment in segments:
        p0 = (segment.p0_x3d, segment.p0_y3d, segment.p0_z3d)
        p0_5 = (segment.p0_5_x3d, segment.p0_5_y3d, segment.p0_5_z3d)
        p1 = (segment.p1_x3d, segment.p1_y3d, segment.p1_z3d)
        
        # Normalize mean_firing_rate to [0, 1] and get the color from the colormap
        if max_rate == min_rate:
            norm_val = 0  # or 0.5 or whatever default value you'd like in this scenario
        else:
            norm_val = (segment.presynaptic_cell_mean_firing_rate - min_rate) / (max_rate - min_rate)
        color = cmap(norm_val)

        ax.plot([p0[0], p0_5[0]], [p0[1], p0_5[1]], [p0[2], p0_5[2]], color=color)
        ax.plot([p0_5[0], p1[0]], [p0_5[1], p1[1]], [p0_5[2], p1[2]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect([1, 3, 1])  # x, y, z
    
    # Adding colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_rate, vmax=max_rate))
    cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Mean Firing Rate')
    
    #add mean of mean firing rates to color bar
    if mean_value:
      # Get current ticks from the colorbar
      ticks = cbar.get_ticks().tolist()
      # Add the mean value to the list of ticks
      ticks.append(mean_value)
      if std_value:
        mean_plus_std = mean_value + std_value
        mean_minus_std = mean_value - std_value
        ticks.append(mean_plus_std)
        ticks.append(mean_minus_std)
      ticks.sort()  # Sort the ticks in increasing order
      # Remove the highest tick because for some reason introduces one outside of data range
      ticks = ticks[:-1]
      # Set the updated ticks to the colorbar
      cbar.set_ticks(ticks)
      # Optionally, you can customize the tick labels. For instance, label the mean as 'Mean':
      if std_value:
        labels = {mean_value: 'Mean', mean_plus_std: 'Mean + std', mean_minus_std: 'Mean - std'}
      else:
        labels = {mean_value: 'Mean'}
      #print(ticks)
      #print([labels.get(t, f"{t:.2f}") for t in ticks])
      cbar.set_ticklabels([labels.get(t, f"{t:.2f}") for t in ticks])

    plt.savefig(save_name)
    plt.close()


def main(cluster_types, plotting_modes, output_folder):
  # read detailed seg info and clustering csvs
  data = load_data(cluster_types=cluster_types, output_folder=output_folder)
  save_path = os.path.join(output_folder, "Analysis Clusters")
  if os.path.exists(save_path):
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Directory already exists: {save_path}')
  else:
    os.mkdir(save_path)
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Creating Directory: {save_path}')
  
  num_segments = len(data["detailed_seg_info"])
  logger.log(f'number of detailed segments used for clustering:{num_segments}')
  detailed_segments=[]
  
  with open(os.path.join(save_path, 'info.txt'), 'w') as file:
    pass  # This block is simply to truncate (empty) the file
  
  for i in range(num_segments):
      # Build seg_data
      seg_data = {} # placeholder
      seg = Segment(seg_info = data["detailed_seg_info"].iloc[i], seg_data = {})
      detailed_segments.append(seg)
  
  for cluster_type in cluster_types:
    # Reset segment assignments
    reset_segment_assignments(detailed_segments)
    for plotting_mode in plotting_modes:
      logger.log(f'analyzing {cluster_type} {plotting_mode}')
      # Assign segments to FuncGroups or PreCells based on the current cluster_type and plotting mode
      mean_mean_fr, std_mean_fr, mean_num_synapses, std_num_synapses = assign_funcgroups_and_precells_to_segments(cluster_type, plotting_mode, detailed_segments, data, logger=logger)
      #mean_mean_fr, std_mean_fr, mean_num_synapses, std_num_synapses, mean_distance, std_distance = assign_funcgroups_and_precells_to_segments(cluster_type, plotting_mode, detailed_segments, data, logger=logger)
      with open(os.path.join(save_path, 'info.txt'), 'a') as file:
          print(f"'{cluster_type}' '{plotting_mode}':", file=file)  # Added a newline character after the colon
          print(f"Mean number of synapses: {mean_num_synapses}", file=file)  # Added a newline character after the value
          print(f"Standard deviation of number of synapses: {std_num_synapses}", file=file)
          if mean_mean_fr:
              print(f"Mean of PreCell mean firing rates: {mean_mean_fr}", file=file)
              print(f"Standard deviation of PreCell mean firing rates: {std_mean_fr}", file=file)
    
      # make color maps for clustering assignment
      num_groups = len(data[plotting_mode][cluster_type])
      group_colors = plt.cm.get_cmap('tab20', num_groups)
      with open(os.path.join(save_path, 'info.txt'), 'a') as file:  # 'a' stands for append mode
        print(f"number of '{cluster_type}' '{plotting_mode}': '{num_groups}'", file=file)
        # Add a separator for clarity after each block of information
        print("\n", file=file)
      
      for seg in detailed_segments:
        if plotting_mode == 'functional_groups' and seg.functional_group_indices:
          seg.color = group_colors(seg.functional_group_indices[0])
        elif plotting_mode == 'presynaptic_cells' and seg.presynaptic_cell_indices:
          seg.color = group_colors(seg.presynaptic_cell_indices[0])
      
      # plot
      logger.log(f'plotting {cluster_type} {plotting_mode}\n\n\n\n')
      save_name = os.path.join(save_path, f'{cluster_type}_{plotting_mode}.png')
      plot_segments(detailed_segments, save_name)
      if plotting_mode == 'presynaptic_cells':
        save_name = os.path.join(save_path, f'mean_fr_{cluster_type}_{plotting_mode}.png')
        plot_segments_mean_firing_rate(detailed_segments, save_name, mean_value=mean_mean_fr, std_value=std_mean_fr)
  
if __name__ == "__main__":
    main(cluster_types, plotting_modes, output_folder)