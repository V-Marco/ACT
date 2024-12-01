import sys
sys.path.append("../")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from Modules.logger import Logger
from Modules.plotting_utils import plot_adjacent_segments
from Modules.segment import SegmentManager
from Modules.constants import SimulationParameters

from multiprocessing import Pool, cpu_count

# Settings
segs_to_plot = {
	'Soma': True,
	'Soma_Adj': False,
	'Axon': True,
	'Nexus': True,
	'Basal': True,
	'Tuft': True
}

how_to_plot = {
	'soma spikes': True, # Index to plot
	'specific_time': False, # (ms)
	'values_at_specific_time': False,
	'seg_locations': True
}

soma_spike_settings = {
	'indices': [0, 100], # List of spike numbers (not used currently)
	'range': 100, # (ms) before and after
	'number': 5, # Should probably change to use either number or indices # currently number
	'plot_adj_Vm': True, # Whether to include adj Vm in plot
	'plot_total_AC': False # whether to include seg net Ax current.
}

specific_time_settings = {
	'time' : 500, # (ms)
	'range': 100, # (ms) before and after
	'plot_adj_Vm': True, # whether to include adj Vm in plot
	'plot_total_AC': False # whether to include seg net Ax current.
}

def get_segments_of_type(segments, segment_type):
	return [seg for seg in segments if segment_type in seg.seg]

def get_segment_with_specific_string(segments, substr):
	for seg in segments:
		if substr in seg.seg:
			return seg
	return None

def load_segment_indexes(output_folder):
	with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
		return pickle.load(file)

def subset_data(t, xlim):
	return np.where((t >= xlim[0]) & (t <= xlim[1]))[0]

def plot_all_segments(segments_to_plot, t, current_types, save_path, specific_time, sm):
	indices = subset_data(t, [specific_time - 100, specific_time + 100])
	for prefix, segments in segments_to_plot.items():
		for i, seg in enumerate(segments):
			plot_all(
				segment = seg, 
				t = t, 
				current_types = current_types, 
				indices = indices, 
				index = -1, 
				save_to = save_path, 
				title_prefix = prefix + str(i), 
				ylim = [-1, 1] if prefix == "Nexus_" else None, 
				vlines = np.array(sm.soma_spiketimes))
			plot_all(
				segment = seg, 
				t = t, 
				current_types = current_types, 
				indices = None, 
				index = None, 
				save_to = save_path, 
				title_prefix = prefix, 
				ylim=[-1, 1] if prefix == "Nexus" else None, vlines = np.array(sm.soma_spiketimes))

def get_soma_adjacent_segments(soma):
	"""
	Create a dictionary of segment types based on the segments that are adjacent to the soma
	and includes the soma itself.

	Parameters:
	- soma: The soma segment object, expected to have a property 'adj_segs' listing all its adjacent segments.

	Returns:
	A dictionary with keys being segment type names (inferred from segment names) and values being lists of segments.
	"""
	segment_types = {"Soma": [soma]}  # Initialize with the soma

	for adj_seg in soma.adj_segs:
		if adj_seg.type.lower().startswith("dend"):
			seg_type = "Basal"
		else:
			seg_type = adj_seg.type.capitalize()

		# Prefix the type with "Soma_Adj_"
		soma_adj_type = "Soma_Adj_" + seg_type

		if seg_type not in segment_types:
			segment_types[seg_type] = []
		if soma_adj_type not in segment_types:
			segment_types[soma_adj_type] = []

		segment_types[seg_type].append(adj_seg)
		segment_types[soma_adj_type].append(adj_seg)

	return segment_types


def print_steady_state_values(segment, t, steady_state_time_index, data_types=[], title_prefix=None, return_values=False, show_individuals=False):
	'''
	Print (and optionally return) the steady state values of currents and axial currents 
	for a given segment at a specific time index.
	
	Parameters:
	- segment: SegmentManager Segment object
	- t: time vector
	- steady_state_time_index: Index at which the steady state values should be printed
	- title_prefix: Prefix for the title (typically denotes the segment type)
	- return_values: If set to True, return the currents as a dictionary instead of printing
	- show_individuals: If set to True, show individual axial currents. Otherwise, show only summed currents.
	'''

	values = {}  # Dictionary to hold the values if return_values is True

	# Current types present in the segment
	#data_types = ['v','ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas', 'ica', 'iampa','inmda','igaba']

	# Print title
	if title_prefix and not return_values:
		print(f"{title_prefix} - Steady State Values at time {t[steady_state_time_index]}ms:")
	elif not return_values:
		print(f"Steady State Values at time {t[steady_state_time_index]}ms:")

	for current in data_types:
		data = getattr(segment, current)
		if current == 'v':
			units = 'mV'
		else:
			units = 'nA'

		if return_values:
			values[current] = data[steady_state_time_index]
		else:
			print(f"{current}: {data[steady_state_time_index]} {units}")

	# If there are axial currents
	if hasattr(segment, 'axial_currents'):
		axial_current_by_type = {}  # Store summed axial currents by type
		for idx, adj_seg in enumerate(segment.adj_segs):
			axial_current_value = segment.axial_currents[idx][steady_state_time_index]

			# Sum the axial currents by type
			if adj_seg.type in axial_current_by_type:
				axial_current_by_type[adj_seg.type] += axial_current_value
			else:
				axial_current_by_type[adj_seg.type] = axial_current_value

			if show_individuals and not return_values:
				print(f"Axial current from {segment.name} to {adj_seg.name} (Type: {adj_seg.type}): {axial_current_value} nA")

		for seg_type, axial_current_sum in axial_current_by_type.items():
			if not return_values:
				print(f"Total axial current to {seg_type} type segments: {axial_current_sum} nA")

		total_AC = sum(axial_current_by_type.values())
		if not return_values:
			print(f"Total summed axial currents: {total_AC} nA")

	if not return_values:
		print("\n")  # For readability

	if return_values:
		return values
		
def plot_around_spikes(spiketimes, number_to_plot, segments_to_plot, t, current_types, save_path, sm, t_range, plot_adj_Vm, plot_total_AC):
	for i, AP_time in enumerate(np.array(spiketimes)):
		if i < number_to_plot:
			before_AP = AP_time - t_range  # ms
			after_AP = AP_time + t_range  # ms
			xlim = [before_AP, after_AP]  # time range
		
			# Subset the data for the time range
			indices = subset_data(t, xlim)
			for prefix, segments in segments_to_plot.items():
				for j, seg in enumerate(segments):
					plot_all(segment=seg, t=t, current_types=current_types, indices=indices, index=i+1, save_to=save_path, title_prefix=prefix+str(j), ylim=[-1, 1] if prefix == "Nexus" else None, vlines=np.array(sm.soma_spiketimes), plot_adj_Vm=plot_adj_Vm, plot_total_AC=plot_total_AC)

def plot_membrane_currents(segment, t, currents, ax, indices):
	for current in currents:
		if '+' in current:
			currents_to_sum = current.split('+')
			max_index = np.max(indices)
			array_length = len(getattr(segment, currents_to_sum[0]))

			if max_index >= array_length:
				raise RuntimeError(f"Trying to access index {max_index} in an array of size {array_length}")

			indices = [i for i in indices if i < array_length]

			data = getattr(segment, currents_to_sum[0])[indices] if indices is not None else getattr(segment, currents_to_sum[0])

			for current_to_sum in currents_to_sum[1:]:
				data += getattr(segment, current_to_sum)[indices] if indices is not None else getattr(segment, current_to_sum)
		else:
			data = getattr(segment, current)[indices] if indices is not None else getattr(segment, current)

		if np.shape(t) != np.shape(data):
			ax.plot(t[:-1], data, label = current)
		else:
			ax.plot(t, data, label = current)

def plot_voltage(segment, t, indices, plot_adj_Vm, ax):
	v_data = segment.v[indices] if indices is not None else segment.v

	ax.plot(t, v_data, color = segment.color, label = segment.name)

	if plot_adj_Vm:
		for adj_seg in segment.adj_segs:
			adj_v_data = adj_seg.v[indices] if indices is not None else adj_seg.v
			if adj_seg.color == segment.color:
				ax.plot(t, adj_v_data, label=adj_seg.name, color = 'Magenta')
			else:
				ax.plot(t, adj_v_data, label=adj_seg.name, color = adj_seg.color)

def plot_axial_currents(segment, t, indices, plot_total_AC, ax):
	# For axial currents 'Axial Current from [{}]'
	total_AC, total_dend_AC, total_to_soma_AC, total_away_soma_AC = [np.zeros(len(segment.v))] * 4

	# Gather axial currents
	for adj_seg_index, adj_seg in enumerate(segment.adj_segs):
		# All dendrites
		total_AC += segment.axial_currents[adj_seg_index]

		# Plotting soma's ACs
		# Sum AC from basal dendrites
		if (segment.type == 'soma') and (adj_seg.type == "dend"):
			total_dend_AC += segment.axial_currents[adj_seg_index]

		# Plot axon & apical trunk ACs
		elif (segment.type == 'soma') and (adj_seg.type != "dend"):
			axial_current = segment.axial_currents[adj_seg_index][indices] if indices is not None else segment.axial_currents[adj_seg_index]
			ax.plot(t, axial_current, label = adj_seg.name, color = adj_seg.color)

		# Plotting any other segment's ACs, sum axial currents to or away soma.
		# Parent segs will be closer to soma with our model.
		elif (segment.type != 'soma') and (adj_seg in segment.parent_segs):
			total_to_soma_AC += segment.axial_currents[adj_seg_index]
		
		elif (segment.type != 'soma') and (adj_seg not in segment.parent_segs):
			total_away_soma_AC += segment.axial_currents[adj_seg_index]
				
	# If we are plotting for soma segment, sum basal axial currents
	if segment.type == 'soma':
		basal_axial_current = total_dend_AC[indices] if indices is not None else total_dend_AC
		ax.plot(t, basal_axial_current, label = 'Summed axial currents to basal segments', color = 'red')

	# If not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
	else:
		total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
		ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
		total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
		ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')

	if plot_total_AC:
		total_AC = total_AC[indices] if indices is not None else total_AC
		ax.plot(t, total_AC, label = 'Summed axial currents out of segment', color = 'Magenta')

def plot_all(
		segment, 
		t, 
		current_types = [], 
		indices = None, 
		xlim = None, 
		ylim = None, 
		index = None, 
		save_to = None, 
		title_prefix = None, 
		vlines = None, 
		plot_adj_Vm = True, 
		plot_total_AC = True):
	
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
		
	titles = [
		'Axial Current from [{}] to adjacent segments',
		'Vm from [{}] and adjacent segments',
		'Currents from [{}]'
	]

	if index:
		for i, title in enumerate(titles):
			titles[i] = 'Spike ' + str(int(index)) + ' ' + title
			
	ylabels = ['nA', 'mV', 'nA']
	data_types = ['axial_currents', 'v', current_types]

	fig, axs = plt.subplots(len(titles), figsize = (12.8, 4.8 * len(titles)))
	
	for j, ax in enumerate(axs):

		title = titles[j].format(segment.name)
		ylabel = ylabels[j]
		data_type = data_types[j]

		# Membrane current plots
		if type(data_type) == list: 
			plot_membrane_currents(segment, t, data_type, ax, indices)

		# Voltage plots
		elif data_type == 'v':
			plot_voltage(segment, t, indices, plot_adj_Vm, ax)

		# Axial currents
		elif data_type == 'axial_currents':
			plot_axial_currents(segment, t, indices, plot_total_AC, ax)
			
		else:
			raise(ValueError(f"Cannot analyze {data_type}"))

		# Indicate action potentials via dashed vertical lines
		# Only the axial currents plot
		if (vlines is not None) and (j == 0):
			for ap_index, vline in enumerate(vlines):
				if ap_index == 0: # only label one so that legend is not outrageous
					ax.vlines(vline, ymin = ax.get_ylim()[0], ymax = ax.get_ylim()[1], color = 'black', label = 'AP time', linestyle = 'dashed')
				else:
					ax.vlines(vline, ymin = ax.get_ylim()[0], ymax = ax.get_ylim()[1], color = 'black', linestyle = 'dashed')

		ax.axhline(0, color = 'grey')
		if xlim: ax.set_xlim(xlim)
		ax.set_ylabel(ylabel)
		ax.set_xlabel('Time (ms)')
		ax.legend(loc = 'upper right')

		if title_prefix:
			ax.set_title(title_prefix+title)
		else:
			ax.set_title(title)
			
	plt.tight_layout()

	if save_to:
		# Plotting the entire sim
		if index is None:
			index = "wholesim_"
		else:
			index = "AP" + "{}_".format(index)

		if title_prefix:
			fig.savefig(os.path.join(save_to, index + title_prefix + ".png"))
		else:
			fig.savefig(os.path.join(save_to, index + ".png"))

	plt.close()

def get_segments_of_type(segments, segment_type):
	return [seg for seg in segments if segment_type in seg.seg]

def get_segment_with_specific_string(segments, string):
	for seg in segments:
		if string in seg.name:  # Assuming the segment has a name attribute
			return seg
	return None

def analyze_currents(parameters: SimulationParameters):
  
	save_path = os.path.join(parameters.path, "currents/")
  
	step_size = int(parameters.save_every_ms / parameters.h_dt) # Timestamps
	steps = range(step_size, int(parameters.h_tstop / parameters.h_dt) + 1, step_size) # Timestamps
	
	sm = SegmentManager(
		output_folder = parameters.path, 
		steps = steps, 
		dt = parameters.h_dt, 
		skip = parameters.skip, 
		transpose = False
		)#channel_names = parameters.channel_names)

	# Can probably change this to read the recorded t_vec
	t = np.arange(0, len(sm.segments[0].v) * parameters.h_dt, parameters.h_dt)

	# Compute axial currents from each segment toward its adjacent segments.
	# Compute axial currents between all segments
	sm.compute_axial_currents()
  
	# Finding segments
	soma_adj_segs = get_soma_adjacent_segments(sm.segments[0])
  
	soma_segs = get_segments_of_type(sm.segments, 'soma')
	if len(soma_segs) != 1:
		soma_segs = [soma_segs[3]]
  
	seg_indexes = load_segment_indexes(parameters.path)
#	if 'BenModel' in parameters:
#		nexus_seg_index, basal_seg_index = [], []
#	else:
	nexus_seg_index, basal_seg_index, axon_seg_index, tuft_seg_index = seg_indexes["nexus"], seg_indexes["basal"], seg_indexes["axon"], seg_indexes["tuft"]
  
	nexus_segs = [sm.segments[nexus_seg_index]]
	basal_segs = [sm.segments[basal_seg_index]]
	axon_segs = [sm.segments[axon_seg_index]]
	tuft_segs = [sm.segments[tuft_seg_index]]
	axon_seg = get_segment_with_specific_string(sm.segments, '[0](0.5)') or get_segment_with_specific_string(sm.segments, '(0.5)')
  
	# if constants.build_cell_reports_cell:
	# 	nexus_seg = get_segment_with_specific_string(sm.segments, 'apic[24]')
	# 	if nexus_seg:
	# 		nexus_segs = [nexus_seg]

	#current_types = parameters.channel_names.copy()
	#current_types.remove('gNaTa_t_NaTa_t')
	#current_types.remove('ihcn_Ih')
	#current_types.remove('ina_NaTa_t')
	current_types = ['i_pas','ina','ica','ik','imembrane','iampa','inmda','igaba']
  
	# Combine the dictionaries
	segments_to_plot = {**soma_adj_segs}
	
	# Add the Nexus, Basal, Axon, and Tuft segments to segments_to_plot
	segments_to_plot['Nexus'] = nexus_segs

	if 'Basal' not in segments_to_plot:
		segments_to_plot['Basal'] = []

	segments_to_plot['Basal'].extend(basal_segs)
	segments_to_plot['Axon'] = axon_segs
	segments_to_plot['Tuft'] = tuft_segs
  
	# Filter out segment types based on segs_to_plot setting
	segments_to_plot = {seg_type: segments for seg_type, segments in segments_to_plot.items() 
						if segs_to_plot.get(seg_type, False) or (seg_type.startswith('Soma_Adj_') and segs_to_plot.get('Soma_Adj', False))}
					
	if how_to_plot['seg_locations']: # plots adjacent seg locations
		for seg_type, segs in segments_to_plot.items():
			plot_adjacent_segments(segs = segs, sm=sm, title_prefix=f"{seg_type}_", save_to=save_path)
					
	if how_to_plot["values_at_specific_time"]: # prints values at a single time index.
		# Filter segments_to_plot to only include Soma_Adj segments
		filtered_segment_types = {k: v for k, v in segments_to_plot.items() if k.startswith('Soma_Adj')}
			
		steady_state_index = int(specific_time_settings['time'] / parameters.h_dt)
		
		# Initializing the dictionary for summed dendritic currents
		summed_dend_currents = {}
		
		# Loop over the filtered segment types and call print_steady_state_values
		for title_prefix, segments in filtered_segment_types.items():
			for seg in segments:
				# Checking if it's a Basal dendrite segment
				if title_prefix == "Soma_Adj_Basal":
					dend_currents = print_steady_state_values(seg, t, steady_state_index, data_types = current_types, return_values = True)
					
					for channel, current in dend_currents.items():
						if channel not in summed_dend_currents:
							summed_dend_currents[channel] = 0
						summed_dend_currents[channel] += current
				else:
					print_steady_state_values(seg, t, steady_state_index, data_types = current_types, title_prefix=title_prefix)
	  
		# Print summed dendritic currents
		for channel, current in summed_dend_currents.items():
			if channel == 'v':
				print(f"{channel}: {current:.4f} mV")
			else:
				print(f"{channel}: {current:.4f} nA")

	if how_to_plot['soma spikes']: # plots voltage, axial current, and membrane currents around spike times.
		print(f'number of spikes: {len(sm.soma_spiketimes)}, firing rate: {len(sm.soma_spikestimes)/(len(sm.segments[0].v)*parameters.h_dt/1000)}')
		plot_around_spikes(sm.soma_spiketimes, number_to_plot=soma_spike_settings["number"], segments_to_plot=segments_to_plot, t=t, current_types=current_types, save_path=save_path, sm=sm, t_range=soma_spike_settings["range"], plot_adj_Vm=soma_spike_settings['plot_adj_Vm'], plot_total_AC=soma_spike_settings['plot_total_AC'])

	if how_to_plot["specific_time"]:  # plots voltage, axial current, and membrane currents around a specific time.
		plot_all_segments(segments_to_plot, t, current_types, save_path, specific_time=specific_time_settings["time"], sm=sm, plot_adj_Vm=specific_time_settings['plot_adj_Vm'], plot_total_AC=specific_time_settings['plot_total_AC'])

if __name__ == "__main__":

	# Parse cl args
	if len(sys.argv) != 3:
		raise RuntimeError("usage: python file -f folder")
	else:
		sim_folder = sys.argv[sys.argv.index("-f") + 1]

	logger = Logger(None)
	
	# Retrieve all jobs
	jobs = []
	for job in os.listdir(sim_folder):
		if job.startswith("."): continue
		with open(os.path.join(sim_folder, job, "parameters.pickle"), "rb") as parameters:
			jobs.append(pickle.load(parameters))

	logger.log(f"Total number of jobs found: {len(jobs)}")
	logger.log(f"Total number of proccessors: {cpu_count()}")

	pool = Pool(processes = len(jobs))
	pool.map(analyze_currents, jobs)
	pool.close()
	pool.join()