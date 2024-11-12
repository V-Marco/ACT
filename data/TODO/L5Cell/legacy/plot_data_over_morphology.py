import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.animation as animation

from cell_inference.config import params
from Modules.segment import SegmentManager
from Modules.plotting_utils import get_nested_property, plot_morphology
import constants


output_folder = "output/2023-10-10_13-37-26_seeds_130_90PTcell[0]_174nseg_102nbranch_13999NCs_13999nsyn"
#output_folder = "output/BenModel"
# constants.save_every_ms = 200
# constants.h_tstop = 2500
# if 'BenModel' in output_folder:
#   constants.save_every_ms = 3000
#   constants.h_tstop = 3000
#   transpose =True
# skip = 300 # (ms)

find_average = False
animate_plot = False
interactive = False
constants.cmap_type = 'cool' # 'Greys'

constants.show_electrodes = False
if constants.show_electrodes:
	elec_pos = params.ELECTRODE_POSITION
else:
	elec_pos = None

# Default position parameters
loc_param = [0., 0., 45., 0., 1., 0.]
# [0., 0., 25., 0., 1., 0.] # resulted in 5 uV LFP
# [0., 0., 50., 0., 1., 0.] # resulted in 10 uV LFP
# [0., 0., 80., 0., 1., 0.] # resulted in 5 uV LFP # original

# Default view
elev, azim = 10, -45 # 90

new_property = None #['inmda','iampa','net_exc_i'] # set to None if not using existing property

time_index = 300

#property_list_to_analyze = ['seg_gcanbar']#['v']

property_list_to_analyze = ['netcon_SA_density_per_seg','exc']
# property_list_to_analyze = ['seg_elec_info','beta','passive_soma']

animation_step = int(1 / constants.h_dt)
	
def main(property_list_to_analyze):

	step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
	steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

	#sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, no_data=True)
	sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, no_data=False)
	sm.compute_axial_currents()

	if new_property is not None:
		sm.sum_currents(currents = new_property[:-1], var_name = new_property[-1])
		property_list_to_analyze = [new_property[-1]]
	
	# Get seg property
	if find_average: # Average value of all time points
		seg_prop = np.array([np.mean(get_nested_property(seg, property_list_to_analyze)) for seg in sm.segments])
	else:
		seg_prop = np.array([get_nested_property(seg, property_list_to_analyze, time_index) for seg in sm.segments])

	# Get label
	label = '_'.join(property_list_to_analyze)
	if find_average: label = 'mean_' + label

	# Robust normalization
	lower, upper = np.percentile(seg_prop, [1, 95])
	robust_norm = plt.Normalize(vmin = lower, vmax = upper)
	normalized_seg_prop = robust_norm(seg_prop)

	# Generate Color map
	cmap = plt.get_cmap(constants.cmap_type)
	segment_colors = cmap(normalized_seg_prop)

	# Create a ScalarMappable object to represent the colormap
	smap = plt.cm.ScalarMappable(cmap = cmap, norm = robust_norm)
	
	def configure_plot(x, y, z, alpha, beta, phi, elev, azim):
		loc_param = (x, y, z, np.pi / 180 * alpha, np.cos(np.pi / 180 * beta), np.pi / 180 * phi)
		fig, ax = plot_morphology(segments = sm.segments, electrodes = elec_pos, move_cell = loc_param, elev = -elev, azim = -azim, figsize = (12, 8), 
			    				  seg_property = label, segment_colors = segment_colors, sm = smap)
		return fig, ax
	
	if animate_plot:
			plots = []
			for i in range(0, int(constants.h_tstop / constants.h_dt) + 1, animation_step):
				if i == 0:
					fig, ax = configure_plot(x = loc_param[0], y = loc_param[1], z = loc_param[2], alpha = 180 / np.pi * loc_param[3], 
			      							 beta = 180 / np.pi * np.arccos(loc_param[4]), phi = 180 / np.pi * loc_param[5], elev = -elev, azim = -azim)
				else:
					seg_prop = np.array([get_nested_property(seg, property_list_to_analyze, i) for seg in sm.segments])
					normalized_seg_prop = robust_norm(seg_prop)
					segment_colors = cmap(normalized_seg_prop)
					smap = plt.cm.ScalarMappable(cmap = cmap, norm = robust_norm)
					dummy_fig, ax = configure_plot(x = loc_param[0], y = loc_param[1], z = loc_param[2], alpha = 180 / np.pi * loc_param[3], 
				    							   beta = 180 / np.pi * np.arccos(loc_param[4]), phi = 180 / np.pi * loc_param[5], elev = -elev, azim = -azim)
					ax.set(animated = True)
					ax.remove()
					ax.figure = fig
					fig.add_axes(ax)
					plt.close(dummy_fig)
				plots.append([ax])

			ani = animation.ArtistAnimation(fig, plots)
			ani.save(os.path.join(output_folder, f"video.mp4"))
	else:
		fig, _ = configure_plot(x = loc_param[0], y = loc_param[1], z = loc_param[2], alpha = 180 / np.pi * loc_param[3], 
		       				 beta = 180 / np.pi * np.arccos(loc_param[4]), phi = 180 / np.pi * loc_param[5], elev = -elev, azim = -azim)
		fig.savefig(os.path.join(output_folder, label + '.png'))
		plt.show()


	
if __name__ == "__main__":
		main(property_list_to_analyze)
		
		