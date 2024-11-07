import numpy as np
from sklearn.cluster import KMeans
from cell_model import CellModel
from neuron import h
import pandas as pd

class FunctionalGroup:
  
	def __init__(
			self, 
			seg_idxs: list, 
			seg_coords: np.ndarray,
			name: str = None):
		
		self.seg_idxs = seg_idxs
		self.seg_coords = seg_coords
		self.name = name
		self.presynaptic_cells = []

class PresynapticCell:
  
	def __init__(
			self, 
			seg_idxs: list, 
			name: str = None, 
			cluster_center: np.ndarray = None):
		
		self.seg_idxs = seg_idxs
		self.name = name
		self.cluster_center = cluster_center

		self.mean_fr = None
		self.spike_train = None
		self.vecstim = None

	def set_spike_train(self, mean_fr, spike_train):
		self.mean_fr = mean_fr
		self.spike_train = spike_train
		self._set_vecstim(spike_train)

	def _set_vecstim(self, spike_train):
		vec = h.Vector(spike_train)
		stim = h.VecStim()
		stim.play(vec)
		self.vecstim = stim

class PCBuilder:

	@staticmethod
	def assign_presynaptic_cells(cell, n_func_gr, n_pc_per_fg, synapse_names):

		seg_coords = [c.coords[["pc_0", "pc_1", "pc_2"]] for c in cell.get_segments(["all"])[1]]
		seg_coords = pd.concat(seg_coords).to_numpy()

		# Cluster cell segments into functional groups
		labels, _ = PCBuilder._cluster_segments(seg_coords = seg_coords, n_clusters = n_func_gr)

		for fg_idx in np.unique(labels):

			# Find indexes of segments belonging to the fg
			fg_seg_idxs = []
			for seg_ind in range(len(labels)):
				if labels[seg_ind] == fg_idx: fg_seg_idxs.append(seg_ind)

			functional_group = FunctionalGroup(
				seg_idxs = fg_seg_idxs,
				seg_coords = seg_coords[fg_seg_idxs],
				name = "fg_" + str(fg_idx))
			
			PCBuilder._build_presynaptic_cells_for_a_fg(functional_group, n_pc_per_fg)
			PCBuilder._map_synapses_to_pc(cell, synapse_names, functional_group)

	@staticmethod
	def _cluster_segments(seg_coords: np.ndarray, n_clusters: int):
		km = KMeans(n_clusters = n_clusters, n_init = "auto")
		seg_id_to_cluster_index = km.fit_predict(seg_coords)
		cluster_centers = km.cluster_centers_
		return seg_id_to_cluster_index, cluster_centers
	
	@staticmethod
	def _build_presynaptic_cells_for_a_fg(fg: FunctionalGroup, n_pc_per_fg: int):

		# Cluster functional group segments into presynaptic cells
		labels, centers = PCBuilder._cluster_segments(
			seg_coords = fg.seg_coords, 
			n_clusters = np.min((n_pc_per_fg, len(fg.seg_coords))))
		
		# Create presynaptic cells
		for pc_idx in np.unique(labels):

			# Gather presynaptic cell segments
			pc_seg_idxs = []
			for seg_ind, global_seg_index in enumerate(fg.seg_idxs):
				if labels[seg_ind] == pc_idx: pc_seg_idxs.append(global_seg_index)
		
			# Create PresynapticCell object
			presynaptic_cell = PresynapticCell(
				seg_idxs = pc_seg_idxs, 
				name = fg.name + '_cell' + str(pc_idx), 
				cluster_center = centers[pc_idx])
			
			fg.presynaptic_cells.append(presynaptic_cell)
	
	@staticmethod
	def _map_synapses_to_pc(cell: CellModel, synapse_names: list, functional_group: list):

		all_segments, _ = cell.get_segments(["all"])

		# Need to separate synapses list because we do not want to give exc and inh synapses the same presynaptic cell
		for name in synapse_names:
			synapses = cell.get_synapses(name)

			for synapse in synapses:
				if synapse.pc is not None:
					continue

				seg_index = all_segments.index(synapse.h_syn.get_segment())
				if seg_index in functional_group.seg_idxs == False:
					continue

				for pc in functional_group.presynaptic_cells:
					if seg_index in pc.seg_idxs:
						cell.synapses[cell.synapses.index(synapse)].pc = pc