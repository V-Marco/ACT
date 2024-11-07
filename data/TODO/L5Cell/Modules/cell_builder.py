from enum import Enum
import pickle
import os
import numpy as np
from functools import partial
import scipy.stats as st

from neuron import h

from logger import Logger
from spike_generator import PoissonTrainGenerator
from constants import SimulationParameters
from cell_model import CellModel
from presynaptic import PCBuilder
from reduction import Reductor

class SkeletonCell(Enum):

	def __eq__(self, other):
		if type(self).__qualname__ != type(other).__qualname__: 
			return NotImplemented
		return self.name == other.name and self.value == other.value
	
	Hay = {
		"biophys": "L5PCbiophys3ActiveBasal.hoc",
		"morph": "cell1.asc",
		"template": "L5PCtemplate.hoc",
		"pickle": None,
		"modfiles": "../modfiles/hay"
		}
	HayNeymotin = {
		"biophys": "M1_soma_L5PC_dendrites.hoc",
		"morph": "cell1.asc",
		"template": "L5PCtemplate.hoc",
		"pickle": "../cells/pickled_parameters/neymotin_detailed/PT5B_full_cellParams.pkl"
	}
	NeymotinReduced = {
		"biophys": None,
		"morph": None,
		"template": "ziao_templates.hoc",
		"pickle": None
	}
	NeymotinDetailed = {
		"biophys": None,
		"morph": None,
		"template": "PTcell.hoc",
		"pickle": None
	}

def log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip):
	val = np.random.lognormal(gmax_mean, gmax_std, size)
	s = gmax_scalar * float(np.clip(val, clip[0], clip[1]))
	return s

def binned_log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip):
	val = np.random.lognormal(gmax_mean, gmax_std, size)
	s = gmax_scalar * float(np.clip(val, clip[0], clip[1]))

	# Bin
	num_bins = 10
	bin_size = (clip[1] - clip[0]) / num_bins
	bins = np.arange(0, clip[1], bin_size)
	ind = np.digitize(s, bins)

	if ind == num_bins:
		return bins[-1]
	else:
		return bins[ind]

# Firing rate distribution
def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
	return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, loc = loc, scale = scale, size = size)) + 1e-15

def gamma_dist(mean, size = 1):
	shape = 5
	scale = mean / shape
	return np.random.gamma(shape, scale, size) + 1e-15

# Release probability distribution
def P_release_dist(P_mean, P_std, size):
	val = np.random.normal(P_mean, P_std, size)
	s = float(np.clip(val, 0, 1))
	return s

class CellBuilder:

	templates_folder = "../cells/templates"

	def __init__(self, cell_type: SkeletonCell, parameters: SimulationParameters, logger: Logger) -> None:

		self.cell_type = cell_type
		self.parameters = parameters
		self.logger = logger

	def build_cell(self):

		random_state = np.random.RandomState(self.parameters.numpy_random_state)
		np.random.seed(self.parameters.numpy_random_state)
		neuron_r = h.Random()
		neuron_r.MCellRan4(self.parameters.neuron_random_state)

		# Build skeleton cell
		self.logger.log(f"Building {self.cell_type}.")

		if self.cell_type == SkeletonCell.Hay:
			skeleton_cell = self.build_Hay_cell()

		elif self.cell_type == SkeletonCell.HayNeymotin:
			skeleton_cell = self.build_HayNeymotin_cell()

		elif self.cell_type == SkeletonCell.NeymotinDetailed:
			skeleton_cell = self.build_Neymotin_detailed_cell()

		cell = CellModel(skeleton_cell, random_state, neuron_r, self.logger)

		# # Build synapses
		# self.logger.log("Building excitatory synapses.")
		# self.build_excitatory_synapses(cell = cell)

		# self.logger.log("Building inhibitory synapses.")
		# self.build_inhibitory_synapses(cell = cell)

		# self.logger.log("Building soma synapses.")
		# self.build_soma_synapses(cell = cell)

		# # Assign spike trains

		# self.logger.log("Assigning excitatory spike trains.")
		# exc_spike_trains = self.assign_exitatory_spike_trains(cell = cell, random_state = random_state)

		# self.logger.log("Assigning inhibitory spike trains.")
		# inh_spike_trains = self.assign_inhibitory_spike_trains(cell = cell, random_state = random_state)

		# self.logger.log("Assigning soma spike trains.")
		# soma_spike_trains = self.assign_soma_spike_trains(cell = cell, random_state = random_state)
      
		# self.logger.log("Finished creating a CellModel object.")

		# # @CHECK ----
		# # Turn off certain presynaptic neurons to simulate in vivo
		# if (self.parameters.CI_on == False) and (self.parameters.trunk_exc_synapses == False):
		# 	for synapse in cell.synapses:
		# 		if (
		# 			(synapse.h_syn.get_segment().sec in cell.apic) and 
		# 			(synapse.syn_mod in self.parameters.exc_syn_mod) and 
		# 			(synapse.h_syng.get_segment().sec in cell.get_tufts_obliques()[1] == False) and 
		# 			(synapse.h_syn.get_segment().sec.y3d(0) < 600)):
		# 			for netcon in synapse.netcons: netcon.active(False)
		
		# # Turn off perisomatic exc neurons
		# if (self.parameters.perisomatic_exc_synapses == False):
		# 	for synapse in cell.synapses:
		# 		if (
		# 			(h.distance(synapse.h_syn.get_segment(), cell.soma[0](0.5)) < 75) and 
		# 			(synapse.syn_mod in self.parameters.exc_syn_mod)):
		# 			for netcon in synapse.netcons: netcon.active(False)
		
		# # ----
		
		# # Merge synapses
		# # if self.parameters.merge_synapses:
		# # 	reductor.merge_synapses(cell)

		# # Set recorders
		cell.add_segment_recorders(var_name = "v")
		# cell.add_spike_recorder(sec = cell.soma[0], var_name = "soma_spikes", spike_threshold = self.parameters.spike_threshold)

		# Add current injection
		if self.parameters.CI_on:
			cell.set_soma_injection(
				amp = self.parameters.h_i_amplitude,
				dur = self.parameters.h_i_duration, 
				delay = self.parameters.h_i_delay)

		# Also return skeleton_cell to keep references
		return cell, skeleton_cell

	def assign_soma_spike_trains(self, cell, random_state) -> None:

		soma_spike_trains = []

		PCBuilder.assign_presynaptic_cells(
			cell = cell,
			n_func_gr = 1,
			n_pc_per_fg = 1,
			synapse_names = ["soma"]
		)

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		for i, synapse in enumerate(cell.synapses):
			if synapse.name == "soma":
				mean_fr = proximal_inh_dist(size = 1)
				firing_rates = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
					num = self.parameters.h_tstop,
					random_state = random_state,
					lambda_mean = mean_fr)
				spike_train = PoissonTrainGenerator.generate_spike_train(
					lambdas = firing_rates, 
					random_state = random_state)

				cell.synapses[i].set_spike_train_for_pc(mean_fr = mean_fr, spike_train = spike_train.spike_times)
				soma_spike_trains.append(spike_train.spike_times)
			else:
				soma_spike_trains.append([])
		
		return soma_spike_trains


	def assign_inhibitory_spike_trains(self, cell, random_state) -> None:

		inh_spike_trains = []

		PCBuilder.assign_presynaptic_cells(
			cell = cell,
			n_func_gr = self.parameters.inh_distributed_n_FuncGroups,
			n_pc_per_fg = self.parameters.inh_distributed_n_PreCells_per_FuncGroup,
			synapse_names = ["inh"]
		)

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		# Distal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_distal_mean_fr, self.parameters.inh_distal_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		soma_coords = cell.get_segments(["soma"])[1][0].coords[["pc_0", "pc_1", "pc_2"]].to_numpy()

		for i, synapse in enumerate(cell.synapses):
			if synapse.name == "inh":
				if np.linalg.norm(soma_coords - synapse.pc.cluster_center) < 100:
					mean_fr = proximal_inh_dist(size = 1)
				else:
					mean_fr = distal_inh_dist(size = 1)

				firing_rates = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
					num = self.parameters.h_tstop,
					random_state = random_state,
					lambda_mean = mean_fr)
				spike_train = PoissonTrainGenerator.generate_spike_train(
					lambdas = firing_rates, 
					random_state = random_state)
				
				cell.synapses[i].set_spike_train_for_pc(mean_fr = mean_fr, spike_train = spike_train.spike_times)
				inh_spike_trains.append(spike_train.spike_times)
			else:
				inh_spike_trains.append([])
		
		return inh_spike_trains


	def assign_exitatory_spike_trains(self, cell, random_state) -> None:

		exc_spike_trains = []

		PCBuilder.assign_presynaptic_cells(
			cell = cell,
			n_func_gr = self.parameters.exc_n_FuncGroups,
			n_pc_per_fg = self.parameters.exc_n_PreCells_per_FuncGroup,
			synapse_names = ["exc"]
		)

		# Distribution of mean firing rates
		# mean_fr_dist = partial(gamma_dist, mean = self.parameters.exc_mean_fr, size = 1)
		mean_fr, std_fr = self.parameters.exc_mean_fr, self.parameters.exc_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		mean_fr_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		for i, synapse in enumerate(cell.synapses):
			if synapse.name == "exc":
				mean_fr = mean_fr_dist(size = 1)
				firing_rates = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
					num = self.parameters.h_tstop,
					random_state = random_state,
					lambda_mean = mean_fr)
				spike_train = PoissonTrainGenerator.generate_spike_train(
					lambdas = firing_rates, 
					random_state = random_state)
				cell.synapses[i].set_spike_train_for_pc(mean_fr = mean_fr, spike_train = spike_train.spike_times)
				exc_spike_trains.append(spike_train.spike_times)
			else:
				exc_spike_trains.append([])
		
		return exc_spike_trains

				
	def build_soma_synapses(self, cell) -> None:
		
		if (self.parameters.CI_on) or (not self.parameters.add_soma_inh_synapses):
			return None
		
		# inh_soma_P_dist = partial(
		# 	P_release_dist, 
		# 	P_mean = self.parameters.inh_soma_P_release_mean, 
		# 	P_std = self.parameters.inh_soma_P_release_std, 
		# 	size = 1)
		
		segments, seg_data = cell.get_segments(["soma"])
		probs = [data.membrane_surface_area for data in seg_data]
		
		cell.add_synapses_over_segments(
			segments = segments,
			nsyn = self.parameters.num_soma_inh_syns,
			syn_mod = self.parameters.inh_syn_mod,
			syn_params = self.parameters.inh_syn_params,
			gmax = self.parameters.soma_gmax_dist,
			name = "soma",
			density = False,
			seg_probs = probs,
			release_p = None)
			
	def build_inhibitory_synapses(self, cell) -> None:
		
		if self.parameters.CI_on:
			return None
		
		# Inhibitory release probability distributions
		# inh_apic_P_dist = partial(
		# 	P_release_dist, 
		# 	P_mean = self.parameters.inh_apic_P_release_mean, 
		# 	P_std = self.parameters.inh_apic_P_release_std, 
		# 	size = 1)
		# inh_basal_P_dist = partial(
		# 	P_release_dist, 
		# 	P_mean = self.parameters.inh_basal_P_release_mean, 
		# 	P_std = self.parameters.inh_basal_P_release_std, 
		# 	size = 1)
		
		# inh_P_dist = {}
		# inh_P_dist["apic"] = inh_apic_P_dist
		# inh_P_dist["dend"] = inh_basal_P_dist
		
		segments, seg_data = cell.get_segments(["apic", "dend"])
		probs = [data.membrane_surface_area for data in seg_data]

		cell.add_synapses_over_segments(
			segments = segments,
			nsyn = self.parameters.inh_synaptic_density,
			syn_mod = self.parameters.inh_syn_mod,
			syn_params = self.parameters.inh_syn_params,
			gmax = self.parameters.inh_gmax_dist,
			name = "inh",
			density = True,
			seg_probs = probs,
			release_p = None)
			
	def build_excitatory_synapses(self, cell) -> None:
		
		if self.parameters.CI_on:
			return None

		# Excitatory gmax distribution
		gmax_exc_dist = partial(
			log_norm_dist, 
			self.parameters.exc_gmax_mean_0, 
			self.parameters.exc_gmax_std_0, 
			self.parameters.exc_scalar, 
			size = 1, 
			clip = self.parameters.exc_gmax_clip)
		
		# exc release probability distribution everywhere
		# exc_P_dist = partial(
		# 	P_release_dist, 
		# 	P_mean = self.parameters.exc_P_release_mean, 
		# 	P_std = self.parameters.exc_P_release_std, 
		# 	size = 1)

		segments, seg_data = cell.get_segments(["apic", "dend"])
		if self.parameters.use_SA_exc:
			probs = [data.membrane_surface_area for data in seg_data]
		else:
			raise NotImplementedError

		cell.add_synapses_over_segments(
			segments = segments,
			nsyn = self.parameters.exc_synaptic_density,
			syn_mod = self.parameters.exc_syn_mod,
			syn_params = self.parameters.exc_syn_params,
			gmax = gmax_exc_dist,
			name = "exc",
			density = True,
			seg_probs = probs,
			release_p = None)

	def build_Hay_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.Hay.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.Hay.value["template"]))

		# Build skeleton_cell object
		skeleton_cell = h.L5PCtemplate(os.path.join(self.templates_folder, SkeletonCell.Hay.value["morph"]))

		return skeleton_cell

	def build_HayNeymotin_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["template"]))

		# Build skeleton_cell object
		skeleton_cell = h.L5PCtemplate(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["morph"]))

		# Swap soma and axon with the parameters from the pickle
		soma = skeleton_cell.soma[0] if self.is_indexable(skeleton_cell.soma) else skeleton_cell.soma
		axon = skeleton_cell.axon[0] if self.is_indexable(skeleton_cell.axon) else skeleton_cell.axon
		self.set_pickled_parameters_to_sections((soma, axon), SkeletonCell.HayNeymotin["pickle"])

		return skeleton_cell

	def build_Neymotin_detailed_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.NeymotinDetailed.value["template"]))
		skeleton_cell = h.CP_Cell(3, 3, 3)

		return skeleton_cell

	def build_Neymotin_reduced_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.NeymotinReduced.value["template"]))
		skeleton_cell = h.CP_Cell()

		return skeleton_cell

	def is_indexable(self, obj: object):
		"""
		Check if the object is indexable.
		"""
		try:
			_ = obj[0]
			return True
		except:
			return False
		
	def set_pickled_parameters_to_sections(self, sections: tuple, path: str):

		with open(path, 'rb') as file:
			params = pickle.load(file, encoding = 'latin1')

		for sec in sections:
			section_name = sec.name().split(".")[1]  # Remove Cell from name

			if "[" in section_name:
				section_type, section_type_index = section_name.split("[")
				section_type_index = section_type_index.strip("]")
				
				# Concatenate with "_"
				section_name_as_stored_in_pickle = f"{section_type}" #_{section_type_index}"
			else:
				# For sections like soma and axon
				section_name_as_stored_in_pickle = section_name  
		
			if section_name_as_stored_in_pickle in params['secs']:
				self.assign_parameters_to_section(sec, params['secs'][section_name_as_stored_in_pickle])
			else:
				raise ValueError(f"No parameters found for {section_name_as_stored_in_pickle}.")
					
	def assign_parameters_to_section(self, sec, section_data):

		# List of common state variables
		state_variables = []  # e.g. 'o_na', 'o_k', 'o_ca', 'm', 'h', 'n', 'i_na', ...
		
		# Initialize a dictionary for the section
		section_row = {'Section': sec.name()}
		
		# Set and record geometry parameters
		geom = section_data.get('geom', {})
		for param, value in geom.items():
			if str(param) not in ['pt3d']:
				setattr(sec, param, value)
				section_row[f"geom.{param}"] = value
		
		# Set and record ion parameters
		ions = section_data.get('ions', {})
		for ion, params in ions.items():
			for param, value in params.items():
				if param not in state_variables:
					main_attr_name = f"{ion}_ion"
					if param[-1] == 'o':
						sub_attr_name = f"{ion}{param}"
					else:
						sub_attr_name = f"{param}{ion}"
						for seg in sec:
							ion_obj = getattr(seg, main_attr_name)
							setattr(ion_obj, sub_attr_name, value)
					section_row[f"ions.{ion}.{param}"] = value
		
		# Set and record mechanism parameters
		mechs = section_data.get('mechs', {})
		for mech, params in mechs.items():
			if not hasattr(sec(0.5), mech):
				sec.insert(mech)
			for param, value in params.items():
				if param not in state_variables:
					for i, seg in enumerate(sec):
						if isinstance(value, list):
							try:
								setattr(seg, f"{param}_{mech}", value[i])
							except:
								print(f"Warning: Issue setting {mech} {param} in {seg} to {value[i]}. | value type: {type(value[i])} | nseg: {sec.nseg}; len(value): {len(value)}")
						else:
							try:
								setattr(seg, f"{param}_{mech}", value)
							except:
								print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value}. | value type {type(value)}")
		
					section_row[f"mechs.{mech}.{param}"] = value