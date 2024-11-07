import numpy as np
import pandas as pd
import os, h5py

from neuron import h

from recorder import SegmentRecorder, SynapseRecorder, SpikeRecorder, EmptySegmentRecorder
from recorder import SynapseRecorderList, SegmentRecorderList
from synapse import Synapse
from logger import Logger

from dataclasses import dataclass

@dataclass
class SegmentData:
	L: float
	membrane_surface_area: int
	coords: pd.DataFrame
	section: str
	index_in_section: int
	seg_half_seg_RA: float

class CellModel:

	FREQS = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, 'gamma': 30}

	def __init__(
			self, 
			skeleton_cell: object,
			random_state: np.random.RandomState,
			neuron_r: h.Random,
			logger: Logger):
		
		self.random_state = random_state
		self.neuron_r = neuron_r
		self.logger = logger
	
		# Morphology & Geometry (parse the hoc model)
		self.all = []
		self.soma = None
		self.apic = None
		self.dend = None
		self.axon = None
		for model_part in ["all", "soma", "apic", "dend", "axon"]:
			setattr(self, model_part, self._convert_section_list(getattr(skeleton_cell, model_part)))

		# Adjust the number of soma segments
		if self.soma[0].nseg != 1:
			self.logger.log(f"CellModel: changed soma nseg from {self.soma[0].nseg} to 1.")
			self.soma[0].nseg = 1

		# Adjust coordinates
		self._assign_sec_coords(random_state)

		# Connectivity
		self.synapses = []

		# Current Injection
		self.current_injection = None

		# Recorders
		self.recorders = []

	# ---------- HOC PARSING ----------

	def _convert_section_list(self, section_list: object) -> list:

		# If the section list is a hoc object, add its sections to the python list
		if str(type(section_list)) == "<class 'hoc.HocObject'>":
			new_section_list = [sec for sec in section_list]

		# Else, the section list is actually one section, add it to the list
		elif str(type(section_list)) == "<class 'nrn.Section'>":
			new_section_list = [section_list]

		# Python lists can also be passed
		elif str(type(section_list)) == "<class 'list'>":
			new_section_list = section_list
		
		else:
			raise TypeError(f"Expected input 'section_list' to be either of type hoc.HocObject, nrn.Section, or list, but got {type(section_list).__name__}")

		return new_section_list

	# ---------- COORDINATES ----------
	
	def _assign_sec_coords(self, random_state: np.random.RandomState) -> None:

		for sec in self.all:
			# Do only for sections without already having 3D coordinates
			if sec.n3d() != 0: continue

			# Store for a check later
			old_length = sec.L

			if sec is self.soma:
				new_length = self._assign_coordinates_to_soma_sec(sec)
			else:
				# Get the parent segment, sec
				pseg = sec.parentseg()
				if pseg is None: raise RuntimeError("Section {sec} is attached to None.")
				psec = pseg.sec

				# Process and get the new length
				new_length = self._assign_coordinates_to_non_soma_sec(sec, psec, pseg, random_state)
			
			if np.abs(new_length - old_length) >= 1: # Otherwise, it is a precision issue
				self.logger.log(f"Generation of 3D coordinates resulted in change of section length for {sec} from {old_length} to {sec.L}")

	def _assign_coordinates_to_soma_sec(self, sec: h.Section) -> float:
		sec.pt3dclear()
		sec.pt3dadd(*[0., -1 * sec.L / 2., 0.], sec.diam)
		sec.pt3dadd(*[0., sec.L / 2., 0.], sec.diam)
		return sec.L

	def _assign_coordinates_to_non_soma_sec(
			self, 
			sec: h.Section, 
			psec: h.Section, 
			pseg: object, 
			random_state: np.random.RandomState) -> float:
		
		# Get random theta and phi values for apical tuft and basal dendrites
		theta, phi = self._generate_phi_theta_for_apical_tuft_and_basal_dendrites(sec, random_state)

		# Find starting position using parent segment coordinates
		pt0 = self._find_starting_position_for_a_non_soma_sec(psec, pseg)

		# Calculate new coordinates using spherical coordinates
		xyz = [sec.L * np.sin(theta) * np.cos(phi), 
			   sec.L * np.cos(theta), 
			   sec.L * np.sin(theta) * np.sin(phi)]
		
		pt1 = [pt0[k] + xyz[k] for k in range(3)]

		sec.pt3dclear()
		sec.pt3dadd(*pt0, sec.diam)
		sec.pt3dadd(*pt1, sec.diam)

		return sec.L

	def _generate_phi_theta_for_apical_tuft_and_basal_dendrites(
			self, 
			sec: h.Section, 
			random_state: np.random.RandomState) -> tuple:
		
		if sec in self.apic:
			if sec != self.apic[0]: # Trunk
				theta, phi = random_state.uniform(0, np.pi / 2), random_state.uniform(0, 2 * np.pi)
			else:
				theta, phi = 0, np.pi/2
		elif sec in self.dend:
			theta, phi = random_state.uniform(np.pi / 2, np.pi), random_state.uniform(0, 2 * np.pi)
		else:
			theta, phi = 0, 0
		
		return theta, phi
	
	def _find_starting_position_for_a_non_soma_sec(self, psec: h.Section, pseg: object) -> list:
		for i in range(psec.n3d() - 1):
			arc_length = (psec.arc3d(i), psec.arc3d(i + 1)) # Before, After
			if (arc_length[0] / psec.L) <= pseg.x <= (arc_length[1] / psec.L):
				# pseg.x is between 3d coordinates i and i+1
				psec_x_between_coordinates = (pseg.x * psec.L - arc_length[0]) / (arc_length[1] - arc_length[0])

				#  Calculate 3d coordinates at psec_x_between_coordinates
				xyz_before = [psec.x3d(i), psec.y3d(i), psec.z3d(i)]
				xyz_after = [psec.x3d(i + 1), psec.y3d(i+1), psec.z3d(i + 1)]
				xyz = [xyz_before[k] + (xyz_after[k] - xyz_before[k]) * psec_x_between_coordinates for k in range(3)]
				break

		return xyz
	
	def get_coords_of_segments_in_section(self, sec) -> pd.DataFrame:

		seg_coords = np.zeros((sec.nseg, 13))

		seg_length = sec.L / sec.nseg
		arc_lengths = [sec.arc3d(i) for i in range(sec.n3d())]
		coords = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])

		seg_idx_in_sec = 0
		for seg in sec:
			start = seg.x * sec.L - seg_length / 2
			end = seg.x * sec.L + seg_length / 2
			mid = seg.x * sec.L
		
			for i in range(len(arc_lengths) - 1):
				# Check if segment's middle is between two 3D coordinates
				if (arc_lengths[i] <= mid < arc_lengths[i+1]) == False:
					continue

				t = (mid - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i])
				pt = coords[i] + (coords[i+1] - coords[i]) * t
	
				# Calculate the start and end points of the segment
				direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])

				# p0
				seg_coords[seg_idx_in_sec, 0:3] = pt - direction * seg_length / 2
				# Correct the start point if it goes before 3D coordinates
				while (i > 0) and (start < arc_lengths[i]):  # Added boundary check i > 0
					i -= 1
					direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
					seg_coords[seg_idx_in_sec, 0:3] = coords[i] + direction * (start - arc_lengths[i])

				# p05
				seg_coords[seg_idx_in_sec, 3:6] = pt

				# p1
				seg_coords[seg_idx_in_sec, 6:9] = pt + direction * seg_length / 2
	
				# Correct the end point if it goes beyond 3D coordinates
				while (end > arc_lengths[i+1]) and (i+2 < len(arc_lengths)):
					i += 1
					direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
					seg_coords[seg_idx_in_sec, 6:9] = coords[i] + direction * (end - arc_lengths[i])
	
				seg_coords[seg_idx_in_sec, 9] = seg.diam / 2
				seg_idx_in_sec += 1

		# Compute length (dl)
		seg_coords[:, 10:13] = seg_coords[:, 6:9] - seg_coords[:, 0:3]

		# Create a dataframe
		colnames = [f'p0_{x}' for x in range(3)] + [f'pc_{x}' for x in range(3)] + [f'p1_{x}' for x in range(3)]
		colnames = colnames + ['r'] + [f'dl_{x}' for x in range(3)]
		seg_coords = pd.DataFrame(seg_coords, columns = colnames)

		return seg_coords
	
	# ---------- SEGMENTS ----------

	def get_segments(self, section_names: list) -> tuple:
		segments = []
		datas = []

		for sec in self.all:
			if (sec.name().split(".")[1].split("[")[0] in section_names) or ("all" in section_names):
				for index_in_section, seg in enumerate(sec):
					data = SegmentData(
						L = seg.sec.L / seg.sec.nseg,
						membrane_surface_area = np.pi * seg.diam * (seg.sec.L / seg.sec.nseg),
						coords = self.get_coords_of_segments_in_section(sec).iloc[index_in_section, :].to_frame(1).T,
						section = sec.name(),
						index_in_section = index_in_section,
						seg_half_seg_RA = 0.01 * seg.sec.Ra * (sec.L / 2 / seg.sec.nseg) / (np.pi * (seg.diam / 2) ** 2)
					)
					segments.append(seg)
					datas.append(data)

		return segments, datas
	
	def get_seg_index(self, segment: object):
		indx = 0
		for sec in self.all:
			for seg in sec:
				if seg == segment: return indx
				indx += 1

	# ---------- SYNAPSES ----------
				
	def add_synapses_over_segments(
			self, 
			segments,
			nsyn,
			syn_mod,
			syn_params,
			gmax,
			name,
			density = False,
			seg_probs = None,
			release_p = None) -> None:
		
		total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
		if (density == True):
			nsyn = int(total_length * nsyn)
		
		if (seg_probs is None):
			seg_probs = [seg_length / total_length for seg_length in [seg.sec.L / seg.sec.nseg for seg in segments]]
		
		for _ in range(nsyn):
			segment = self.random_state.choice(segments, 1, True, seg_probs / np.sum(seg_probs))[0]

			if release_p is not None:
				if isinstance(release_p, dict):
					sec_type = segment.sec.name().split('.')[1][:4]
					p = release_p[sec_type](size = 1)
				else:
					p = release_p(size = 1) # release_p is partial

				pu = self.random_state.uniform(low = 0, high = 1, size = 1)
				
				# Drop synapses with too low release probability
				if p < pu: continue
			
			# Create synapse
			segment_distance = h.distance(segment, self.soma[0](0.5))
			if (isinstance(syn_params, tuple)) or ((isinstance(syn_params, list))):
				# Excitatory
				if 'AMPA' in syn_mod:
					syn_params = np.random.choice(syn_params, p = (0.9, 0.1))
				# Inhibitory
				elif 'GABA' in syn_mod:
					# Second option is for > 100 um from soma, else first option
					syn_params = syn_params[1] if segment_distance > 100 else syn_params[0]

			self.synapses.append(Synapse(
				segment = segment,
				syn_mod = syn_mod, 
				syn_params = syn_params, 
				gmax = gmax(size = 1) if callable(gmax) else gmax,
				neuron_r = self.neuron_r,
				name = name))
	
	def get_synapses(self, synapse_names: list):
		return [syn for syn in self.synapses if syn.name in synapse_names]
	
	# ---------- CURRENT INJECTION ----------

	def set_soma_injection(self, amp: float = 0, dur: float = 0, delay: float = 0):
		"""
		Add current injection to soma.
		"""
		self.current_injection = h.IClamp(self.soma[0](0.5))
		self.current_injection.amp = amp
		self.current_injection.dur = dur
		self.current_injection.delay = delay

	# ---------- MORPHOLOGY ----------

	def get_basals(self) -> list:
		return self.find_terminal_sections(self.dend)
	
	def get_tufts_obliques(self) -> tuple:
		tufts = []
		obliques = []
		for sec in self.find_terminal_sections(self.apic):
			if h.distance(self.soma[0](0.5), sec(0.5)) > 800:
				tufts.append(sec)
			else:
				obliques.append(sec)

		return tufts, obliques
	
	def get_nbranch(self) -> int:
		tufts, _ = self.get_tufts()
		basals = self.get_basals()
		return len(tufts) + len(basals) if len(tufts) == 1 else len(tufts) - 1 + len(basals)
	
	def find_terminal_sections(self, region: list) -> list:
		'''
		Finds all terminal sections by iterating over all sections and returning those which are not parent sections.
		'''
		# Find non-terminal sections
		parent_sections = []
		for sec in self.all:
			if sec.parentseg() is None:
				continue
			
			if sec.parentseg().sec not in parent_sections:
				parent_sections.append(sec.parentseg().sec)
			
		terminal_sections = []
		for sec in region:
			if (sec not in parent_sections):
				terminal_sections.append(sec)

		return terminal_sections
	
	def compute_electrotonic_distance(self, from_segment) -> pd.DataFrame:
		passive_imp = h.Impedance()
		passive_imp.loc(from_segment)
		active_imp = h.Impedance()
		active_imp.loc(from_segment)
		
		segments, _ = self.get_segments(["all"])
		elec_distance = np.zeros((len(segments), 2 * len(self.FREQS.items())))

		colnames = []
		col_idx = 0
		for freq_name, freq_hz in self.FREQS.items():
			# 9e-9 is a Segev's value
			passive_imp.compute(freq_hz + 9e-9, 0)
			active_imp.compute(freq_hz + 9e-9, 1)
			for i, seg in enumerate(segments):
				elec_distance[i, col_idx] = active_imp.ratio(seg.sec(seg.x))
				elec_distance[i, col_idx + 1] = passive_imp.ratio(seg.sec(seg.x))
			colnames.append(f"{freq_name}_active")
			colnames.append(f"{freq_name}_passive")
			col_idx = col_idx + 2

		return pd.DataFrame(elec_distance, columns = colnames)
	
	# ---------- RECORDERS ----------

	def add_spike_recorder(self, sec: object, var_name: str, spike_threshold: float):
		self.recorders.append(SpikeRecorder(sec = sec, var_name = var_name, spike_threshold = spike_threshold))
	
	def add_synapse_recorders(self, var_name: str) -> None:
		rec_list = SynapseRecorderList(var_name)
		for syn in self.synapses:
			try: rec_list.add(SynapseRecorder(syn.h_syn, var_name))
			except: continue
		self.recorders.append(rec_list)

	def add_segment_recorders(self, var_name: str) -> None:
		rec_list = SegmentRecorderList(var_name)
		segments, _ = self.get_segments(["all"])
		for seg in segments:
			try: rec_list.add(SegmentRecorder(seg, var_name))
			except: rec_list.add(EmptySegmentRecorder())
		self.recorders.append(rec_list)
	
	def write_recorder_data(self, path: str, step: int) -> None:
		os.mkdir(path)

		for recorder in self.recorders:
			if type(recorder) == SpikeRecorder:
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.vec.as_numpy().reshape(1, -1))

			elif (type(recorder) == SegmentRecorder) or (type(recorder) == SynapseRecorder):
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.vec.as_numpy()[::step].reshape(1, -1))

			elif (type(recorder) == SegmentRecorderList):
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.get_combined_data()[:, ::step])

			elif (type(recorder) == SynapseRecorderList):
				segments, _ = self.get_segments(["all"])
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.get_combined_data(segments, self.synapses)[:, ::step])
	
	def _write_datafile(self, reportname, data):
		with h5py.File(reportname, 'w') as file:
			file.create_dataset("data", data = data)
