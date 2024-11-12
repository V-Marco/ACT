from neuron import h
import numpy as np

# Reinitialize vectors: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=2579

class SynapseRecorderList:

	def __init__(self, var_name: str) -> None:
		self.var_name = var_name
		self.recorders = []

	def add(self, recorder) -> None:
		if type(recorder) != SynapseRecorder:
			raise ValueError
		self.recorders.append(recorder)

	def get_combined_data(self, segments, synapses) -> np.ndarray:
		data = np.zeros((len(segments), len(self.recorders[0].vec.as_numpy())))
		for syn_ind, rec in enumerate(self.recorders):
			seg_ind = segments.index(synapses[syn_ind].h_syn.get_segment())
			data[seg_ind, :] += rec.vec.as_numpy()
		return data
	
	def clear(self):
		for rec in self.recorders:
			rec.vec.resize(0)

class SegmentRecorderList:

	def __init__(self, var_name: str) -> None:
		self.var_name = var_name
		self.recorders = []

	def add(self, recorder) -> None:
		if (type(recorder) != SegmentRecorder) and (type(recorder) != EmptySegmentRecorder):
			raise ValueError
		self.recorders.append(recorder)

	def get_combined_data(self) -> np.ndarray:
		# Find the first non-empty recorder and get its length; otherwise produce vectors of length 1
		data_length = 1
		for rec in self.recorders:
			if type(rec) != EmptySegmentRecorder:
				data_length = len(rec.vec.as_numpy())

		data = np.zeros((len(self.recorders), data_length))
		for i, rec in enumerate(self.recorders):
			if type(rec) == EmptySegmentRecorder:
				data[i, :] = np.nan
			else:
				data[i, :] = rec.vec.as_numpy()
		return data
	
	def clear(self):
		for rec in self.recorders:
			if type(rec) != EmptySegmentRecorder:
				rec.vec.resize(0)

class EmptySegmentRecorder: pass

class SegmentRecorder:

	def __init__(self, seg: object, var_name: str):
		self.var_name = var_name
		self.vec = h.Vector()

		attr = getattr(seg, '_ref_' + var_name)
		self.vec.record(attr)

	def clear(self):
		self.vec.resize(0)

class SynapseRecorder:

	def __init__(self, syn: object, var_name: str):
		self.var_name = var_name
		self.vec = h.Vector()

		attr = getattr(syn, '_ref_' + var_name)
		self.vec.record(attr)
	
	def clear(self):
		self.vec.resize(0)

class SpikeRecorder:

	def __init__(self, sec: object, var_name: str, spike_threshold: float):
		self.var_name = var_name
		self.vec = h.Vector()
		self.spike_threhsold = spike_threshold

		nc = h.NetCon(sec(0.5)._ref_v, None, sec = sec)
		nc.threshold = spike_threshold
		nc.record(self.vec)

	def clear(self):
		self.vec.resize(0)
