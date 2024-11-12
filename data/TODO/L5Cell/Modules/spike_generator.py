import numpy as np
from scipy.signal import lfilter
import warnings

def minmax(x):
	if np.min(x) == np.max(x):
		return x
	return (x - np.min(x)) / (np.max(x) - np.min(x))

class SpikeTrain:

	def __init__(self, spike_times, T):
		self.spike_times = spike_times
		self.T = T

	def to_continuous(self):
		cv = np.zeros(self.T)
		cv[self.spike_times] = 1
		return cv

class PoissonTrainGenerator:
	# Win_size = 1 ms

	@staticmethod
	def generate_lambdas_from_pink_noise(
			num: int,
			random_state: np.random.RandomState,
			lambda_mean: float = 1.0,
			rhythmic_modulation: bool = False) -> np.ndarray:
		
		# Get lambdas
		lambdas = PoissonTrainGenerator.generate_pink_noise(num_obs = num, random_state = random_state, mean = lambda_mean)

		# Apply modulation
		if rhythmic_modulation:
			lambdas = PoissonTrainGenerator.rhythmic_modulation(lambdas)
		
		# Can't have negative firing rates (precision reasons)
		if np.sum(lambdas < 0) != 0:
			warnings.warn("Found non-positive lambdas when generating a spike train.")
			lambdas[lambdas < 0] = 0 
		
		return lambdas
	
	@staticmethod
	def generate_pink_noise(
			random_state: np.random.RandomState, 
			num_obs: int, 
			mean: float = 1,
			std: float = 0.5,
			bounds: tuple = (0.5, 1.5)) -> np.ndarray:
		'''
		Produce pink ("1/f") noise out of the white noise.
		The idea is to generate a white noise and then filter it to impose autocovariance structure with
		1/f psd. The filter used here is the scipy.signal's FIR / IIR filter which replaces observation t
		with an AR(k) sum of the previous k unfiltered AND filtered observations.

		The resulting pink noise is minmaxed and shifted to be in [shift[0], shift[1]] region.

		Parameters:
		----------
		num_obs: int
			Length of the profile.

		A: list[float]
			AR coefficients of the filtered observations.

		B: list[float]
			AR coefficients of the unfiltered (original) observations.

		bounds: tuplew
			The profile bounds: [min, max]. 

		Returns:
		----------
		fr_profile: np.ndarray
			Firing rate profile.
		'''
		# These values produce stable pink noise
		A = [1, -2.494956002, 2.017265875, -0.522189400]
		B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
		
		white_noise = random_state.normal(loc = mean, scale = std, size = num_obs + 2000)

		# Apply the FIR/IIR filter to create the 1/f noise, minmax and shift to bounds
		fr_profile = minmax(lfilter(B, A, white_noise)[2000:]) * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile
	
	@staticmethod
	def delay_spike_trains(
			spike_trains_to_delay: list,
			shift: int = 4) -> list:
		
		delayed_trains = []
		for train in spike_trains_to_delay:
			delayed_times = train.spike_times - shift
			delayed_times[delayed_times < 0] = 0
			delayed_trains.append(SpikeTrain(delayed_times, train.T))

		return delayed_trains

	@staticmethod
	def rhythmic_modulation(lambdas: np.ndarray):
		t = np.linspace(0, 1, len(lambdas))
		lambdas = lambdas + lambdas * np.sin(2 * np.pi * t)
		return lambdas
	
	@staticmethod
	def generate_spike_train(
			lambdas: np.ndarray,
			random_state: np.random.RandomState) -> np.ndarray:
		
		t = np.zeros(len(lambdas))
		for i, lambd in enumerate(lambdas):
			# Because we are using win-size = 1 ms; lambda = num_spikes_per_second * win_length / 1000
			num_points = random_state.poisson(lambd / 1000)
			if num_points > 0: t[i] = 1

		spike_times = np.where(t > 0)[0]
		return SpikeTrain(spike_times, len(lambdas))
