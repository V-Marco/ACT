import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../")

from Modules.plotting_utils import plot_simulation_results
from Modules.constants import SimulationParameters
from Modules.logger import Logger

from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces
from cell_inference.config import params
from multiprocessing import Pool, cpu_count

import pickle

def wrap_plot_lfp(parameters: SimulationParameters):

	step_size = int(parameters.save_every_ms / parameters.h_dt) # Timestamps
	steps = range(step_size, int(parameters.h_tstop / parameters.h_dt) + 1, step_size) # Timestamps

	t = []
	Vm = []
	lfp = []

	for step in steps:
		dirname = os.path.join(parameters.path, f"saved_at_step_{step}")
		with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
			Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
		with h5py.File(os.path.join(dirname, "lfp.h5")) as file:
			lfp.append(np.array(file["report"]["biophysical"]["data"])[:step_size, :])
		with h5py.File(os.path.join(dirname, "t.h5")) as file:
			t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])

	Vm = np.hstack(Vm)
	lfp = np.vstack(lfp)
	t = np.hstack(t) # (ms)

	with open(os.path.join(parameters.path, "seg_indexes.pickle"), "rb") as file:
		seg_indexes = pickle.load(file)
	
	loc_param = [0., 0., 45., 0., 1., 0.]
	elec_pos = params.ELECTRODE_POSITION

	plot_simulation_results(
		t, 
		Vm, 
		seg_indexes['soma'], 
		seg_indexes['axon'], 
		seg_indexes['basal'], 
		seg_indexes['tuft'], 
		seg_indexes['nexus'], 
		seg_indexes['trunk'], 
		loc_param, 
		lfp, 
		elec_pos, 
		plot_lfp_heatmap, 
		plot_lfp_traces, 
		vlim = [-0.023, 0.023],
		show = False,
		save_dir = parameters.path)


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
	pool.map(wrap_plot_lfp, jobs)
	pool.close()
	pool.join()

	
