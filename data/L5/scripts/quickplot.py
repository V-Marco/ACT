import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../Modules/")

from analysis import DataReader

def plot_sim_file(h5file):
	with h5py.File(h5file, 'r') as file:
		data = np.array(file["data"])
	plt.plot(data)
	plt.savefig(f"quickplot.png")

def plot_sim_folder(sim_folder, sim_file_name):
	data_to_plot = DataReader.read_data(sim_folder, sim_file_name)
	plt.plot(data_to_plot)
	plt.savefig(f"quickplot.png")


if __name__ == "__main__":

	if "-f" in sys.argv:
		sim_file = sys.argv[sys.argv.index("-f") + 1]
		plot_sim_file(sim_file)
	
	if ("-d" in sys.argv) and ("-v" in sys.argv):
		sim_folder = sys.argv[sys.argv.index("-d") + 1]
		sim_file_name = sys.argv[sys.argv.index("-v") + 1]
		plot_sim_folder(sim_folder, sim_file_name)

		
			
