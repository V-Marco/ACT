import h5py
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../Modules/")

from analysis import DataReader

def print_sim_file(h5file):
	with h5py.File(h5file, 'r') as file:
		data = np.array(file["data"])
	print(data)
	print(f"len: {len(data)}")
	try: print(f"shape: {data.shape}")
	except: pass

def print_sim_folder(sim_folder, sim_file_name):
	data_to_plot = DataReader.read_data(sim_folder, sim_file_name)
	print(data_to_plot)
	print(f"len: {len(data_to_plot)}")
	try: print(f"shape: {data_to_plot.shape}")
	except: pass


if __name__ == "__main__":

	if "-f" in sys.argv:
		sim_file = sys.argv[sys.argv.index("-f") + 1]
		print_sim_file(sim_file)
	
	if ("-d" in sys.argv) and ("-v" in sys.argv):
		sim_folder = sys.argv[sys.argv.index("-d") + 1]
		sim_file_name = sys.argv[sys.argv.index("-v") + 1]
		print_sim_folder(sim_folder, sim_file_name)

		
			
