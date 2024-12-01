# needs an update to plot axon voltage based on constants.seg_to_record
import sys
sys.path.append("../")

import numpy as np
import h5py, os
import matplotlib.pyplot as plt
import importlib



# Output folder should store folders 'saved_at_step_xxxx'
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neymotin"

import importlib
def load_constants_from_folder(output_folder):
    # Get the absolute path to the output_folder
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = os.path.join(current_script_path, output_folder)
    
    # List all folders in the output_folder
    all_folders = [f for f in os.listdir(absolute_path) if os.path.isdir(os.path.join(absolute_path, f))]
    
    # Sort the folders (optional)
    all_folders.sort()
    
    # Iterate over sorted folders to find the first one that contains 'constants_image.py'
    for folder in all_folders:
        folder_path = os.path.join(absolute_path, folder)
        
        # Check if 'constants_image.py' exists in this folder
        if 'constants_image.py' in os.listdir(folder_path):
            # Update the absolute path to point to this folder
            absolute_path = folder_path
            
            sys.path.append(absolute_path)
            
            # Import the constants module
            try:
                constants_module = importlib.import_module('constants_image')
            except ModuleNotFoundError:
                print(f"Failed to import constants_image from {absolute_path}")
                sys.path.remove(absolute_path)
                return None
            
            sys.path.remove(absolute_path)
            return constants_module
    
    print("No folders found containing constants_image.py in output directory.")
    return None
    
constants = load_constants_from_folder(output_folder)

skip = constants.h_i_delay#400 # (ms)

def main():
    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

    firing_rates = []

    for ampl in constants.h_i_amplitudes:
        spikes = []
        Vm = []
        t = []
        #print("amplitude: ", ampl)
        #print(f"_{int(ampl * 1000)}")
        for ampl_dir in os.listdir(output_folder): # list folders in directory
            if (ampl_dir.endswith(f"_{int(ampl * 1000)}")): # go over all amplitudes
                print(ampl, ampl_dir)
                #print(step_size)
                for step in steps:
                    dirname = os.path.join(output_folder, ampl_dir, f"saved_at_step_{step}")
                    with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
                        Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
                    with h5py.File(os.path.join(dirname, "t.h5")) as file:
                        t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
                    with h5py.File(os.path.join(dirname, "spikes_report.h5")) as file:
                        spikes.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
        t = np.hstack(t) # (ms)
        Vm = np.hstack(Vm)
        spikes = np.hstack(spikes)
        #print("spikes:", spikes)
        plt.figure(figsize = (7,8))
        if constants.seg_to_record == 'axon':
          if constants.build_ziao_cell:
            seg_index = 194
          elif constants.build_cell_reports_cell:
            seg_index = 1#194
          else:
            raise('find axon seg index (can be done using exam_axial_currents)')
          title_prefix = "Axon"
        else:
          seg_index=0
          title_prefix = "Soma"
        #seg_index=0
        plt.plot(t, Vm[seg_index])
        for spike in spikes:
          plt.scatter(spike, 30, color = 'black', marker='*')
        plt.xlabel("Time (ms)")
        plt.ylabel("Vm (mV)")
        plt.title(f"{title_prefix} Vm at {ampl}")
        plt.savefig(os.path.join(output_folder, f"{title_prefix}_{str(ampl)}.png"))
        plt.close()   
        print(len(spikes))
        print(len(spikes[(spikes > skip) & (spikes < (skip + constants.h_i_duration))]))
        firing_rate = len(spikes[(spikes > skip) & (spikes < (skip + constants.h_i_duration))]) / (constants.h_i_duration / 1000)
        firing_rates.append(firing_rate)

    # Save FI curve
    plt.figure(figsize = (7, 8))
    plt.plot(constants.h_i_amplitudes, firing_rates, color='blue')
    plt.scatter(constants.h_i_amplitudes, firing_rates, color='r', marker='*')
    plt.xlabel("Amplitude (nA)")
    plt.ylabel("Hz")
    plt.title(f"{constants.seg_to_record} FI curve")
    plt.savefig(os.path.join(output_folder, f"FI_{constants.seg_to_record}.png"))

    # Save firing rates
    with open(os.path.join(output_folder, "firing_rates.csv"), "a") as file:
        for i in range(len(constants.h_i_amplitudes)):
            file.writelines(f"{constants.h_i_amplitudes[i]},{firing_rates[i]}\n")

if __name__ == "__main__":
    main()
