# get clamp current from files
# compute mean and std of maximum current magnitude

# will need to not count magnitudes of 0.

import sys
sys.path.append("../")

import numpy as np
import h5py, os
#import constants
import matplotlib.pyplot as plt

from multiprocessing import Process

import os
import sys
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

# Output folder should store folders 2023...
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/PSC_2023-08-23_21-56-54"

def process_cluster(cluster_index, magnitudes, skip, step_size, steps):
    spikes = []
    Vm = []
    t = []
    clamp_current = []

    # Track if there are multiple found for a given cluster_index
    matching_filenames = 0
    print("Looking for cluster: ", cluster_index)

    for cluster_dir in os.listdir(output_folder): # List folders in directory
        if cluster_dir.endswith(f"_cluster{cluster_index}"): # go over all amplitudes
            print(cluster_index, cluster_dir)
            matching_filenames += 1
            if matching_filenames > 1: raise(ValueError(f"Multiple filenames found that end with '_cluster{cluster_index}'"))
            full_cluster_dir = os.path.join(output_folder, cluster_dir)  # Full path
            if len(os.listdir(full_cluster_dir)) == 0:
                print(f"{cluster_index} did not have synapses")
            else:
                for step in steps:
                    dirname = os.path.join(output_folder, cluster_dir, f"saved_at_step_{step}")
                    with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
                        Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
                    with h5py.File(os.path.join(dirname, "t.h5")) as file:
                        t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
                    with h5py.File(os.path.join(dirname, "spikes_report.h5")) as file:
                        spikes.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
                    for filename in os.listdir(dirname):
                        if filename.endswith('_Vclamp_i.h5'):
                            clamp_filename=filename
                            with h5py.File(os.path.join(dirname, filename)) as file:
                                clamp_current.append(np.array(file["report"]["biophysical"]["data"])[:step_size])   
        
        
                    t = np.hstack(t) # (ms)
                    Vm = np.hstack(Vm)
                    clamp_current = np.hstack(clamp_current)
                        
                    #inact skip
                    t=t[skip:]
                    Vm=Vm[0][skip:]
                    clamp_current = clamp_current[skip:]
                        
                    # update to plot voltage from simulations.
                    spikes = np.hstack(spikes)
                    #print("spikes:", spikes)
                    #plt.figure(figsize = (7,8))
                    titles = [
                        'Soma Membrane Voltage. PSC: [{}]',
                        'Voltage Clamp Current: [{}]'
                    ]
                    ylabels= ['mV', 'nA']
                    to_plot = [Vm, clamp_current]
                    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
                    #for spike in spikes:
                    #  plt.scatter(spike, 30, color = 'black', marker='*')
                    #plt.xlabel("Time (ms)")
                    #plt.ylabel("Vm (mV)")
                    #plt.title(str(ampl))
                    #plt.savefig(str(ampl)+".png")
                    #plt.close()   
                    #firing_rate = len(spikes[spikes > skip]) / (constants.h_tstop / 1000)
                    
                    # sort PSC
                    if 'dend' in clamp_filename:
                        sec_type = 'dend'
                        if 'exc' in clamp_filename:
                            syn_type = 'exc'
                            magnitudes["basal"]['exc'].append(max(abs(clamp_current)))
                        elif 'inh' in clamp_filename:
                            syn_type = 'inh'
                            magnitudes["basal"]['inh'].append(max(abs(clamp_current)))
                        else:
                            raise(ValueError("Neither 'exc' nor 'inh' in clamp_filename"))
                    elif ('soma' in clamp_filename) or ('True' in clamp_filename):
                        sec_type = 'soma'
                        if 'exc' in clamp_filename:
                            syn_type = 'exc'
                            magnitudes["somatic"]['exc'].append(max(abs(clamp_current)))
                        elif 'inh' in clamp_filename:
                            syn_type = 'inh'
                            magnitudes["somatic"]['inh'].append(max(abs(clamp_current)))
                        else:
                            raise(ValueError("Neither 'exc' nor 'inh' in clamp_filename"))
                    elif 'apic' in clamp_filename:
                        sec_type = 'apic'
                        if 'exc' in clamp_filename:
                            syn_type = 'exc'
                            magnitudes["apical"]['exc'].append(max(abs(clamp_current)))
                        elif 'inh' in clamp_filename:
                            syn_type = 'inh'
                            magnitudes["apical"]['inh'].append(max(abs(clamp_current)))
                        else:
                            raise(ValueError("Neither 'exc' nor 'inh' in clamp_filename"))
                    else:
                        raise(ValueError("Could not find 'dend', 'apic', or 'soma' in clamp_filename {clamp_filename}"))
                    
                    title_suffix = str(cluster_index)+'_' + str(syn_type) + '_' + str(sec_type)
        
                    for j, ax in enumerate(axs):
                        title = titles[j].format(title_suffix)
                        ylabel = ylabels[j]
                        ax.plot(t, to_plot[j])
                        ax.set_ylabel(ylabel)
                        ax.set_xlabel('time (ms)')
                        ax.set_title(title)
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_folder, cluster_dir, title_suffix+"_PSC.png"))#title_prefix + "Combined_plot" + "_{}".format(index) + ".png"))
                    plt.close(fig)

def main():
    # Dictionary for sorting PSCs
    magnitudes={}
    magnitudes["basal"] = {}
    magnitudes["somatic"] = {}
    magnitudes["apical"] = {}
    magnitudes["basal"]["exc"] = []
    magnitudes["basal"]["inh"] = []
    magnitudes["somatic"]["exc"] = []
    magnitudes["somatic"]["inh"] = []
    magnitudes["apical"]["exc"] = []
    magnitudes["apical"]["inh"] = []

    # Use the function
    constants = load_constants_from_folder(output_folder)
    constants.h_tstop = 55
    constants.save_every_ms = 55
    constants.log_every_ms = 55
    constants.parallelize = True
    skip = int(constants.PSC_start/constants.h_dt) # (ms)
    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
    pool = []
    for cluster_index in range(1, constants.number_of_presynaptic_cells + 1):
        if constants.parallelize:
            pool.append(Process(target=process_cluster, args=[cluster_index, magnitudes, skip, step_size, steps]))

        else:
            process_cluster(cluster_index, magnitudes, skip, step_size, steps)

    if constants.parallelize:
        for p in pool: p.start()
        for p in pool: p.join()
        for p in pool: p.terminate()
            
    # Calculate the total number of subplots required (total combinations of region and syn_type)
    subplot_count = len(magnitudes) * len(magnitudes[next(iter(magnitudes))])

    # Create the figure and the subplots
    fig, axs = plt.subplots(subplot_count, 1, figsize=(10, 5 * subplot_count))

    # Initialize subplot index
    subplot_index = 0

    for region in magnitudes.keys():
        for syn_type in magnitudes[region].keys():
            # Calculate mean and standard deviation
            mean = np.mean(magnitudes[region][syn_type])
            std = np.std(magnitudes[region][syn_type])
            # Save to file
            with open(os.path.join(output_folder, "PSC_Magnitudes.csv"), "a") as file:
                for i in range(len(constants.h_i_amplitudes)):
                    file.writelines(f"{region},{syn_type},{mean},{std}\n")
                
            # Prepare data for histogram
            data = magnitudes[region][syn_type]
            
            # Plot a histogram for the data
            axs[subplot_index].hist(data, bins=50)  
            axs[subplot_index].set_xlabel('Magnitude')
            axs[subplot_index].set_ylabel('Frequency')
            axs[subplot_index].set_title('Histogram of Magnitudes: Region {} Syn_Type {}'.format(region, syn_type))
            
            # increment subplot index
            subplot_index += 1

    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(output_folder, "PSC_Magnitudes_All.png"))
    plt.close(fig)
#            # calculate mean and standard deviation
#            for region in magnitudes.keys():
#              for syn_type in magnitudes[region].keys():
#                mean = np.mean(magnitudes[region][syn_type])
#                std = np.std(magnitudes[region][syn_type])
#                #save to file
#                with open(os.path.join(output_folder, "PSC_Magnitudes.csv"), "a") as file:
#                  for i in range(len(constants.h_i_amplitudes)):
#                      file.writelines(f"{region},{syn_type},{mean},{std}\n")
#                title_suffix = str(region) + '_' + str(syn_type)
#                title = 'Distribution of magnitudes of PSCs: [{}]'.format(title_suffix)
#                to_plot = magnitudes[region][syn_type]
#                fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
#                title_suffix = str(cluster_index)+'_'+str(syn_type)+'_'+str(sec_type)
#                for j, ax in enumerate(axs):
#                title = titles[j].format(title_suffix)
#                ylabel = ylabels[j]
#                ax.plot(t, to_plot[j])
#                ax.set_ylabel(ylabel)
#                ax.set_xlabel('time (ms)')
#                ax.set_title(title)
#                plt.tight_layout()
#                fig.savefig(os.path.join(output_folder, cluster_dir, title_suffix+"_PSC.png"))#title_prefix + "Combined_plot" + "_{}".format(index) + ".png"))
#                plt.close(fig)
#    # Save FI curve # update to plot PSC distribution
#    plt.figure(figsize = (7, 8))
#    plt.plot(constants.h_i_amplitudes, firing_rates)
#    plt.xlabel("Amplitude (nA)")
#    plt.ylabel("Hz")
#    plt.savefig(os.path.join(output_folder, f"FI.png"))

if __name__ == "__main__":
	main()