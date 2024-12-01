import sys
sys.path.append("../")


from Modules.complex_cell import build_L5_cell, build_L5_cell_ziao, build_cell_reports_cell, unpickle_params, inspect_pickle, set_hoc_params, adjust_soma_and_axon_geometry
from Modules.complex_cell import build_cell_reports_cell, assign_parameters_to_section, create_cell_from_template_and_pickle
from Modules.cell_utils import get_segments_and_len_per_segment
from Modules.logger import Logger
from Modules.recorder import Recorder
from Modules.reduction import Reductor
from Modules.cell_model import CellModel

from cell_inference.config import params as ecp_params
from cell_inference.utils.currents.ecp import EcpMod

import numpy as np
from functools import partial
import scipy.stats as st
import time, datetime
import os, h5py, pickle, shutil
from multiprocessing import Process
import pandas as pd

from neuron import h
h.load_file("stdrun.hoc")

import importlib
import constants  # your constants module
importlib.reload(constants)

from Build_M1_cell.build_M1_cell import build_m1_cell


def main(numpy_random_state, neuron_random_state, logger, i_amplitude=None):

    print(f"Running for seeds ({numpy_random_state}, {neuron_random_state}); CI = {i_amplitude}...")

    # Random seed
    logger.log_section_start(f"Setting random states ({numpy_random_state}, {neuron_random_state})")

    random_state = np.random.RandomState(numpy_random_state)
    np.random.seed(numpy_random_state)

    neuron_r = h.Random()
    neuron_r.MCellRan4(neuron_random_state)

    logger.log_section_end("Setting random states")

    logger.log(f"Amplitude is set to {i_amplitude}")

    # Time vector for generating inputs
    t = np.arange(0, constants.h_tstop, 1)

    # Build cell
    logger.log_section_start("Building complex cell")
    # decide which cell to build
    if constants.build_m1:
        complex_cell = build_m1_cell() # use older Neymotin detailed cell template
    elif constants.build_ziao_cell:
        complex_cell = build_L5_cell_ziao(constants.complex_cell_folder) # build Neymotin reduced from ziao template
    elif constants.build_cell_reports_cell: # build Neymotin detailed cell from template and pickled params # *********** current use mainly
        complex_cell = create_cell_from_template_and_pickle()
    else: # Build Hay et al model then replace axon & soma with Neymotin detailed
        complex_cell = build_L5_cell(constants.complex_cell_folder, constants.complex_cell_biophys_hoc_name)
        adjust_soma_and_axon_geometry(complex_cell, somaL = constants.SomaL, somaDiam = constants.SomaDiam, axonDiam = constants.AxonDiam, axonL = constants.AxonL, axon_L_scale = constants.Axon_L_scale)
        set_hoc_params()
        
    logger.log_section_end("Building complex cell")

    h.celsius = constants.h_celcius
    
    try:h.v_init = complex_cell.soma[0].e_pas
    except:
      h.v_init = complex_cell.soma.e_pas
      #print(f"warning soma is h.Section {complex_cell.soma} and not list")

    # Sim runtime
    h.tstop = constants.h_tstop

    # Timestep (ms)
    h.dt = constants.h_dt
    h.steps_per_ms = 1 / h.dt

    # Measure time
    runtime_start_time = time.time()

    # init cell
    logger.log_section_start("Initializing Reductor and cell model for simulation |NR:"+str(constants.reduce_cell)+"|optimize nseg:"+str(constants.optimize_nseg_by_lambda)+"|Expand Cable:"+str(constants.expand_cable))
    logger.log_memory()
    reductor = Reductor()
    
    cell = reductor.reduce_cell(complex_cell = complex_cell, reduce_cell = constants.reduce_cell, 
                                optimize_nseg = constants.optimize_nseg_by_lambda, synapses_list = [],
                                netcons_list = [], spike_trains = [],
                                spike_threshold = constants.spike_threshold, random_state = random_state,
                                var_names = constants.channel_names, reduction_frequency = constants.reduction_frequency, 
                                expand_cable = constants.expand_cable, choose_branches = constants.choose_branches, seg_to_record= constants.seg_to_record)
    
    logger.log_section_start("Setting up cell var recorders")
    logger.log_memory()
    cell.setup_recorders(vector_length = constants.save_every_ms)
    logger.log_section_end("Setting up cell var recorders")
    # Add injections for F/I curve
    if i_amplitude is not None:
        cell.add_injection(sec_index = cell.all.index(cell.soma[0]), record = True, delay = constants.h_i_delay, dur = constants.h_i_duration, amp = i_amplitude)

    # ---- Prepare simulation

    logger.log_section_start("Finding segments of interest")

    # find segments of interest
    soma_seg_index = cell.segments.index(cell.soma[0](0.5))
    axon_seg_index = cell.segments.index(cell.axon[-1](0.9))
    basal_seg_index = cell.segments.index(cell.basals[0](0.5))
    trunk_seg_index = cell.segments.index(cell.apic[0](0.999))
    # find tuft and nexus
    if (constants.reduce_cell == True) and (constants.expand_cable == True): # Dendritic reduced model
        tuft_seg_index = tuft_seg_index=cell.segments.index(cell.tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        nexus_seg_index = cell.segments.index(cell.apic[0](0.99))
    elif (constants.reduce_cell == True) and (constants.expand_cable == False): # NR model
        tuft_seg_index = cell.segments.index(cell.tufts[0](0.9)) # tufts[0] will be the cable that is both trunk and tuft in this case, so we have to specify near end of cable
        nexus_seg_index = cell.segments.index(cell.apic[0](0.289004))
#    else: # Complex cell
#        tuft_seg_index=cell.segments.index(cell.tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
#        nexus_seg_index=cell.segments.index(cell.apic[36](0.961538))
    seg_indexes = {
        "soma": soma_seg_index,
        "axon": axon_seg_index,
        "basal": basal_seg_index,
        "trunk": trunk_seg_index,
#        "tuft": tuft_seg_index,
#        "nexus": nexus_seg_index
    }
    logger.log_section_end("Finding segments of interest")
    
    # Compute electrotonic distances from nexus
    logger.log_section_start("Recomputing elec distance")

#    cell.recompute_segment_elec_distance(segment = cell.segments[nexus_seg_index], seg_name = "nexus")

    logger.log_section_end("Recomputing elec distance")

    logger.log_section_start("Initializing t_vec and V_rec recorder")

    # Record time points
    t_vec = h.Vector(1000 / h.dt).record(h._ref_t)

    # Record membrane voltage of all segments
    V_rec = Recorder([cell.soma[0](0.5)], vector_length = constants.save_every_ms)

    logger.log_section_end("Initializing t_vec and V_rec recorder")

    logger.log_section_start("Creating ecp object")

    elec_pos = ecp_params.ELECTRODE_POSITION
    ecp = EcpMod(cell, elec_pos, min_distance = ecp_params.MIN_DISTANCE)  # create an ECP object for extracellular potential

    logger.log_section_end("Creating ecp object")

    # ---- Run simulation
    sim_duration = h.tstop / 1000 # Convert from ms to s

    logger.log_section_start(f"Running sim for {sim_duration} sec")
    logger.log_memory()

    sim_start_time = time.time()

    time_step = 0 # In time stamps, i.e., ms / dt
    time_steps_saved_at = [0]

    # Create a folder to save to#datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_seeds_" +\
    random_seed_name = "_seeds_" + \
                       str(numpy_random_state) + "_" + str(neuron_random_state) + cell.get_output_folder_name()
    if i_amplitude is not None:
        random_seed_name += f"_{int(i_amplitude * 1000)}"
    save_folder = os.path.join(constants.save_dir, random_seed_name)
    os.mkdir(save_folder)

    # Save indexes for plotting
    with open(os.path.join(save_folder, "seg_indexes.pickle"), "wb") as file:
        pickle.dump(seg_indexes, file)
        
    # Save biophys
    #biophys_path = os.path.join(constants.complex_cell_folder, constants.biophys_hoc_name)
    #shutil.copy2(biophys_path, save_folder)
    #os.rename(os.path.join(save_folder, biophys_path), os.path.join(save_folder, constants.complex_cell_biophys_hoc_name.split('.')[0]+'_image.hoc'))
    biophys_path = os.path.join(constants.complex_cell_folder, constants.complex_cell_biophys_hoc_name)
    # Copy the file
    if os.path.exists(biophys_path):
        try:
            shutil.copy2(biophys_path, save_folder)
        except Exception as e:
            print(f"An error occurred while copying: {e}")
    else:
        print(f"The file {biophys_path} does not exist.")
    
    # Rename the copied file
    try:
        destination_path = os.path.join(save_folder, constants.complex_cell_biophys_hoc_name)
        new_name = os.path.join(save_folder, constants.complex_cell_biophys_hoc_name.split('.')[0] + '_image.hoc')
        os.rename(destination_path, new_name)
    except Exception as e:
        print(f"An error occurred while renaming: {e}")
        
    
    # Save constants
    shutil.copy2("constants.py", save_folder)
    os.rename(os.path.join(save_folder, "constants.py"), os.path.join(save_folder, "constants_image.py"))

    h.finitialize(h.v_init)
    while h.t <= h.tstop + 1:

        if time_step % (constants.log_every_ms / constants.h_dt) == 0:
            logger.log(f"Running simulation step {time_step}")
            logger.log_memory()

        if (time_step > 0) & (time_step % (constants.save_every_ms / constants.h_dt) == 0):
            # Save data
            cell.generate_recorder_data(constants.save_every_ms)
            cell.write_data(os.path.join(save_folder, f"saved_at_step_{time_step}"))

            # Save lfp
            loc_param = [0., 0., 45., 0., 1., 0.]
            #lfp = ecp.calc_ecp(move_cell = loc_param).T  # Unit: mV

            #with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "lfp.h5"), 'w') as file:
            #    file.create_dataset("report/biophysical/data", data = lfp)
            # save net membrane current
            #with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "i_membrane_report.h5"), 'w') as file:
            #    file.create_dataset("report/biophysical/data", data = ecp.im_rec.as_numpy())

            # Save time
            directory_to_save = os.path.join(save_folder, f"saved_at_step_{time_step}")
            #os.makedirs(directory_to_save, exist_ok=True)
            
            # Then create the file
            with h5py.File(os.path.join(directory_to_save, "t.h5"), 'w') as file:
                file.create_dataset("report/biophysical/data", data = t_vec.as_numpy())

            logger.log(f"Saved at time step {time_step}")

            time_steps_saved_at.append(time_step)

            # Reinitialize vectors: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=2579
            t_vec.resize(0)
            for vec in V_rec.vectors: vec.resize(0)
            for vec in cell.Vm.vectors: vec.resize(0)
            for recorder in cell.recorders.items():
                for vec in recorder[1].vectors: vec.resize(0)
            cell.spikes.resize(0)

            for inj in cell.injection: inj.rec_vec.resize(0)

            #for syn in all_syns:
            #    for vec in syn.rec_vec: vec.resize(0)
            
            #for vec in ecp.im_rec.vectors: vec.resize(0)

        h.fadvance()
        time_step += 1

    sim_end_time = time.time()

    logger.log_section_end("Running simulation")

    elapsedtime = sim_end_time - sim_start_time
    total_runtime = sim_end_time - runtime_start_time
    logger.log(f'Simulation time: {round(elapsedtime)} sec.')
    logger.log(f'Total runtime: {round(total_runtime)} sec.')
    print(os.path.join(save_folder, logger.log_file_name))
    print(logger.log_file_name)
    os.system(f"mv {logger.log_file_name} {os.path.join(save_folder, logger.log_file_name)}")

if __name__ == "__main__":
    #constants.CI_on=True
    # make unique output folder for PSCs
    if constants.CI_on: # F/I curve simulation
      constants.save_dir = os.path.join(constants.save_dir, 'FI_in_vitro'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(constants.save_dir):
        try:
            os.mkdir(constants.save_dir)
        except Exception as e:
            print(f"Failed to create directory {constants.save_dir}: {e}")
    logger = Logger(output_dir = constants.save_dir, active = True)
    
    # Compile and load modfiles
    #os.system(f"nrnivmodl {constants.modfiles_folder}")
    ret_code = os.system(f"nrnivmodl {constants.modfiles_folder}")
    if ret_code != 0:
        print(f"Failed to execute nrnivmodl. Return code: {ret_code}")

    h.load_file('stdrun.hoc')
    h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')

    pool = []
    for np_state in constants.numpy_random_states:
        for neuron_state in constants.neuron_random_states:
            if constants.CI_on:
                for i_amplitude in constants.h_i_amplitudes:
                    if constants.parallelize:
                        pool.append(Process(target = main, args=[np_state, neuron_state, logger, i_amplitude]))
                    else:
                        p = Process(target = main, args=[np_state, neuron_state, logger, i_amplitude])
                        try:
                            p.start()
                            p.join()
                        except Exception as e:
                            print(f"Failed to run process: {e}")
                        p.terminate()
            else:
                if constants.parallelize:
                    pool.append(Process(target = main, args=[np_state, neuron_state, logger]))
                else:
                    p = Process(target = main, args=[np_state, neuron_state, logger])
                    try:
                        p.start()
                        p.join()
                    except Exception as e:
                        print(f"Failed to run process: {e}")
                    p.terminate()
                    
                    
    
    if constants.parallelize:
        for p in pool:
            p.start()
            # Start the next process with delay to prevent name conflicts
            time.sleep(1)
        for p in pool: p.join()
        for p in pool: p.terminate()

    os.system("rm -r x86_64")