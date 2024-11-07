import sys
sys.path.append("../")

from Modules.complex_cell import build_L5_cell, build_L5_cell_ziao, build_cell_reports_cell, unpickle_params, inspect_pickle, set_hoc_params, adjust_soma_and_axon_geometry, is_indexable, set_pickled_parameters_to_sections
from Modules.complex_cell import build_cell_reports_cell, assign_parameters_to_section, create_cell_from_template_and_pickle, assign_parameters_from_csv

from Modules.synapse_generator import SynapseGenerator
from Modules.spike_generator import SpikeGenerator
from Modules.complex_cell import build_L5_cell

from Modules.logger import Logger
from Modules.recorder import Recorder
from Modules.reduction import Reductor
from Modules.clustering import create_functional_groups_of_presynaptic_cells
from Modules.cell_model import CellModel

from cell_inference.config import params
from cell_inference.utils.currents.ecp import EcpMod

import numpy as np

import time, datetime
import os, h5py, pickle, shutil
from multiprocessing import Process, cpu_count
import pandas as pd

from neuron import h

import constants

def main(numpy_random_state, neuron_random_state, logger, i_amplitude=None):

    print(f"Running for seeds ({np_state}, {neuron_state}); CI = {i_amplitude}...")

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

    # Unaccounted for: 
    # - build_m1_cell()
    # - set_hoc_params()

    logger.log_section_end("Building complex cell")
    soma = complex_cell.soma[0] if is_indexable(complex_cell.soma) else complex_cell.soma
    print(f"dir(soma) : {dir(soma)} | dir(soma(0.5)) : {dir(soma(0.5))} | dir(soma(0.5).na_ion) : {dir(soma(0.5).na_ion)} | dir(soma(0.5).nax) : {dir(soma(0.5).nax)}")

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

    # Get segments and lengths
    logger.log_section_start("Getting segments and lengths")
    
    

    logger.log_section_end("Getting segments and lengths")

    # ---- Excitatory

    logger.log_section_start("Generating Excitatory func groups")

    

    logger.log_memory()
    
    logger.log_memory()
    logger.log_section_end("Generating Excitatory func groups")

    # ---- Inhibitory

    logger.log_section_start("Generating inhibitory func groups for dendrites")

    
    logger.log_memory()
    logger.log_section_end("Generating inhibitory func groups for dendrites")

    # ---- Soma

    logger.log_section_start("Generating inhibitory func groups for soma")
    logger.log_memory()
    
    
    logger.log_memory()
    logger.log_section_end("Generating inhibitory func groups for soma")

    # ---- Set up a cell model

    logger.log_section_start("Adding all synapses")

    # Get all synapses
    all_syns = []
    for synapse_list in synapse_generator.synapses: # synapse_generator.synapses is a list of synapse lists
        for synapse in synapse_list:
            all_syns.append(synapse)
    
    logger.log_section_end("Adding all synapses")

    logger.log_section_start("Initializing detailed cell model for kmeans clustering")
    logger.log_memory()
    
    # get segment coordinates # can also increase the number of segments here for better clustering resolution.
    cell = CellModel(hoc_model = complex_cell, random_state = random_state)

    logger.log_section_end("Initializing detailed cell model for kmeans clustering")

    logger.log_section_start("Reading detailed seg coordinates nseg="+str(len(cell.segments)))
    logger.log_memory()

    # get segment coordinates, and cluster cell segments into functional groups
    exc_segment_coordinates = []
    segment_coordinates = np.zeros((len(cell.seg_info), 3))
    detailed_seg_info = cell.seg_info.copy()
    soma_coordinates = np.zeros(3)
    
    exc_segment_indices=[]
    
    for ind, seg in enumerate(cell.seg_info): # can probably split perisomatic and distal here.
        segment_coordinates[ind, 0] = seg['p0.5_x3d']
        segment_coordinates[ind, 1] = seg['p0.5_y3d']
        segment_coordinates[ind, 2] = seg['p0.5_z3d']
    
        # segments with exc synapses: trunk segments that not distal oblique sections
        if (not ((seg['sec'] in cell.apic) and (seg['sec'] not in cell.obliques) and (seg['p0.5_y3d'] < 600))) and (h.distance(seg['seg'], cell.soma[0](0.5)) > 100):
            # Store the coordinates of the segment that meets the conditions
            exc_segment_coordinates.append([seg['p0.5_x3d'], seg['p0.5_y3d'], seg['p0.5_z3d']])
            exc_segment_indices.append(ind)
            
        if seg['seg'] == cell.soma[0](0.5):
            soma_coordinates[0] = seg['p0.5_x3d']
            soma_coordinates[1] = seg['p0.5_y3d']
            soma_coordinates[2] = seg['p0.5_z3d']
    
    # Convert the list to a numpy array for easier manipulation later on
    exc_segment_coordinates = np.array(exc_segment_coordinates)

    
    logger.log_section_end("Reading detailed seg coordinates nseg="+str(len(cell.segments)))   
    
    logger.log_section_start("Creating Excitatory Functional groups")
    logger.log_memory()
    # create excitatory functional groups
    exc_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,
                                                                          n_functional_groups=constants.exc_n_FuncGroups,
                                                                          n_presynaptic_cells_per_functional_group=constants.exc_n_PreCells_per_FuncGroup,
                                                                          name_prefix='exc',
                                                                          synapses = exc_synapses, 
                                                                          cell=cell, mean_firing_rate = mean_fr_dist, 
                                                                          spike_generator=spike_generator, 
                                                                          t = t, 
                                                                          random_state=random_state, 
                                                                          method = '1f_noise')
    
    logger.log_section_end("Creating Excitatory Functional groups")
    
    # get exc spikes for inh delay modulation # further implementation could potentially separate delay modulation by functional group.
    exc_spikes=spike_generator.spike_trains.copy()
    #print("exc_spikes:",exc_spikes)
    
    logger.log_section_start("Creating Inhibitory Distributed Functional groups")
    logger.log_memory()
    # generate inh functional groups
    #dendritic
    if not constants.CI_on:
      inh_distributed_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,
                                                                        n_functional_groups=constants.inh_distributed_n_FuncGroups,
                                                                        n_presynaptic_cells_per_functional_group=constants.inh_distributed_n_PreCells_per_FuncGroup,
                                                                        name_prefix='inh',cell=cell, 
                                                                        synapses = inh_synapses, 
                                                                        proximal_fr_dist = proximal_inh_dist, 
                                                                        distal_fr_dist=distal_inh_dist, 
                                                                        spike_generator=spike_generator, 
                                                                        t = t, 
                                                                        random_state=random_state, 
                                                                        spike_trains_to_delay = exc_spikes, 
                                                                        fr_time_shift = constants.inh_firing_rate_time_shift, 
                                                                        soma_coordinates=soma_coordinates, 
                                                                        method = 'delay')
    else:
      inh_distributed_functional_groups = []
    logger.log_section_end("Creating Inhibitory Distributed Functional groups")
    
    logger.log_section_start("Creating Inhibitory SOMA Functional groups")
    logger.log_memory()
    #somatic
    if not constants.CI_on:
      if constants.add_soma_inh_synapses:
        inh_soma_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,
                                                                                    n_functional_groups=1,
                                                                                    n_presynaptic_cells_per_functional_group=1,
                                                                                    name_prefix='soma_inh',
                                                                                    cell=cell, 
                                                                                    synapses = soma_inh_synapses, 
                                                                                    proximal_fr_dist = proximal_inh_dist, 
                                                                                    distal_fr_dist=distal_inh_dist, 
                                                                                    spike_generator=spike_generator, 
                                                                                    t = t, random_state=random_state, 
                                                                                    spike_trains_to_delay = exc_spikes, 
                                                                                    fr_time_shift = constants.inh_firing_rate_time_shift, 
                                                                                    soma_coordinates=soma_coordinates, 
                                                                                    method = 'delay')
    else:
      inh_soma_functionl_groups = []
    logger.log_section_end("Creating Inhibitory SOMA Functional groups")
    
    logger.log_section_start("Storing FuncGroup and PreCell data")
    logger.log_memory()
    # save fg and pc to csv
    
    def functional_group_to_dict(functional_group, functional_group_index): # REPLACED BY PICKLING
        """Converts a FunctionalGroup object to a dictionary with an index."""
        data = {attr: value for attr, value in functional_group.__dict__.items() if attr != "presynaptic_cells"}
        data['functional_group_index'] = functional_group_index
        return data
    
    def presynaptic_cell_to_dict(presynaptic_cell, presynaptic_cell_index):
        """Converts a PresynapticCell object to a dictionary with an index."""
        data = {attr: value for attr, value in presynaptic_cell.__dict__.items()}
        data['presynaptic_cell_index'] = presynaptic_cell_index
        return data
    
    def functional_groups_to_dataframe_with_index(functional_groups):
        """Converts a list of FunctionalGroup objects to a DataFrame with indices."""
        functional_group_data = []
        presynaptic_cell_data = []
        functional_group_index = 0
        presynaptic_cell_index = 0
    
        for fg in functional_groups:
            fg_data=functional_group_to_dict(fg, functional_group_index)
            fg_synapses=0
            for pc in fg.presynaptic_cells:
                pc_data = presynaptic_cell_to_dict(pc, presynaptic_cell_index)
                pc_data['functional_group_index'] = functional_group_index  # Reference to the FunctionalGroup it belongs to
                pc_data['num_synapses'] = len(pc.synapses)
                fg_synapses += len(pc.synapses)
                presynaptic_cell_data.append(pc_data)
                presynaptic_cell_index += 1
            fg_data['num_synapses'] = fg_synapses
            functional_group_data.append(fg_data)
            functional_group_index += 1
    
        functional_group_df = pd.DataFrame(functional_group_data)
        presynaptic_cell_df = pd.DataFrame(presynaptic_cell_data)
        return functional_group_df, presynaptic_cell_df

    # Convert dictionary to dataframe ######### replaced with pickling
    #exc_functional_groups_df, exc_presynaptic_cells_df = functional_groups_to_dataframe_with_index(exc_functional_groups)
    #inh_distributed_functional_groups_df, inh_distributed_presynaptic_cells_df = functional_groups_to_dataframe_with_index(inh_distributed_functional_groups)
    #inh_soma_functional_groups_df, inh_soma_presynaptic_cells_df = functional_groups_to_dataframe_with_index(inh_soma_functional_groups)
    
    logger.log_section_end("Storing FuncGroup and PreCell data") 
    
    logger.log_section_start("Initializing Reductor and cell model for simulation |NR:"+str(constants.reduce_cell)+"|optimize nseg:"+str(constants.optimize_nseg_by_lambda)+"|Expand Cable:"+str(constants.expand_cable))
    logger.log_memory()
    
    reductor = Reductor()
#    print('netcons list:',len(spike_generator.netcons))
#    print('unique netcons in list:',len(np.unique(spike_generator.netcons)))
    cell = reductor.reduce_cell(complex_cell = complex_cell, reduce_cell = constants.reduce_cell, 
                                optimize_nseg = constants.optimize_nseg_by_lambda, synapses_list = all_syns,
                                netcons_list = spike_generator.netcons, spike_trains = spike_generator.spike_trains,
                                spike_threshold = constants.spike_threshold, random_state = random_state,
                                var_names = constants.channel_names, reduction_frequency = constants.reduction_frequency, 
                                expand_cable = constants.expand_cable, choose_branches = constants.choose_branches)
                                
    logger.log_section_end("Initializing Reductor and cell model for simulation |NR:"+str(constants.reduce_cell)+"|optimize nseg:"+str(constants.optimize_nseg_by_lambda)+"|Expand Cable:"+str(constants.expand_cable))
    if not constants.CI_on:
      # Turn off certain presynaptic neurons to simulate in vivo
      if not constants.trunk_exc_synapses: # turn off trunk exc synapses.
        for synapse in cell.synapses:
          if (synapse.get_segment().sec in cell.apic) & (synapse.syn_type in constants.exc_syn_mod) & (synapse.get_segment().sec not in cell.obliques) & (synapse.get_segment().sec.y3d(0)<600):
              for netcon in synapse.ncs:
                netcon.active(False)
      
      # Turn off perisomatic exc neurons
      if not constants.perisomatic_exc_synapses:
        perisomatic_inputs_disabled=0
        for synapse in cell.synapses:
          if (h.distance(synapse.get_segment(), cell.soma[0](0.5)) < 75) & (synapse.syn_type in constants.exc_syn_mod):
              for netcon in synapse.ncs:
                perisomatic_inputs_disabled+=1
                netcon.active(False)
        print("perisomatic inputs disabled:",perisomatic_inputs_disabled)
      
      if constants.merge_synapses: # may already be merged if reductor reduced the cell.
          logger.log_section_start("Merging Synapses")
          logger.log_memory()
          reductor.merge_synapses(cell)
          logger.log_section_end("Merging Synapses")
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
    else: # Complex cell
        tuft_seg_index=cell.segments.index(cell.tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        if constants.build_cell_reports_cell:
          nexus_seg_index = cell.segments.index(cell.apic[24](0.99)) # may need to adjust
        else:
          nexus_seg_index=cell.segments.index(cell.apic[36](0.961538))
    seg_indexes = {
        "soma": soma_seg_index,
        "axon": axon_seg_index,
        "basal": basal_seg_index,
        "trunk": trunk_seg_index,
        "tuft": tuft_seg_index,
        "nexus": nexus_seg_index
    }
    logger.log_section_end("Finding segments of interest")
    
    # Compute electrotonic distances from nexus
    logger.log_section_start("Recomputing elec distance")

    cell.recompute_segment_elec_distance(segment = cell.segments[nexus_seg_index], seg_name = "nexus")

    logger.log_section_end("Recomputing elec distance")

    logger.log_section_start("Initializing t_vec and V_rec recorder")

    # Record time points
    t_vec = h.Vector(1000 / h.dt).record(h._ref_t)

    # Record membrane voltage of all segments
    V_rec = Recorder(cell.segments, vector_length = constants.save_every_ms)

    logger.log_section_end("Initializing t_vec and V_rec recorder")

    logger.log_section_start("Creating ecp object")

    elec_pos = params.ELECTRODE_POSITION
    ecp = EcpMod(cell, elec_pos, min_distance = params.MIN_DISTANCE)  # create an ECP object for extracellular potential

    logger.log_section_end("Creating ecp object")

    # ---- Run simulation
    sim_duration = h.tstop / 1000 # Convert from ms to s

    logger.log_section_start(f"Running sim for {sim_duration} sec")
    logger.log_memory()

    sim_start_time = time.time()

    time_step = 0 # In time stamps, i.e., ms / dt
    time_steps_saved_at = [0]

    # Create a folder to save to
    if constants.CI_on:
        random_seed_name = "_seeds_" +\
                           str(numpy_random_state) + "_" + str(neuron_random_state) + cell.get_output_folder_name()
    else:
        random_seed_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_seeds_" +\
                           str(numpy_random_state) + "_" + str(neuron_random_state) + cell.get_output_folder_name()
    if i_amplitude is not None:
        random_seed_name += f"_{int(i_amplitude * 1000)}"
    save_folder = os.path.join(constants.save_dir, random_seed_name)
    os.mkdir(save_folder)

    # Save indexes for plotting
    with open(os.path.join(save_folder, "seg_indexes.pickle"), "wb") as file:
        pickle.dump(seg_indexes, file)
        
    # examine spikes # needs to be moved to analysis script
    # logger.log_section_start('plotting spike rasters')
    # plot_spike_rasters(functional_groups=exc_functional_groups, title_prefix="exc", save_to=save_folder)
    # plot_spike_rasters(functional_groups=inh_distributed_functional_groups, title_prefix="inh_distal", save_to=save_folder)
    # plot_spike_rasters(functional_groups=inh_soma_functional_groups, title_prefix="inh_perisomatic", save_to=save_folder)
    # #calculate firing rates
    # exam_inc_spikes(functional_groups=exc_functional_groups, tstop=constants.h_tstop, title_prefix="exc", save_to=save_folder)
    # exam_inc_spikes(functional_groups=inh_distributed_functional_groups, tstop=constants.h_tstop, title_prefix="inh_distal", save_to=save_folder)
    # exam_inc_spikes(functional_groups=inh_soma_functional_groups, tstop=constants.h_tstop, title_prefix="inh_perisomatic", save_to=save_folder)
    # logger.log_section_end('plotting spike rasters')
        
    # Save fg and pc to CSV within the save_folder for plotting # has been replaced by pickling
    # logger.log_section_start('saving seg_info, functional groups, and presynaptic cells to csv') # '\t'
    # exc_functional_groups_df.to_csv(os.path.join(save_folder, "exc_functional_groups.csv"), index=False, sep = ';')
    # exc_presynaptic_cells_df.to_csv(os.path.join(save_folder, "exc_presynaptic_cells.csv"), index=False, sep = ';')
    # inh_distributed_functional_groups_df.to_csv(os.path.join(save_folder, "inh_distributed_functional_groups.csv"), index=False, sep = ';')
    # inh_distributed_presynaptic_cells_df.to_csv(os.path.join(save_folder, "inh_distributed_presynaptic_cells.csv"), index=False, sep = ';')
    # inh_soma_functional_groups_df.to_csv(os.path.join(save_folder, "inh_soma_functional_groups.csv"), index=False, sep = ';')
    # inh_soma_presynaptic_cells_df.to_csv(os.path.join(save_folder, "inh_soma_presynaptic_cells.csv"), index=False, sep = ';')
    # logger.log_section_end('saving seg_info, functional groups, and presynaptic cells to csv')

    logger.log_section_start('saving seg_info to csv')
    cell.write_seg_info_to_csv(path=save_folder, seg_info=detailed_seg_info, title_prefix='detailed_')
    logger.log_section_end('saving seg_info to csv')
    
    # temporarily removed
    # logger.log_section_start('pickleing functional groups and presynaptic cells')
    # # Pickle FunctionalGroup objects
    # with open(os.path.join(save_folder, "exc_functional_groups.pkl"), "wb") as f:
    #     pickle.dump(exc_functional_groups, f)
    # with open(os.path.join(save_folder, "inh_distal_functional_groups.pkl"), "wb") as f:
    #     pickle.dump(inh_distributed_functional_groups, f)
    # with open(os.path.join(save_folder, "inh_perisomatic_functional_groups.pkl"), "wb") as f:
    #     pickle.dump(inh_soma_functional_groups, f)
    # logger.log_section_end('pickleing functional groups and presynaptic cells')

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
            lfp = ecp.calc_ecp(move_cell = loc_param).T  # Unit: mV

            with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "lfp.h5"), 'w') as file:
                file.create_dataset("report/biophysical/data", data = lfp)
            # save net membrane current
            with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "i_membrane_report.h5"), 'w') as file:
                file.create_dataset("report/biophysical/data", data = ecp.im_rec.as_numpy())

            # Save time
            with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "t.h5"), 'w') as file:
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

            for syn in all_syns:
                for vec in syn.rec_vec: vec.resize(0)
            
            for vec in ecp.im_rec.vectors: vec.resize(0)

        h.fadvance()
        time_step += 1

    sim_end_time = time.time()

    logger.log_section_end("Running simulation")

    elapsedtime = sim_end_time - sim_start_time
    total_runtime = sim_end_time - runtime_start_time
    logger.log(f'Simulation time: {round(elapsedtime)} sec.')
    logger.log(f'Total runtime: {round(total_runtime)} sec.')

    os.system(f"mv {logger.log_file_name} {os.path.join(save_folder, logger.log_file_name)}")

if __name__ == "__main__":

    # make unique output folder for PSCs
    if constants.CI_on: # F/I curve simulation
      constants.save_dir = os.path.join(constants.save_dir, 'FI_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(constants.save_dir):
        os.mkdir(constants.save_dir)
        #raise FileNotFoundError("No save folder with the given name.")
    logger = Logger(output_dir = constants.save_dir, active = True)
    
    # Compile and load modfiles
    os.system(f"nrnivmodl {constants.modfiles_folder}")
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
                        p.start()
                        p.join()
                        p.terminate()
            else:
                if constants.parallelize:
                    pool.append(Process(target = main, args=[np_state, neuron_state, logger]))
                else:
                    p = Process(target = main, args=[np_state, neuron_state, logger])
                    p.start()
                    p.join()
                    p.terminate()
    
    if constants.parallelize:
        for p in pool:
            p.start()
            # Start the next process with delay to prevent name conflicts
            time.sleep(1)
        for p in pool: p.join()
        for p in pool: p.terminate()

    os.system("rm -r x86_64")