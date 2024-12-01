import sys
sys.path.append("../")

from Modules.synapse_generator import SynapseGenerator
from Modules.spike_generator import SpikeGenerator
from Modules.complex_cell import build_L5_cell
from Modules.cell_utils import get_segments_and_len_per_segment
from Modules.logger import Logger
from Modules.recorder import Recorder
from Modules.reduction import Reductor
from Modules.clustering import create_functional_groups_of_presynaptic_cells
from Modules.cell_model import CellModel

from cell_inference.config import params
from cell_inference.utils.currents.ecp import EcpMod

import numpy as np
from functools import partial
import scipy.stats as st
import time, datetime
import os, h5py, pickle, shutil
from multiprocessing import Process
import pandas as pd

from neuron import h

import constants

def main(numpy_random_state, neuron_random_state, cluster_index_to_stim, logger):

    print(f"Running for seeds ({np_state}, {neuron_state}); CI = {cluster_index_to_stim}...")

    # Random seed
    logger.log_section_start(f"Setting random states ({numpy_random_state}, {neuron_random_state})")

    random_state = np.random.RandomState(numpy_random_state)
    np.random.seed(numpy_random_state)

    neuron_r = h.Random()
    neuron_r.MCellRan4(neuron_random_state)

    logger.log_section_end("Setting random states")

    # Time vector for generating inputs
    t = np.arange(0, constants.h_tstop, 1)

    # Build cell
    logger.log_section_start("Building complex cell")

    complex_cell = build_L5_cell(constants.complex_cell_folder, constants.complex_cell_biophys_hoc_name)

    logger.log_section_end("Building complex cell")

    h.celsius = constants.h_celcius
    h.v_init = complex_cell.soma[0].e_pas

    # Sim runtime
    h.tstop = constants.h_tstop

    # Timestep (ms)
    h.dt = constants.h_dt
    h.steps_per_ms = 1 / h.dt

    # Measure time
    runtime_start_time = time.time()

    # Get segments and lengths
    logger.log_section_start("Getting segments and lengths")
    
    # increase nseg for complex cell # for clustering of synapses by kmeans on segments
    for sec in complex_cell.all:
      sec.nseg=int(sec.L)+1#sec.nseg=int(sec.L*2)+1

    all_segments, all_len_per_segment, all_SA_per_segment,\
    all_segments_center, soma_segments, soma_len_per_segment,\
    soma_SA_per_segment, soma_segments_center, no_soma_segments,\
    no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center =\
    get_segments_and_len_per_segment(complex_cell)

    logger.log_section_end("Getting segments and lengths")

    # ---- Excitatory

    logger.log_section_start("Generating Excitatory func groups")

    # Excitatory gmax distribution
    exc_gmax_mean_0 = constants.exc_gmax_mean_0
    exc_gmax_std_0 = constants.exc_gmax_std_0

    gmax_mean = np.log(exc_gmax_mean_0) - 0.5 * np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1)
    gmax_std = np.sqrt(np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1))

    # gmax distribution
    def log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size):
        val = np.random.lognormal(gmax_mean, gmax_std, size)
        s = gmax_scalar * float(np.clip(val, constants.exc_gmax_clip[0], constants.exc_gmax_clip[1]))
        return s

    gmax_exc_dist = partial(log_norm_dist, gmax_mean, gmax_std, constants.exc_scalar, size = 1)

    # Excitatory firing rate distribution
    def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
        return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, 
                                         loc = loc, scale = scale, size = size)) + 1e-15
    
    spike_generator = SpikeGenerator()
    synapse_generator = SynapseGenerator()

    #exc_number_of_groups = int(sum(all_len_per_segment) / constants.exc_functional_group_span)

    # Number of presynaptic cells
    #cells_per_group = int(constants.exc_functional_group_span * constants.exc_synaptic_density / constants.exc_synapses_per_cluster)

    # Distribution of mean firing rates
    mean_fr_dist = partial(exp_levy_dist, alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1)
    
    # release probability distribution
    def P_release_dist(P_mean, P_std, size):
        val = np.random.normal(P_mean, P_std, size)
        s = float(np.clip(val, 0, 1))
        return s
    
    # exc release probability distribution everywhere
    exc_P_dist = partial(P_release_dist, P_mean=constants.exc_P_release_mean, P_std=constants.exc_P_release_std, size=1)
    
    # New list to change probabilty of exc functional group nearing soma
    adjusted_no_soma_len_per_segment = []
    for i, seg in enumerate(no_soma_segments):
        if h.distance(seg, complex_cell.soma[0](0.5)) < 75:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
        elif seg in complex_cell.apic[0]: # trunk
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 5)
        else:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])

    logger.log_memory()

    exc_synapses = synapse_generator.add_synapses(segments = no_soma_segments,
                                              probs = no_soma_len_per_segment,
                                              density=constants.exc_synaptic_density,
                                              record = True,
                                              vector_length = constants.save_every_ms,
                                              gmax = gmax_exc_dist,
                                              random_state=random_state,
                                              neuron_r = neuron_r,
                                              syn_mod = constants.exc_syn_mod,
                                              P_dist=exc_P_dist
                                              )

    logger.log_memory()
    logger.log_section_end("Generating Excitatory func groups")

    # ---- Inhibitory

    logger.log_section_start("Generating inhibitory func groups for dendrites")

    # Proximal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_prox_mean_fr, constants.inh_prox_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    # Distal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_distal_mean_fr, constants.inh_distal_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)
    
    # inh release probability distributions
    inh_soma_P_dist = partial(P_release_dist, P_mean=constants.inh_soma_P_release_mean, P_std=constants.inh_soma_P_release_std, size=1)
    inh_apic_P_dist = partial(P_release_dist, P_mean=constants.inh_apic_P_release_mean, P_std=constants.inh_apic_P_release_std, size=1)
    inh_basal_P_dist = partial(P_release_dist, P_mean=constants.inh_basal_P_release_mean, P_std=constants.inh_basal_P_release_std, size=1)
    
    inh_P_dist ={}
    inh_P_dist["soma"] = inh_soma_P_dist
    inh_P_dist["apic"] = inh_apic_P_dist
    inh_P_dist["dend"] = inh_basal_P_dist
    
    inh_gmax = constants.inh_gmax_dist * constants.inh_scalar

    logger.log_memory()
    inh_synapses = synapse_generator.add_synapses(segments = all_segments,
                                              probs = all_len_per_segment,
                                              density=constants.inh_synaptic_density,
                                              record = True,
                                              vector_length = constants.save_every_ms,
                                              gmax = inh_gmax,
                                              random_state=random_state,
                                              neuron_r = neuron_r,
                                              syn_mod = constants.inh_syn_mod,
                                              P_dist = inh_P_dist,
                                              cell=complex_cell
                                              )
    
    logger.log_memory()
    logger.log_section_end("Generating inhibitory func groups for dendrites")

    # ---- Soma

    logger.log_section_start("Generating inhibitory func groups for soma")
    logger.log_memory()
    soma_inh_synapses = synapse_generator.add_synapses(segments = soma_segments,
                                              probs = soma_SA_per_segment,
                                              number_of_synapses=150,
                                              record = True,
                                              vector_length = constants.save_every_ms,
                                              gmax = inh_gmax,
                                              random_state=random_state,
                                              neuron_r = neuron_r,
                                              syn_mod = constants.inh_syn_mod,
                                              P_dist=inh_soma_P_dist
                                              )
    
    logger.log_memory()
    logger.log_section_end("Generating inhibitory func groups for soma")

    # ---- Set up a cell model

    logger.log_section_start("Adding all synapses")

    # Get all synapses
    all_syns = []
    for synapse_list in synapse_generator.synapses: # synapse_generator.synapses is a list of synapse lists
        for synapse in synapse_list:
            all_syns.append(synapse)
    
    excit_synapses=[]
    inhib_synapses=[]
    soma_inhib_synapses=[]
    for synapse in exc_synapses:
      excit_synapses.append(synapse.synapse_neuron_obj)
    for synapse in inh_synapses:
      inhib_synapses.append(synapse.synapse_neuron_obj)
    for synapse in soma_inh_synapses:
      soma_inhib_synapses.append(synapse.synapse_neuron_obj)
            
    #print("exc_synapses:", excit_synapses)
    #print("inh_synapses:", inhib_synapses)
    #print("soma_inh_synapses:", soma_inhib_synapses)
    logger.log_section_end("Adding all synapses")

    logger.log_section_start("Initializing detailed cell model for kmeans clustering")
    logger.log_memory()
    
    # get segment coordinates # can also increase the number of segments here for better clustering resolution.
    cell = CellModel(hoc_model = complex_cell, random_state = random_state)

    logger.log_section_end("Initializing detailed cell model for kmeans clustering")

    logger.log_section_start("Reading detailed seg coordinates nseg="+str(len(cell.segments)))
    logger.log_memory()

    # get segmetsncoordinatesCluster cell segments into functional groups
    segment_coordinates = np.zeros((len(cell.seg_info), 3))
    detailed_seg_info = cell.seg_info.copy()
    soma_coordinates = np.zeros(3)
    for ind, seg in enumerate(cell.seg_info):
        segment_coordinates[ind, 0] = seg['p0.5_x3d']
        segment_coordinates[ind, 1] = seg['p0.5_y3d']
        segment_coordinates[ind, 2] = seg['p0.5_z3d']
        if seg['seg'] == cell.soma[0](0.5):
          soma_coordinates[0] = seg['p0.5_x3d']
          soma_coordinates[1] = seg['p0.5_y3d']
          soma_coordinates[2] = seg['p0.5_z3d']
    
    logger.log_section_end("Reading detailed seg coordinates nseg="+str(len(cell.segments)))   
    
    logger.log_section_start("Creating Excitatory Functional groups")
    logger.log_memory()
    # create excitatory functional groups
    exc_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,n_functional_groups=constants.exc_n_FuncGroups,n_presynaptic_cells_per_functional_group=constants.exc_n_PreCells_per_FuncGroup,name_prefix='exc',synapses = exc_synapses, cell=cell, mean_firing_rate = mean_fr_dist, spike_generator=spike_generator, t = t, random_state=random_state, method = '1f_noise')
    
    logger.log_section_end("Creating Excitatory Functional groups")
    
    # get exc spikes for inh delay modulation # further implementation could potentially separate delay modulation by functional group.
    exc_spikes=spike_generator.spike_trains.copy()
    #print("exc_spikes:",exc_spikes)
    #print("exc_spikes if there was no copy:",spike_generator.spike_trains)
    
    logger.log_section_start("Creating Inhibitory Distributed Functional groups")
    logger.log_memory()
    # generate inh functional groups
    #dendritic
    inh_distributed_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,n_functional_groups=constants.inh_distributed_n_FuncGroups,n_presynaptic_cells_per_functional_group=constants.inh_distributed_n_PreCells_per_FuncGroup,name_prefix='inh',cell=cell, synapses = inh_synapses, proximal_fr_dist = proximal_inh_dist, distal_fr_dist=distal_inh_dist, spike_generator=spike_generator, t = t, random_state=random_state, spike_trains_to_delay = exc_spikes, fr_time_shift = constants.inh_firing_rate_time_shift, soma_coordinates=soma_coordinates, method = 'delay')
    logger.log_section_end("Creating Inhibitory Distributed Functional groups")
    
    logger.log_section_start("Creating Inhibitory SOMA Functional groups")
    logger.log_memory()
    #somatic
    inh_soma_functional_groups = create_functional_groups_of_presynaptic_cells(segments_coordinates=segment_coordinates,n_functional_groups=1,n_presynaptic_cells_per_functional_group=1,name_prefix='soma_inh',cell=cell, synapses = soma_inh_synapses, proximal_fr_dist = proximal_inh_dist, distal_fr_dist=distal_inh_dist, spike_generator=spike_generator, t = t, random_state=random_state, spike_trains_to_delay = exc_spikes, fr_time_shift = constants.inh_firing_rate_time_shift, soma_coordinates=soma_coordinates, method = 'delay')
    logger.log_section_end("Creating Inhibitory SOMA Functional groups")
    
    logger.log_section_start("Storing FuncGroup and PreCell data")
    logger.log_memory()
    # save fg and pc to csv
    
    def functional_group_to_dict(functional_group, functional_group_index):
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

    # Convert dictionary to dataframe
    exc_functional_groups_df, exc_presynaptic_cells_df = functional_groups_to_dataframe_with_index(exc_functional_groups)
    inh_distributed_functional_groups_df, inh_distributed_presynaptic_cells_df = functional_groups_to_dataframe_with_index(inh_distributed_functional_groups)
    inh_soma_functional_groups_df, inh_soma_presynaptic_cells_df = functional_groups_to_dataframe_with_index(inh_soma_functional_groups)
    
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
    
    #####################################################
    logger.log_section_start("Setting simulation for PSC")
    from Modules.injection import SEClamp
        # Turn off all presynaptic neurons to simulate in vivo                            
    for synapse in cell.synapses:
      for netcon in synapse.ncs:
        netcon.active(False)
    # Turn on single presynaptic neuron to simulate current injection to presynaptic neuron
    total_num_clusters = {} # count clusters to get numbers for repeated simulations
    total_num_clusters["all"] = 0
    total_num_clusters["dend"] = 0
    total_num_clusters["apic"] = 0
    total_num_clusters["soma"] = 0
    cluster_distances_from_soma = []
    func_grps = exc_functional_groups + inh_distributed_functional_groups + inh_distributed_functional_groups
    for func_grp in func_grps:
      for pc in func_grp.presynaptic_cells:
        a_syn = pc.synapses[0]
        a_seg = a_syn.get_segment()
        pc_type = a_seg.sec.name().split('.')[1][:4]
        total_num_clusters[pc_type] += 1 # sort number by type
        total_num_clusters['all'] += 1 # total number
        if total_num_clusters['all'] == cluster_index_to_stim: # this presynaptic cell's turn
          PSC_sec_type = cluster_type
          PSC_syn_type = cluster.synapses[0].get_exc_or_inh_from_syn_type()
          cluster_distance_from_soma = h.distance(cell.soma[0](0.5),a_seg)
          if cluster_distance_from_soma < 100:
            perisomatic=True
          else:
            perisomatic=False
          stim = spike_generator.create_netstim(start=constants.PSC_start,number=1,noise=0,interval=100)# create single presynaptic spike input
          for synapse in cluster.synapses:
            spike_generator.set_netcon(synapse=synapse,stim=stim) # deliver spike to this synapse
    print("Total number of clusters:", total_num_clusters['all'])
    # Voltage Clamp Soma.
    if PSC_syn_type == 'inh':
      voltage_to_clamp = 0 # (mV) excitatory synapse reversal potential
    elif PSC_syn_type == 'exc':
      voltage_to_clamp = -75 # (mV) inhibitory synapse reversal potential
    else:
      raise(ValueError("PSC_syn_type must be either 'inh' or 'exc'."))
    #Create clamp
    clamp = SEClamp(seg = cell.soma[0](0.5), dur1 = 1000, amp1 = voltage_to_clamp, rs = 0.01, record = True)
    #note: update the simulation reruns to include number_denoting_cluster_ind_to_stimulate
    logger.log_section_end("Setting simulation for PSC")
    ################################################      
    
#    # Turn off certain presynaptic neurons to simulate in vivo
#    if not constants.trunk_exc_synapses: # turn off trunk exc synapses.
#      for synapse in cell.synapses:
#        if (synapse.get_segment().sec in cell.apic) & (synapse.syn_type in constants.exc_syn_mod) & (synapse.get_segment().sec not in cell.obliques) & (synapse.get_segment().sec.y3d(0)<600):
#            for netcon in synapse.ncs:
#              netcon.active(False)
#    
#    # Turn of perisomatic inhibitory neurons
#    if not constants.perisomatic_exc_synapses:
#      perisomatic_inputs_disabled=0
#      for synapse in cell.synapses:
#        if (h.distance(synapse.get_segment(), cell.soma[0](0.5)) < 75) & (synapse.syn_type in constants.inh_syn_mod):
#            for netcon in synapse.ncs:
#              perisomatic_inputs_disabled+=1
#              netcon.active(False)
#      print("perisomatic inputs disabled:",perisomatic_inputs_disabled)
    
    if constants.merge_synapses: # may already be merged if reductor reduced the cell.
        logger.log_section_start("Merging Synapses")
        logger.log_memory()
        reductor.merge_synapses(cell)
        logger.log_section_end("Merging Synapses")
    logger.log_section_start("Setting up cell var recorders")
    logger.log_memory()
    cell.setup_recorders(vector_length = constants.save_every_ms)
    logger.log_section_end("Setting up cell var recorders")

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
    random_seed_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_seeds_" +\
                       str(numpy_random_state) + "_" + str(neuron_random_state) + cell.get_output_folder_name()
    if cluster_index_to_stim is not None:
        random_seed_name += f"_cluster{cluster_index_to_stim}"
    save_folder = os.path.join(constants.save_dir, random_seed_name)
    os.mkdir(save_folder)

    # Save indexes for plotting
    with open(os.path.join(save_folder, "seg_indexes.pickle"), "wb") as file:
        pickle.dump(seg_indexes, file)
        
    # Save fg and pc to CSV within the save_folder for plotting
    exc_functional_groups_df.to_csv(os.path.join(save_folder, "exc_functional_groups.csv"), index=False)
    exc_presynaptic_cells_df.to_csv(os.path.join(save_folder, "exc_presynaptic_cells.csv"), index=False)
    inh_distributed_functional_groups_df.to_csv(os.path.join(save_folder, "inh_distributed_functional_groups.csv"), index=False)
    inh_distributed_presynaptic_cells_df.to_csv(os.path.join(save_folder, "inh_distributed_presynaptic_cells.csv"), index=False)
    inh_soma_functional_groups_df.to_csv(os.path.join(save_folder, "inh_soma_functional_groups.csv"), index=False)
    inh_soma_presynaptic_cells_df.to_csv(os.path.join(save_folder, "inh_soma_presynaptic_cells.csv"), index=False)
    cell.write_seg_info_to_csv(path=save_folder, seg_info=detailed_seg_info, title_prefix='detailed_')
    
    

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
            
            ########################
            # Save clamp current
            with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", str(cluster_index_to_stim)+'_'+str(perisomatic)+'_'+PSC_syn_type+'_'+PSC_sec_type+"_Vclamp_i.h5"), 'w') as file:
                file.create_dataset("report/biophysical/data", data = clamp.rec_vec.as_numpy())
            ########################

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
            ###########################
            clamp.rec_vec.resize(0)
            ###########################

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
    constants.save_dir = os.path.join(constants.save_dir, 'PSC_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
            for cluster_index_to_stim in range(0,1):#constants.number_of_presynaptic_cells+1):
                    if constants.parallelize:
                        pool.append(Process(target = main, args=[np_state, neuron_state, cluster_index_to_stim, logger]))
                    else:
                        p = Process(target = main, args=[np_state, neuron_state, cluster_index_to_stim, logger])
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