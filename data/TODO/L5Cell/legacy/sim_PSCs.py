# Note of changes to record EPSCs:
# Turn off all synapses
# deliver one spike to one presynaptic cell
# Add Voltage clamp to soma
# if active synapse is Inh syn, set voltage clamp at Exc syn reversal potential, and vice versa.

# Recommending updating constants.py as follows:
# constants.h_tstop = 500
# constants.save_every_ms = 500
import sys
sys.path.append("../")

from Modules.synapse_generator import SynapseGenerator
from Modules.spike_generator import SpikeGenerator
from Modules.complex_cell import build_L5_cell
from Modules.functional_group import generate_excitatory_functional_groups, generate_inhibitory_functional_groups
from Modules.cell_utils import get_segments_and_len_per_segment
from Modules.logger import Logger
from Modules.recorder import Recorder
from Modules.reduction import Reductor
from Modules.injection import SEClamp

from cell_inference.config import params
from cell_inference.utils.currents.ecp import EcpMod

import numpy as np
from functools import partial
import scipy.stats as st
import time, datetime
import os, h5py, pickle, shutil
from multiprocessing import Process

from neuron import h

import constants

def main(numpy_random_state, neuron_random_state, i_amplitude, cluster_index_to_stim):

    # Compile and load modfiles
    os.system(f"nrnivmodl {constants.modfiles_folder}")
    h.load_file('stdrun.hoc')
    h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')

    logger = Logger(output_dir = "./", active = True)

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
    def log_norm_dist(gmax_mean, gmax_std, size):
        val = np.random.lognormal(gmax_mean, gmax_std, size)
        s = float(np.clip(val, constants.exc_gmax_clip[0], constants.exc_gmax_clip[1]))
        return s

    gmax_exc_dist = partial(log_norm_dist, gmax_mean, gmax_std, size = 1)

    # Excitatory firing rate distribution
    def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
        return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, 
                                         loc = loc, scale = scale, size = size)) + 1e-15
    
    spike_generator = SpikeGenerator()
    synapse_generator = SynapseGenerator()

    exc_number_of_groups = int(sum(all_len_per_segment) / constants.exc_functional_group_span)

    # Number of presynaptic cells
    cells_per_group = int(constants.exc_functional_group_span * constants.exc_synaptic_density / constants.exc_synapses_per_cluster)

    # Distribution of mean firing rates
    mean_fr_dist = partial(exp_levy_dist, alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1)

    # New list to change probabilty of exc functional group nearing soma
    adjusted_no_soma_len_per_segment = []
    for i, seg in enumerate(no_soma_segments):
        if h.distance(seg, complex_cell.soma[0](0.5)) < 75:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
        else:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])

    logger.log_memory()

    exc_functional_groups = generate_excitatory_functional_groups(all_segments = no_soma_segments,
                                                                  all_len_per_segment = no_soma_len_per_segment,
                                                                  all_segments_centers = no_soma_segments_center,
                                                                  number_of_groups = exc_number_of_groups,
                                                                  cells_per_group = cells_per_group,
                                                                  synapses_per_cluster = constants.exc_synapses_per_cluster,
                                                                  functional_group_span = constants.exc_functional_group_span,
                                                                  cluster_span = constants.exc_cluster_span,
                                                                  gmax_dist = gmax_exc_dist,
                                                                  mean_fr_dist = mean_fr_dist,
                                                                  spike_generator = spike_generator,
                                                                  synapse_generator = synapse_generator,
                                                                  t = t, random_state = random_state,
                                                                  neuron_r = neuron_r,
                                                                  record = True, syn_mod = 'AMPA_NMDA',
                                                                  vector_length = constants.save_every_ms)
    
    exc_spikes = spike_generator.spike_trains

    logger.log_memory()
    logger.log_section_end("Generating Excitatory func groups")

    # ---- Inhibitory

    logger.log_section_start("Generating inhibitory func groups for dendrites")

    inh_number_of_clusters = int(sum(all_len_per_segment) / constants.inh_cluster_span)
    inh_synapses_per_cluster = int(constants.inh_cluster_span * constants.inh_synaptic_density)

    # Proximal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_prox_mean_fr, constants.inh_prox_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    # Distal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_distal_mean_fr, constants.inh_distal_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    logger.log_memory()

    inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                         all_segments = all_segments,
                                                                         all_len_per_segment = all_len_per_segment,
                                                                         all_segments_centers = all_segments_center,
                                                                         number_of_groups = 1,
                                                                         cells_per_group = inh_number_of_clusters,
                                                                         synapses_per_cluster = inh_synapses_per_cluster,
                                                                         functional_group_span = constants.inh_functional_group_span,
                                                                         cluster_span = constants.inh_cluster_span,
                                                                         gmax_dist = constants.inh_gmax_dist,
                                                                         proximal_inh_dist = proximal_inh_dist,
                                                                         distal_inh_dist = distal_inh_dist,
                                                                         spike_generator = spike_generator,
                                                                         synapse_generator = synapse_generator,
                                                                         t = t, f_group_name_prefix = "diffuse_inh_",
                                                                         random_state = random_state, neuron_r = neuron_r,
                                                                         spike_trains_to_delay = exc_spikes, 
                                                                         fr_time_shift = constants.inh_firing_rate_time_shift,
                                                                         record = True, syn_mod = 'GABA_AB',
                                                                         vector_length = constants.save_every_ms)
    
    logger.log_memory()
    logger.log_section_end("Generating inhibitory func groups for dendrites")

    # ---- Soma

    logger.log_section_start("Generating inhibitory func groups for soma")
    logger.log_memory()

    soma_inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                              all_segments = soma_segments,
                                                                              all_len_per_segment = soma_SA_per_segment,
                                                                              all_segments_centers = soma_segments_center,
                                                                              number_of_groups = constants.soma_number_of_groups,
                                                                              cells_per_group = constants.soma_number_of_clusters,
                                                                              synapses_per_cluster = constants.soma_synapses_per_cluster,
                                                                              functional_group_span = constants.soma_functional_group_span,
                                                                              cluster_span = constants.soma_cluster_span,
                                                                              gmax_dist = constants.soma_gmax_dist,
                                                                              proximal_inh_dist = proximal_inh_dist,
                                                                              distal_inh_dist = distal_inh_dist,
                                                                              spike_generator = spike_generator,
                                                                              synapse_generator = synapse_generator,
                                                                              t = t, f_group_name_prefix = "soma_inh_",
                                                                              random_state = random_state, neuron_r = neuron_r,
                                                                              spike_trains_to_delay = exc_spikes, 
                                                                              fr_time_shift = constants.inh_firing_rate_time_shift,
                                                                              record = True, syn_mod = 'GABA_AB',
                                                                              vector_length = constants.save_every_ms)
    
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

    logger.log_section_start("Initializing cell model")
    logger.log_memory()
    
    reductor = Reductor()
    cell = reductor.reduce_cell(complex_cell = complex_cell, reduce_cell = constants.reduce_cell, synapses_list = all_syns,
                                netcons_list = spike_generator.netcons, spike_trains = spike_generator.spike_trains,
                                spike_threshold = constants.spike_threshold, random_state = random_state,
                                var_names = constants.channel_names, reduction_frequency = constants.reduction_frequency, 
                                expand_cable = constants.expand_cable, choose_branches = constants.choose_branches)
                                
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
    func_grps = exc_functional_groups + soma_inhibitory_functional_groups + inhibitory_functional_groups
    for func_grp in func_grps:
      for cluster in func_grp.clusters:
        cluster_type = cluster.center_seg.sec.name().split('.')[1][:4]
        total_num_clusters[cluster_type] += 1 # sort number by type
        total_num_clusters['all'] += 1 # total number
        if total_num_clusters['all'] == cluster_index_to_stim: # this presynaptic cell's turn
          PSC_sec_type = cluster_type
          PSC_syn_type = cluster.synapses[0].get_exc_or_inh_from_syn_type()
          cluster_distance_from_soma = h.distance(cell.soma[0](0.5),cluster.center_seg)
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
    
    
    if constants.optimize_nseg_by_lambda:
        reductor.update_model_nseg_using_lambda(cell, constants.segs_per_lambda)
    if constants.merge_synapses:
        reductor.merge_synapses(cell)
    cell.setup_recorders(vector_length = constants.save_every_ms)

    # Add injections for F/I curve
    if i_amplitude is not None:
        cell.add_injection(sec_index = cell.all.index(cell.soma[0]), record = True, delay = constants.h_i_delay, dur = constants.h_i_delay, amp = i_amplitude)
    
    logger.log_memory()
    logger.log_section_end("Initializing cell model")

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

    logger.log_section_start("Initializing recorder")

    # Record time points
    t_vec = h.Vector(1000 / h.dt).record(h._ref_t)

    # Record membrane voltage of all segments
    V_rec = Recorder(cell.segments, vector_length = constants.save_every_ms)

    logger.log_section_end("Initializing recorder")

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

    # Save constants
    shutil.copy2("constants.py", save_folder)
    os.rename(os.path.join(save_folder, "constants.py"), os.path.join(save_folder, "constants_image.txt"))

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
            
            # Save clamp current
            with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", str(cluster_index_to_stim)+'_'+str(perisomatic)+'_'+PSC_syn_type+'_'+PSC_sec_type+"_Vclamp_i.h5"), 'w') as file:
                file.create_dataset("report/biophysical/data", data = clamp.rec_vec.as_numpy())

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
            clamp.rec_vec.resize(0)

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

if __name__ == "__main__":
    
    if not os.path.exists(constants.save_dir):
        raise FileNotFoundError("No save folder with the given name.")
    
    # make unique output folder for PSCs    
    constants.save_dir = os.path.join(constants.save_dir, 'PSC_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    for np_state in constants.numpy_random_states:
        for neuron_state in constants.neuron_random_states:
            for i_amplitude in constants.h_i_amplitudes:
              for cluster_index_to_stim in range(443,constants.number_of_presynaptic_cells+1):
                print(f"Running for seeds ({np_state}, {neuron_state}); CI = {cluster_index_to_stim}...")
                p = Process(target = main, args=[np_state, neuron_state, i_amplitude, cluster_index_to_stim])
                p.start()
                p.join()
                p.terminate()
                os.system("rm -r x86_64")

    
