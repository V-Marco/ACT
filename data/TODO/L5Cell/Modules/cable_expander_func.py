import collections
import re
import cmath
import datetime
import numpy as np
import neuron
from neuron import h

from neuron_reduce.subtree_reductor_func import (
	load_model, 
	gather_subtrees, 
	mark_subtree_sections_with_subtree_index, 
	create_segments_to_mech_vals, 
	calculate_nsegs_from_lambda, 
	create_sections_in_hoc, 
	append_to_section_lists, 
	calculate_subtree_q,
	type_of_point_process,synapse_properties_match,
	textify_seg_to_seg,
	Neuron)

from neuron_reduce.reducing_methods import (
	_get_subtree_biophysical_properties, 
	measure_input_impedance_of_subtree, 
	find_lowest_subtree_impedance, 
	find_space_const_in_cm, 
	find_best_real_X)
											
from contextlib import contextmanager

# TODO: Perhaps change duplicate_synapse call so that synapses only go where netcons will go

@contextmanager
def push_section(section):
	'''push a section onto the top of the NEURON stack, pop it when leaving the context'''
	section.push()
	try: yield
	finally: h.pop_section()
		
# can replace Neuron class import with another python cell class

h.load_file("stdrun.hoc")

CableParams = collections.namedtuple(
	'CableParams',
	'length, diam, space_const,'
	'cm, rm, ra, e_pas, electrotonic_length, type, furcation_x'
	)

SynapseLocation = collections.namedtuple('SynapseLocation', 'subtree_index, section_num, x, section_type')

SOMA_LABEL = "soma"
EXCLUDE_MECHANISMS = ('pas', 'na_ion', 'k_ion', 'ca_ion', 'h_ion', 'ttx_ion')

def cable_expander(
	original_cell,
	sections_to_expand,
	furcations_x, 
	nbranches,
	synapses_list,
	netcons_list,
	reduction_frequency,
	model_filename = 'model.hoc',
	total_segments_manual = -1,
	PP_params_dict = None,
	mapping_type = 'impedance',
	return_seg_to_seg = False,
	random_state = None):

	'''
	Receives an instance of a cell with a loaded full morphology, a list of
	synapse objects, a list of NetCon objects (the i'th netcon in the list
	should correspond to the i'th synapse), the filename (string) of the model
	template hoc file that the cell was instantiated from, the desired
	reduction frequency as a float, optional parameter for the approximate
	desired number of segments in the new model (if this parameter is empty,
	the number of segments will be such that there is a segment for every 0.1
	lambda), and an optional param for the point process to be compared before
	deciding on whether to merge a synapse or not and reduces the cell (using
	the given reduction_frequency). Creates a reduced instance using the model
	template in the file whose filename is given as a parameter, and merges
	synapses of the same type that get mapped to the same segment
	(same "reduced" synapse object for them all, but different NetCon objects).
	model_filename : model.hoc  will use a default template
	total_segments_manual: sets the number of segments in the reduced model
							can be either -1, a float between 0 to 1, or an int
							if total_segments_manual = -1 will do automatic segmentation
							if total_segments_manual>1 will set the number of segments
							in the reduced model to total_segments_manual
							if 0>total_segments_manual>1 will automatically segment the model
							but if the automatic segmentation will produce a segment number that
							is lower than original_number_of_segments*total_segments_manual it
							will set the number of segments in the reduced model to:
							original_number_of_segments*total_segments_manual
	return_seg_to_seg: if True the function will also return a textify version of the mapping
						between the original segments to the reduced segments 
	Returns the new reduced cell, a list of the new synapses, and the list of
	the inputted netcons which now have connections with the new synapses.
	Notes:
	1) The original cell instance, synapses and Netcons given as arguments are altered
	by the function and cannot be used outside of it in their original context.
	2) Synapses are determined to be of the same type and mergeable if their reverse
	potential, tau1 and tau2 values are identical.
	3) Merged synapses are assigned a single new synapse object that represents them
	all, but keep their original NetCon objects. Each such NetCon now connects the
	original synapse's NetStim with
	the reduced synapse.
	'''
	for nbranch in nbranches: 
		if type(nbranch) is not int:
			raise TypeError('nbranches must be array of int')

	if PP_params_dict is None: PP_params_dict = {}

	h.init()
	
	model_obj_name = load_model(model_filename)
	
	# Find soma properties
	try: soma = original_cell.soma[0] if original_cell.soma.hname()[-1] == ']' else original_cell.soma
	except: soma = original_cell.soma
	
	soma_cable = CableParams(
		length = soma.L, 
		diam = soma.diam, 
		space_const = None,
		cm = soma.cm, 
		rm = 1.0 / soma.g_pas, 
		ra = soma.Ra, 
		e_pas = soma.e_pas,
		electrotonic_length = None,
		type = 'soma',
		furcation_x = None)

	has_apical = len(list(original_cell.hoc_model.apical)) != 0

	soma_ref = h.SectionRef(sec=soma)
	sections_to_keep, is_section_to_keep_soma_parent, soma_sections_to_keep_x = find_and_disconnect_sections_to_keep(soma,sections_to_expand)

	roots_of_subtrees, num_of_subtrees = gather_subtrees(soma_ref)

	sections_to_delete, section_per_subtree_index, mapping_sections_to_subtree_index = gather_cell_subtrees(sections_to_expand)

	# Remove active conductances and get seg_to_mech dictionary
	segment_to_mech_vals = create_segments_to_mech_vals(sections_to_expand)

	# disconnects all the sections_to_expand from the soma
	subtrees_xs = []
	for section_to_expand in sections_to_expand:
		subtrees_xs.append(section_to_expand.parentseg().x)
		h.disconnect(sec=section_to_expand)

	# Expanding the subtrees
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Branching the section using (d)3/2 and electrotonic length rule to preserve service area and electrical properties.")
	all_trunk_properties=[] #list of all trunk cable properties
	all_branch_properties=[] #list of all branch cable properties
	all_trunk_types=[]
	for i,sec in enumerate(sections_to_expand):
		trunk_properties,branch_properties,trunk_type = expand_cable(sections_to_expand[i], reduction_frequency, furcations_x[i], nbranches[i])
		all_trunk_properties.append(trunk_properties)
		all_branch_properties.append(branch_properties)
		all_trunk_types.append(trunk_type)

	trunk_nsegs = calculate_nsegs_from_lambda(all_trunk_properties)
	branch_nsegs = calculate_nsegs_from_lambda(all_branch_properties)
	
				
	(
		cell, 
		basals, 
		apicals, 
		trunk_sec_type_list_indices, 
		trunks, 
		branches,
		all_expanded_sections,
		number_of_sections_in_apical_list,
		number_of_sections_in_basal_list, 
		number_of_sections_in_axonal_list
	) = create_dendritic_cell(
		soma_cable,
		has_apical,
		original_cell,
		model_obj_name,
		all_trunk_properties, 
		all_branch_properties, 
		nbranches,
		sections_to_expand,sections_to_keep,
		trunk_nsegs, 
		branch_nsegs,
		subtrees_xs)
	

	syn_to_netcon = get_syn_to_netcons(netcons_list) # dictionary mapping netcons to their synapse
	
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Spreading synapses onto branches")
	
	new_synapses_list, subtree_ind_to_q = adjust_new_tree_synapses(
		num_of_subtrees,roots_of_subtrees,
		range(len(sections_to_expand)),
		all_trunk_properties, all_branch_properties, nbranches, furcations_x, all_trunk_types, trunk_sec_type_list_indices,
		PP_params_dict,
		synapses_list, syn_to_netcon,
		mapping_sections_to_subtree_index,
		netcons_list,
		has_apical,
		sections_to_expand,
		original_cell,
		basals, apicals,
		cell,
		reduction_frequency)

	# Check synapses_list with netcons_list
	for netcon in netcons_list:
		syn=netcon.syn()
		if syn not in synapses_list:
			print('Not on the list:', syn, netcon)

	syn_to_netcon = get_syn_to_netcons(netcons_list) # dictionary mapping netcons to their synapse # re call to account for changes.. may need to adjust for efficiency
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"duplicating branch 1 synapses onto the other branches and randomly distributing Netcons")
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"number of reduced synapses before duplicating synapses to branches:",len(new_synapses_list))
	new_synapses_list=distribute_branch_synapses(branches,netcons_list,new_synapses_list,PP_params_dict,syn_to_netcon, random_state) #adjust synapses
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"number of reduced synapses after duplicating synapses to branches:",len(new_synapses_list))
	syn_to_netcon = get_syn_to_netcons(netcons_list) # dictionary mapping netcons to their synapse
	#print("netcon mapping after expansion:",syn_to_netcon)

	# Create segment to segment mapping
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Mapping segments")
	original_seg_to_reduced_seg, reduced_seg_to_original_seg, = create_seg_to_seg(
		original_cell,
		section_per_subtree_index,
		sections_to_expand,
		mapping_sections_to_subtree_index,
		all_trunk_properties, 
		all_branch_properties, 
		furcations_x,
		has_apical,
		apicals,
		basals,
		subtree_ind_to_q,
		mapping_type,
		reduction_frequency,
		trunks, 
		branches)

	# Copy active mechanisms
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Mapping mechanisms")
	copy_dendritic_mech(original_seg_to_reduced_seg,
						reduced_seg_to_original_seg,
						apicals,
						basals,
						segment_to_mech_vals, all_expanded_sections,
						mapping_type)
	
	if return_seg_to_seg: 
		original_seg_to_reduced_seg_text = textify_seg_to_seg(original_seg_to_reduced_seg)

	# Connect disconnected sections back to the soma
	if len(sections_to_keep) > 0:
		for i,sec in enumerate(sections_to_keep):
			if is_section_to_keep_soma_parent[i]:
				soma.connect(sec)
			else:
				sections_to_keep[i].connect(soma, soma_sections_to_keep_x[i])
			
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Deleting original model sections")
	
	# Delete the original model sections
	for section in sections_to_expand:
		with push_section(section):
			h.delete_section()
	
	# Add the sections to the list # not sure if this is needed except maybe for apical?
	if cell.hoc_model.axon is not None:
		cell.axon = cell.hoc_model.axon
	
	if cell.hoc_model.dend is not None:
		cell.dend = cell.hoc_model.dend

	if cell.hoc_model.apic is not None:
		cell.apic = cell.hoc_model.apic
	
	# Put sectios in their section lists
	dends = []
	apics = []
	all_sections = []
	axons = []

	if str(type(cell.soma)) == "<class 'nrn.Section'>":
		soma_sections = [cell.soma]
	else:
		soma_sections = cell.soma
	for soma_sec in soma_sections:
		all_sections.append(soma_sec)
   # get soma children
		if soma_sec.children() != []:
			for soma_child in soma_sec.children(): # Takes care of sections attached to soma
				all_sections.append(soma_child)
				soma_child_sec_type=soma_child.name().split(".")[1][:4]
				if soma_child_sec_type == 'dend':
				  dends.append(soma_child)
				elif soma_child_sec_type == 'apic':
				  apics.append(soma_child)
				elif soma_child_sec_type == 'axon':
				  axons.append(soma_child)
				
        # get children of soma children
				if soma_child.children() != []:
				  for sec_child in soma_child.children(): # Takes care of branches
				    all_sections.append(sec_child)
				    sec_child_sec_type=soma_child.name().split(".")[1][:4]
				    if sec_child_sec_type == 'dend':
				      dends.append(sec_child)
				    elif sec_child_sec_type == 'apic':
				      apics.append(sec_child)
				    elif sec_child_sec_type=='axon':
				      axons.append(sec_child)
				    else:
				      raise(ValueError(f"{sec_child_sec_type} is not dend, apic, or axon"))
                                                       
	cell.all = []	
	cell.dend = []
	cell.apic = []
	cell.axon = []
	for i,sec in enumerate(dends):
		cell.dend.append(sec)
	for i,sec in enumerate(apics):
		cell.apic.append(sec)
	for i,sec in enumerate(all_sections):
		cell.all.append(sec)
	for i,sec in enumerate(axons):
		cell.axon.append(sec)

	# don't think we should delete the soma? 
	#with push_section(cell.hoc_model.soma[0]):
	#    h.delete_section()
	if return_seg_to_seg:
		return cell, new_synapses_list, netcons_list, original_seg_to_reduced_seg_text
	else:
		return cell, new_synapses_list, netcons_list
	  
def apply_params_to_section(name, type_of_sectionlist, instance_as_str, section, cable_params, nseg):
	section.L = cable_params.length
	section.diam = cable_params.diam
	section.nseg = nseg

	append_to_section_lists(name, type_of_sectionlist, instance_as_str)

	section.insert('pas')
	section.cm = cable_params.cm
	section.g_pas = 1.0 / cable_params.rm
	section.Ra = cable_params.ra
	section.e_pas = cable_params.e_pas 
	
def expand_cable(section_to_expand, frequency, furcation_x, nbranch):
	'''expand a cylinder (cable) from the reduced_cell into one trunk and nbranch identical branch sections.
	The expansion is done by finding cable parameters of the trunk and branch.
	Trunk length is chosen using the furcation point.
	Trunk diameter is the same as the cable.
	Branch Diameter is chosen by solving using the 3/2 power rule. (d_trunk)3/2=((d_branch)3/2)/nbranch
	Branch length is chosen so that electrotonic length of the dendritic tree is the same as the cable's electrotonic length.
	Ra and Ri are kept the same.
	'''
	
	# calculate the electrotonic length of the cable
	cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=section_to_expand), frequency)
	cable_space_const_in_cm = find_space_const_in_cm(section_to_expand(0.5).diam/10000, rm, ra)
	cable_elec_L = section_to_expand.L/(cable_space_const_in_cm*10000)
	
	# calculate the diameter of each branch
	trunk_diam = section_to_expand.diam
	branch_diam_in_micron = (trunk_diam**(3/2)/nbranch)**(2/3)
	branch_diam_in_cm = branch_diam_in_micron/10000
	
	# calculate the electrotonic length of each branch
	trunk_elec_L = furcation_x * cable_elec_L
	branch_elec_L = cable_elec_L - trunk_elec_L
	branch_space_const_in_cm = find_space_const_in_cm(branch_diam_in_cm, rm, ra)  # Convert back to cm
	branch_space_const_in_micron = 10000 * branch_space_const_in_cm
	branch_L = branch_elec_L * branch_space_const_in_micron
	
	# calculate the other parameters for each branch
	trunk_diam_in_cm = trunk_diam/10000
	trunk_L = section_to_expand.L*furcation_x
	sec_type = section_to_expand.name().split(".")[1][:4]
	
	# create CableParams objects for the trunk and branch
	trunk_params = CableParams(length=trunk_L, diam=trunk_diam, space_const=cable_space_const_in_cm*10000,
							   cm=cm, rm=rm, ra=ra, e_pas=e_pas, electrotonic_length=trunk_elec_L,
							   type=sec_type, furcation_x=furcation_x)
	
	branch_params = CableParams(length=branch_L, diam=branch_diam_in_micron, space_const=branch_space_const_in_micron,
								cm=cm, rm=rm, ra=ra, e_pas=e_pas, electrotonic_length=branch_elec_L,
								type=sec_type, furcation_x=furcation_x)
	print('branch_L:',branch_L,'|branch_diam:',branch_diam_in_micron,'|trunk_L:',trunk_L,'|trunk_diam:',trunk_diam)
	return trunk_params, branch_params, sec_type
 
def create_dendritic_cell(original_cell, model_obj_name, trunk_cable_properties,
                          branch_cable_properties, nbranches, sections_to_expand,
                          trunk_nsegs, branch_nsegs, subtrees_xs):

    h("objref reduced_dendritic_cell")
    h("reduced_dendritic_cell = new " + model_obj_name + "()")

    trunks = []  # list of trunk sections
    branches = []  # list of branch sections for each trunk
    apicals = []  # if you want to keep track of apical dendrites
    basals = []  # if you want to keep track of basal dendrites

    # Get soma reference
    soma = original_cell.soma

    # Create trunks and branches
    for idx, sec in enumerate(sections_to_expand):
        trunk_cable_params = trunk_cable_properties[idx]
        branch_cable_params = branch_cable_properties[idx]
        trunk_nseg = trunk_nsegs[idx]
        branch_nseg = branch_nsegs[idx]
        nbranch = nbranches[idx]

        # Create trunk section
        trunk = h.Section(name='trunk_' + str(idx))
        # Adjusted call to apply_params_to_section
        apply_params_to_section('trunk_' + str(idx), 'reduced_dendritic_cell', trunk, trunk_cable_params, trunk_nseg)
        trunk.connect(soma(0.5))
        trunks.append(trunk)

        # Create branches for the current trunk
        branches_for_current_trunk = []
        for branch_idx in range(nbranch):
            branch = h.Section(name='branch_' + str(idx) + '_' + str(branch_idx))
            # Adjusted call to apply_params_to_section
            apply_params_to_section('branch_' + str(idx) + '_' + str(branch_idx), 'reduced_dendritic_cell', branch, branch_cable_params, branch_nseg)
            branch.connect(trunk(1))
            branches_for_current_trunk.append(branch)
            # ...

        branches.append(branches_for_current_trunk)

    # Create cell python template
    cell = Neuron(h.reduced_dendritic_cell)
    cell.soma = soma
    cell.trunks = trunks
    cell.branches = branches
    cell.apicals = apicals  # if you have apical dendrites
    cell.basals = basals  # if you have basal dendrites

    return cell, basals, apicals, trunks, branches


   

#def create_dendritic_cell(
#		soma_cable,
#		has_apical,
#		original_cell,
#		model_obj_name,
#		trunk_cable_properties, 
#		branch_cable_properties, 
#		nbranches,sections_to_expand,
#		sections_to_keep,
#		trunk_nsegs, 
#		branch_nsegs,
#		subtrees_xs):
#	
#	h("objref reduced_dendritic_cell")
#	h("reduced_dendritic_cell = new " + model_obj_name + "()")
#
#	create_sections_in_hoc("soma", 1, "reduced_dendritic_cell")
#
#	try: soma = original_cell.soma[0] if original_cell.soma.hname()[-1] == ']' else original_cell.soma
#	except: soma = original_cell.soma
#
#	append_to_section_lists("soma[0]", "somatic", "reduced_dendritic_cell")
#	sec_type_list=[]
#	trunk_sec_type_list = []
#	kept_sec_type_list = []
#	apicals = []
#	basals = []
#	all_expanded_sections = []
#	trunks=[] # list of trunk sections
#	branches=[] # list of branch sections for each trunk [[first trunk's branches][2nd trunk's..]]
#
#	for i, sec in enumerate(sections_to_expand):
#		sec_type=sec.name().split(".")[1][:4] # get section type
#		sec_type_list.append(sec_type) #append trunk sec_type 
#		trunk_sec_type_list.append(sec_type) #append trunk sec_type to its own list
#		#include branches
#		for nbranch in nbranches:
#			for i in range(nbranch):
#				sec_type_list.append(sec_type) # append branches sec_type (same as trunk)
#
#	for i, sec in enumerate(sections_to_keep):
#		sec_type=sec.name().split(".")[1][:4] #get section type
#		sec_type_list.append(sec_type)
#		kept_sec_type_list.append(sec_type)
#
#
#	# Create section lists with the total number of sections for each section type
#	unique_sec_types=[]
#	for sec_type in sec_type_list:
#		if sec_type not in unique_sec_types:
#			unique_sec_types.append(sec_type)
#
#	for unique_sec_type in unique_sec_types:
#		num_sec_type_for_this_unique_sec_type=sec_type_list.count(unique_sec_type)
#		create_sections_in_hoc(unique_sec_type,num_sec_type_for_this_unique_sec_type,"reduced_dendritic_cell")
#		if unique_sec_type=='apic':
#			apicals = [h.reduced_dendritic_cell.apic[i] for i in range(num_sec_type_for_this_unique_sec_type)]
#		elif unique_sec_type == 'dend':
#			basals = [h.reduced_dendritic_cell.dend[i] for i in range(num_sec_type_for_this_unique_sec_type)]
#		elif unique_sec_type == 'axon':
#			axonal = [h.reduced_dendritic_cell.axon[i] for i in range(num_sec_type_for_this_unique_sec_type)]
#		else:
#			raise('error: sec_type', sec_type,' is not "apic" or "dend"')
#
#	# Assemble tree sections
#	number_of_sections_in_apical_list = 0 # count as we add sections since cannot do len(h.reduced_cell.apical)
#	number_of_sections_in_basal_list = 0
#	number_of_sections_in_axonal_list = 0
#	trunk_sec_type_list_indices = []
#
#	for i in range(len(trunk_cable_properties)):
#		trunk_cable_params = trunk_cable_properties[i]
#		branch_cable_params = branch_cable_properties[i]
#		trunk_nseg = trunk_nsegs[i]
#		branch_nseg = branch_nsegs[i]
#		nbranch=nbranches[i]
#		trunk_sec_type=trunk_sec_type_list[i]
#
#		if trunk_sec_type == 'dend': # basal 
#		  #trunk
#			trunk_index=number_of_sections_in_basal_list # trunk index of basal list
#			trunk_cable_params.sec_index_for_type=trunk_index
#			# print('test: trunk_cable_params.sec_index_for_type:',trunk_cable_params.sec_index_for_type) #check is this works
#			apply_params_to_section("dend"+"[" + str(trunk_index) + "]", "basal", "reduced_dendritic_cell",  #apply params to trunk
#								basals[trunk_index], trunk_cable_params, trunk_nseg)
#			basals[trunk_index].connect(soma, subtrees_xs[i], 0) #connect trunk to soma where it was previously connected
#			trunk_sec_type_list_indices.append(trunk_index) #get list of trunk indices for trunk's respective sec_type_list (apic or dend)
#			trunks.append(basals[trunk_index])
#			all_expanded_sections.append(basals[trunk_index])
#			number_of_basal_sections_in_basal_list+=1
#			#branches
#			branches_for_trunk = [] # list of branches for this trunk
#			for j in range(nbranch): #apply branch parameters to next nbranch sections
#					branch_index=number_of_sections_in_apical_list
#					apply_params_to_section("dend"+"[" + str(branch_index) + "]", "basal", "reduced_dendritic_cell",  #apply params to branch
#								basals[branch_index], branch_cable_params, branch_nseg)
#					basals[branch_index].connect(basals[trunk_index], 1, 0) # connect branch to distal end of trunk
#					number_of_sections_in_basal_list+=1
#					branches_for_trunk.append(basals[branch_index])
#					all_expanded_sections.append(basals[branch_index])
#			
#			branches.append(branches_for_trunk)
#
#		elif trunk_sec_type=='apic': # apical
#			#trunk
#			trunk_index=number_of_sections_in_apical_list
#			apply_params_to_section("apic"+"[" + str(trunk_index) + "]", "apical", "reduced_dendritic_cell",  #apply params to trunk
#								apicals[trunk_index], trunk_cable_params, trunk_nseg)
#			apicals[trunk_index].connect(soma, subtrees_xs[i], 0) #connect trunk to soma where it was previously connected
#			trunk_sec_type_list_indices.append(trunk_index) #get list of trunk indices for trunk's respective sec_type_list (apic or dend)
#			trunks.append(apicals[trunk_index])
#			all_expanded_sections.append(apicals[trunk_index])
#			number_of_sections_in_apical_list+=1
#			#branches
#			branches_for_trunk = []
#			for j in range(nbranch): #apply branch parameters to next nbranch sections
#					branch_index=number_of_sections_in_apical_list
#					apply_params_to_section("apic"+"[" + str(branch_index) + "]", "apical", "reduced_dendritic_cell", #apply params to branch
#								apicals[branch_index], branch_cable_params, branch_nseg)
#					apicals[branch_index].connect(apicals[trunk_index], 1, 0) # connect branch to distal end of trunk
#					number_of_sections_in_apical_list+=1
#					branches_for_trunk.append(apicals[branch_index])
#					all_expanded_sections.append(apicals[branch_index])
#			branches.append(branches_for_trunk)
#
#		else:
#			raise(trunk_sec_type,'is not "apic" or "dend"')
#		
#	for i in range(len(sections_to_keep)): #add kept sections to the section lists
#		if kept_sec_type_list[i]=='apic':
#			sec_index=number_of_sections_in_apical_list
#			append_to_section_lists("apic"+"[" + str(sec_index) + "]", "apical", "reduced_dendritic_cell")
#			number_of_sections_in_apical_list+=1
#		elif kept_sec_type_list[i]=='dend':
#			sec_index=number_of_sections_in_basal_list
#			append_to_section_lists("dend"+"[" + str(sec_index) + "]", "basal", "reduced_dendritic_cell")
#			number_of_sections_in_basal_list+=1
#		elif kept_sec_type_list[i]=='axon':
#			sec_index=number_of_sections_in_axonal_list
#			append_to_section_lists("axon"+"[" + str(sec_index) + "]", "axonal", "reduced_dendritic_cell")
#			number_of_sections_in_axonal_list+=1
#		else:
#			raise(kept_sec_type_list[i],'is not "apic" , "dend" , "axon"')
#
#	# Create cell python template
#	cell = Neuron(h.reduced_dendritic_cell)
#	cell.soma = original_cell.soma
#	# cell.apic = apic
#	return cell, basals, apicals, trunk_sec_type_list_indices, trunks, branches, all_expanded_sections, number_of_sections_in_apical_list,number_of_sections_in_basal_list, number_of_sections_in_axonal_list

def find_and_disconnect_sections_to_keep(soma, sections_to_expand):
	'''Searching for sections to keep, they can be a child of the soma or a parent of the soma.'''
	sections_to_keep, is_section_to_keep_soma_parent, soma_sections_to_keep_x  = [], [], []
	soma_ref = h.SectionRef(sec=soma)
	
	for sec in soma.children():
		if sec not in sections_to_expand:
			sections_to_keep.append(sec)
			is_section_to_keep_soma_parent.append(False)
			soma_sections_to_keep_x.append(sec.parentseg().x)
			sec.push()
			h.disconnect()
			h.pop_section()  # Ensure that after disconnecting, you pop the section from the stack
			h.define_shape()

	if soma_ref.has_parent():
		sections_to_keep.append(soma_ref.parent())
		is_section_to_keep_soma_parent.append(True)
		soma_sections_to_keep_x.append(None)
		soma_ref.push()
		h.disconnect()
		h.pop_section()  # Ensure that after disconnecting, you pop the section from the stack

	return sections_to_keep, is_section_to_keep_soma_parent, soma_sections_to_keep_x

  
def gather_cell_subtrees(roots_of_subtrees):
	# dict that maps section indexes to the subtree index they are in: keys are
	# string tuples: ("apic"/"basal", orig_section_index) , values are ints:
	# subtree_instance_index
	sections_to_delete = []
	section_per_subtree_index = {}
	mapping_sections_to_subtree_index = {}
	for i, soma_child in enumerate(roots_of_subtrees):
		# inserts each section in this subtree into the above dict, which maps
		# it to the subtree index
		if 'apic' in soma_child.hname():
			assert i == 0, ('The apical is not the first child of the soma! '
							'a code refactoring is needed in order to accept it')
			mark_subtree_sections_with_subtree_index(sections_to_delete,
													 section_per_subtree_index,
													 soma_child,
													 mapping_sections_to_subtree_index,
													 "apic",
													 i)
		elif 'dend' in soma_child.hname() or 'basal' in soma_child.hname():
			mark_subtree_sections_with_subtree_index(sections_to_delete,
													 section_per_subtree_index,
													 soma_child,
													 mapping_sections_to_subtree_index,
													 "basal",
													 i)

	return sections_to_delete, section_per_subtree_index, mapping_sections_to_subtree_index  
  
def find_synapse_loc(synapse_or_segment, mapping_sections_to_subtree_index):
	''' Returns the location  of the given synapse object'''

	if not isinstance(synapse_or_segment, neuron.nrn.Segment):
		synapse_or_segment = synapse_or_segment.get_segment()

	x = synapse_or_segment.x

	with push_section(synapse_or_segment.sec):
		# extracts the section type ("soma", "apic", "dend") and the section number
		# out of the section name
		full_sec_name = h.secname()
		sec_name_as_list = full_sec_name.split(".")
		short_sec_name = sec_name_as_list[len(sec_name_as_list) - 1]
		section_type = short_sec_name.split("[")[0]
		section_num = re.findall(r'\d+', short_sec_name)[0]
		# print('section_num: ',section_num)

	# finds the index of the subtree that this synapse belongs to using the
	# given mapping_sections_to_subtree_index which maps sections to the
	# subtree indexes that they belong to
	if section_type == "apic":
		subtree_index = mapping_sections_to_subtree_index[("apic", section_num)]
	elif section_type == "dend":
		subtree_index = mapping_sections_to_subtree_index[("basal", section_num)]
	else:  # somatic synapse
		subtree_index, section_num, x = SOMA_LABEL, 0, 0

	return SynapseLocation(subtree_index, int(section_num), x, section_type)

def expand_synapse(cell_instance,
				   synapse_location,
				   on_basal,
				   imp_obj,
				   root_input_impedance,
				   trunk_properties,branch_properties,furcation_x,
				   q_subtree):
	'''
	Receives an instance of a cell, the location (section + relative
	location(x)) of a synapse to be reduced, a boolean on_basal that is True if
	the synapse is on a basal subtree, the number of segments in the reduced
	cable that this synapse is in, an Impedance calculating Hoc object, the
	input impedance at the root of this subtree, and the electrotonic length of
	the reduced cable that represents the current subtree
	(as a real and as a complex number) -
	and maps the given synapse to its new location on the reduced cable
	according to the NeuroReduce algorithm.  Returns the new "post-merging"
	relative location of the synapse on the reduced cable (x, 0<=x<=1), that
	represents the middle of the segment that this synapse is located at in the
	new reduced cable.
	'''
	# measures the original transfer impedance from the synapse to the
	# somatic-proximal end in the subtree root section
	if synapse_location.section_type=='apic':  # apical subtree
		# print('not on_basal','|synapse_location.section_num: ',synapse_location.section_num)
		try: section = cell_instance.apic[synapse_location.section_num]
		except: 
			if synapse_location.section_num==0:
					section=cell_instance.apic
					# print(section)
			else:
				raise(print('Exception led to error. Check cell_instance.apic'))
	elif synapse_location.section_type=='dend':             # basal subtree
		section = cell_instance.dend[synapse_location.section_num]
	else:
		raise(print('synapse_location.section_type not "apic" or "dend"'))
	# print('section: ',section)

	with push_section(section):
		orig_transfer_imp = imp_obj.transfer(synapse_location.x) * 1000000  # ohms
		orig_transfer_phase = imp_obj.transfer_phase(synapse_location.x)
		# creates a complex Impedance value with the given polar coordinates
		orig_synapse_transfer_impedance = cmath.rect(orig_transfer_imp, orig_transfer_phase)

	# synapse location could be calculated using:
	# X = L - (1/q) * arcosh( (Zx,0(f) / ZtreeIn(f)) * cosh(q*L) ),
	# derived from Rall's cable theory for dendrites (Gal Eliraz)
	# but we chose to find the X that will give the correct modulus. See comment about L values

	elec_L_dend=trunk_properties.electrotonic_length+branch_properties.electrotonic_length


	synapse_new_electrotonic_location = find_best_real_X(root_input_impedance,
														 orig_synapse_transfer_impedance,
														 q_subtree,
														 elec_L_dend)
														 
	#relative location along entire dendrite                                                     
	new_relative_loc_in_section = (float(synapse_new_electrotonic_location) /
								   elec_L_dend)
	#determine x loc is  trunk or branch
	if new_relative_loc_in_section<furcation_x: #trunk
		on_trunk=True
		new_relative_loc_in_section = new_relative_loc_in_section/furcation_x #adjust for section x loc
	else: #branch case
		on_trunk=False
		branch_elec_L_for_synapse = synapse_new_electrotonic_location-trunk_properties.electrotonic_length
		# solve branch_elec_L_for_synapse = branch_syn_L/branch_space_const for branch_syn_L (the length up the branch to the synapses electrotonic length)
		branch_L_for_synapse = branch_elec_L_for_synapse*branch_properties.space_const
		# find proportionate length for x doing L_syn/L_branch
		new_relative_loc_in_section = branch_L_for_synapse/branch_properties.length

	if new_relative_loc_in_section > 1:  # PATCH
		new_relative_loc_in_section = 0.999999

	return new_relative_loc_in_section, on_trunk

def find_branch_synapse_X(cell_instance,
				   synapse_location,
				   on_basal,
				   imp_obj,
				   root_input_impedance,
				   new_cable_electrotonic_length,
				   q_subtree,
				   trunk_properties, branch_properties):
	'''
	Receives an instance of a cell, the location (section + relative
	location(x)) of a synapse to be reduced, a boolean on_basal that is True if
	the synapse is on a basal subtree, the number of segments in the reduced
	cable that this synapse is in, an Impedance calculating Hoc object, the
	input impedance at the root of this subtree, and the electrotonic length of
	the reduced cable that represents the current subtree
	(as a real and as a complex number) -
	and maps the given synapse to its new location on the reduced cable
	according to the NeuroReduce algorithm.  Returns the new "post-merging"-
	relative location of the synapse on the reduced cable (x, 0<=x<=1), that
	represents the middle of the segment that this synapse is located at in the
	new reduced cable.
	'''
	# measures the original transfer impedance from the synapse to the
	# somatic-proximal end in the subtree root section
	if not on_basal:  # apical subtree
		# print('not on_basal')
		try: section = cell_instance.apic[synapse_location.section_num]
		except: 
			if 0==synapse_location.section_num:
					section=cell_instance.apic
					# print(section)
			else:
				raise(print('Exception led to error. Check cell_instance.apic'))
	else:             # basal subtree
		section = cell_instance.dend[synapse_location.section_num]
	# print('section: ',section)
	with push_section(section):
		orig_transfer_imp = imp_obj.transfer(synapse_location.x) * 1000000  # ohms
		orig_transfer_phase = imp_obj.transfer_phase(synapse_location.x)
		# creates a complex Impedance value with the given polar coordinates
		orig_synapse_transfer_impedance = cmath.rect(orig_transfer_imp, orig_transfer_phase)

	# synapse location could be calculated using:
	# X = L - (1/q) * arcosh( (Zx,0(f) / ZtreeIn(f)) * cosh(q*L) ),
	# derived from Rall's cable theory for dendrites (Gal Eliraz)
	# but we chose to find the X that will give the correct modulus. See comment about L values

	synapse_new_electrotonic_location = find_best_real_X(root_input_impedance,
														 orig_synapse_transfer_impedance,
														 q_subtree,
														 new_cable_electrotonic_length)
	#solve syn_elec_L=trunk_elec_L+branch_elec_L for branch_elec_L
	branch_elec_L_for_synapse = synapse_new_electrotonic_location-trunk_properties.electrotonic_length
	# solve branch_elec_L_for_synapse = branch_syn_L/branch_space_const for branch_syn_L (the length up the branch to the synapses electrotonic length)
	branch_L_for_synapse = branch_elec_L_for_synapse*branch_properties.space_const
	# find proportionate length for x doing L_syn/L_branch
	new_relative_loc_in_section = branch_L_for_synapse/branch_properties.length

	if new_relative_loc_in_section > 1:  # PATCH
		new_relative_loc_in_section = 0.999999

	return new_relative_loc_in_section
  
def adjust_new_tree_synapses(num_of_subtrees, roots_of_subtrees,
						   num_sections_to_expand,
						   trunk_properties, branch_properties, nbranches, furcations_x, all_trunk_sec_type, trunk_sec_type_list_indices, #list of indices for dend[], apic[] of trunk sections
						   PP_params_dict,
						   synapses_list,syn_to_netcon,
						   mapping_sections_to_subtree_index,
						   netcons_list,
						   has_apical,
						   sections_to_expand,
						   original_cell,
						   basals, apicals,
						   cell,
						   reduction_frequency):
	# dividing the original synapses into baskets, so that all synapses that are
	# on the same subtree will be together in the same basket

	# a list of baskets of synapses, each basket in the list will hold the
	# synapses of the subtree of the corresponding basket index
#     print('num_sections_to_expand:',num_sections_to_expand)
	baskets = [[] for _ in num_sections_to_expand]
	soma_synapses_syn_to_netcon = {}

	new_synapses_list, subtree_ind_to_q = [], {}

	for syn_index, synapse in enumerate(synapses_list):
		if synapse.get_segment().sec in sections_to_expand:
			synapse_location = find_synapse_loc(synapse, mapping_sections_to_subtree_index)
		
		# For a somatic synapse
		# TODO: 'axon' is never returned by find_synapse_loc...
		if synapse_location.subtree_index in (SOMA_LABEL, 'axon'):
			soma_synapses_syn_to_netcon[synapse] = netcons_list[syn_index]
		else:
			baskets[synapse_location.subtree_index].append((synapse, synapse_location, syn_index))

		# Not sure what it is
		# else: #leave synapses not on new trees synapses alone
		# new_synapses_list.append(synapse)

	for section_to_expand_index in range(len(sections_to_expand)):
		imp_obj, subtree_input_impedance = measure_input_impedance_of_subtree(
			sections_to_expand[section_to_expand_index], reduction_frequency)
		subtree_ind_to_q[section_to_expand_index] = calculate_subtree_q(
			sections_to_expand[section_to_expand_index], reduction_frequency)
		
		trunk_index = trunk_sec_type_list_indices[section_to_expand_index]
		x_furcation = furcations_x[section_to_expand_index]
		# iterates over the synapses in the curr basket
		for synapse, synapse_location, syn_index in baskets[section_to_expand_index]:
			# get trunk synapses
			if synapse_location.x < x_furcation: #synapse proximal to furcation point is on trunk
				#locate this trunk section
				if all_trunk_sec_type[section_to_expand_index]=='dend':
					section_for_synapse = basals[trunk_index] #get the trunk section
				elif all_trunk_sec_type[section_to_expand_index]=='apic':
					section_for_synapse = apicals[trunk_index]
				else:
					raise(all_trunk_sec_type[section_to_expand_index],' is not "apic" or "dend"')
				
				# Adjust x location since trunk is fraction of cable length
				x = synapse_location.x/x_furcation
			
			else: #synapse is distal to furcation meaning on branch
				nbranch=nbranches[section_to_expand_index] # number of branches on this tree
				branch_index=trunk_index+1 #select first branch on this tree to move synapse (later to distribute to each branch)
				if all_trunk_sec_type[section_to_expand_index]=='dend':
					section_for_synapse = basals[branch_index]
				
				elif all_trunk_sec_type[section_to_expand_index]=='apic':
					section_for_synapse = apicals[branch_index]
					
				else:
					raise(all_trunk_sec_type[section_to_expand_index],' is not "apic" or "dend"')
			  
			  # Adjust x location to the point on the branch that has the same electrotonic length as originally
				dend_elec_L=trunk_properties[section_to_expand_index].electrotonic_length+branch_properties[section_to_expand_index].electrotonic_length
				on_basal_subtree = not (has_apical and section_to_expand_index == 0)
				x = find_branch_synapse_X(
  				original_cell,
  				synapse_location,
  				on_basal_subtree,
  				imp_obj,
  				subtree_input_impedance,
  				dend_elec_L,
  				subtree_ind_to_q[section_to_expand_index],
  				trunk_properties=trunk_properties[section_to_expand_index], branch_properties=branch_properties[section_to_expand_index])
			  
			if (x > 1) or (x < 0):
				raise(ValueError(f"{x} is not between 0 and 1 for {original_cell}, {synapse_location}, {on_basal_subtree}, {imp_obj}, {dend_elec_L}, {trunk_properties}"))
			# go over all point processes in this segment and see whether one
			# of them has the same proporties of this synapse
			# If there's such a synapse link the original NetCon with this point processes
			# If not, move the synapse to this segment.
			for PP in section_for_synapse(x).point_processes():
				# print(PP_params_dict)
				if type_of_point_process(PP) not in PP_params_dict:
					print("adding",PP,"to PP_params_dict")
					add_PP_properties_to_dict(PP, PP_params_dict)

				if synapse_properties_match(synapse, PP, PP_params_dict):
					#netcons_list[syn_index].setpost(PP) #this does not work because there is no loger 1:1 correspondence between netcon and synapse
					for netcon in syn_to_netcon[synapse]:
						netcon.setpost(PP)
					break
			else:  # If for finish the loop -> first appearance of this synapse
				##########testing
				if type_of_point_process(synapse) not in PP_params_dict:
					print("adding",synapse,"to PP_params_dict")
					add_PP_properties_to_dict(synapse, PP_params_dict)
				#########
				#print("moving", synapse, "from",synapse_location,"to",section_for_synapse(x))              
				#synapse.loc(x, sec=section_for_synapse)
				#new_synapses_list.append(synapse)
				# Unlink the synapse from its current section
				# Set the synapse to the new section and specify its location
				synapse.loc(x, sec=section_for_synapse)  # This should automatically adjust the section and location of the synapse
				new_synapses_list.append(synapse)

	# merging somatic and axonal synapses
	synapses_per_seg = collections.defaultdict(list)
	for synapse in soma_synapses_syn_to_netcon:
		seg_pointer = synapse.get_segment()

		for PP in synapses_per_seg[seg_pointer]:
			if type_of_point_process(PP) not in PP_params_dict:
				add_PP_properties_to_dict(PP, PP_params_dict)

			if synapse_properties_match(synapse, PP, PP_params_dict):
				soma_synapses_syn_to_netcon[synapse].setpost(PP)
				break
		else:  # If for finish the loop -> first appearance of this synapse
			synapse.loc(seg_pointer.x, sec=seg_pointer.sec)
			new_synapses_list.append(synapse)
			synapses_per_seg[seg_pointer].append(synapse)

	return new_synapses_list, subtree_ind_to_q
  
def create_seg_to_seg(original_cell,
					  section_per_subtree_index,
					  sections_to_expand,
					  mapping_sections_to_subtree_index,
					  all_trunk_properties, all_branch_properties,furcations_x,
					  has_apical,
					  apicals,
					  basals,
					  subtree_ind_to_q,
					  mapping_type,
					  reduction_frequency,
					  trunks, branches):
	'''create mapping between segments in the original model to segments in the reduced model
	   if mapping_type == impedance the mapping will be a response to the
	   transfer impedance of each segment to the soma (like the synapses)
	   if mapping_type == distance  the mapping will be a response to the
	   distance of each segment to the soma (like the synapses) NOT IMPLEMENTED
	   YET
	   '''

	assert mapping_type == 'impedance', 'distance mapping not implemented yet'
	# the keys are the segments of the original model, the values are the
	# segments of the reduced model
	original_seg_to_expanded_seg = collections.defaultdict(list) #originally these two dictionaires were flipped
	expanded_seg_to_original_seg = collections.defaultdict(list)
	subtree_index=0
	for sec in sections_to_expand:
			for seg in sec:
				synapse_location = find_synapse_loc(seg, mapping_sections_to_subtree_index)
				imp_obj, cable_input_impedance = measure_input_impedance_of_subtree(
					sec, reduction_frequency)

				# if synapse is on the apical subtree
				on_basal_cable = not (has_apical and subtree_index == 0)

				mid_of_segment_loc, on_trunk = expand_synapse(
					original_cell,
					synapse_location,
					on_basal_cable,
					imp_obj,
					cable_input_impedance,
					all_trunk_properties[subtree_index], all_branch_properties[subtree_index],furcations_x[subtree_index],
					subtree_ind_to_q[subtree_index])
				if on_trunk:
					new_section_for_synapse = trunks[subtree_index] # returns trunk section
				else:
					new_section_for_synapse = branches[subtree_index] # returns list of branch sections for each trunk
				if on_trunk == False: # case for mapping to branches
					expanded_seg = [None] * len(new_section_for_synapse) #initial array for branches
					for i in range(len(new_section_for_synapse)):
						new_section=new_section_for_synapse[i]
						expanded_seg[i] = new_section(mid_of_segment_loc)
						original_seg_to_expanded_seg[seg].append(expanded_seg)
						expanded_seg_to_original_seg[expanded_seg[i]].append(seg)
				else: #normal case
					expanded_seg = new_section_for_synapse(mid_of_segment_loc)
					original_seg_to_expanded_seg[seg].append(expanded_seg) 
					expanded_seg_to_original_seg[expanded_seg].append(seg)
			subtree_index+=1
	
	return original_seg_to_expanded_seg, dict(expanded_seg_to_original_seg)
  
def copy_dendritic_mech(original_seg_to_reduced_seg,
						reduced_seg_to_original_seg,
						apicals,
						basals,
						segment_to_mech_vals, all_expanded_sections,
						mapping_type='impedance'):
	''' copies the mechanisms from the original model to the reduced model'''

	# copy mechanisms
	# this is needed for the case where some segements were not been mapped
	mech_names_per_segment = collections.defaultdict(list)
	vals_per_mech_per_segment = {}
	for reduced_seg, original_segs in reduced_seg_to_original_seg.items():
		vals_per_mech_per_segment[reduced_seg] = collections.defaultdict(list)

		for original_seg in original_segs:
			for mech_name, mech_params in segment_to_mech_vals[original_seg].items():
				for param_name, param_value in mech_params.items():
					vals_per_mech_per_segment[reduced_seg][param_name].append(param_value)

				mech_names_per_segment[reduced_seg].append(mech_name)
				reduced_seg.sec.insert(mech_name)

		for param_name, param_values in vals_per_mech_per_segment[reduced_seg].items():
			setattr(reduced_seg, param_name, np.mean(param_values))

	all_segments = []
	for sec in all_expanded_sections:
		for seg in sec:
			all_segments.append(seg)
	
	if len(all_segments) != len(reduced_seg_to_original_seg):
		logger.warning('There is no segment to segment copy, it means that some segments in the'
					'reduced model did not receive channels from the original cell.'
					'Trying to compensate by copying channels from neighboring segments')
		handle_orphan_segments(original_seg_to_reduced_seg,
							   all_segments,
							   vals_per_mech_per_segment,
							   mech_names_per_segment)
		
		
def distribute_branch_synapses(branch_sets, netcons_list, synapses_list, PP_params_dict, syn_to_netcon, random_state):
    ''' 
    Distributes synapses among branches based on netcons.
    '''
    num_duplicated_synapses=0
    for branch_set in branch_sets:
        branch_with_synapses = branch_set[0]
        for seg in branch_with_synapses:
            for synapse in seg.point_processes():
                x = synapse.get_loc()
                h.pop_section()

                synapses_for_netcons = [synapse]  # start with original synapse as an option for netcon
                
                associated_netcons = syn_to_netcon.get(synapse, [])
                for netcon in associated_netcons:
                    # Randomly select a branch for the netcon
                    selected_branch = random_state.choice(branch_set)
                    
                    # Check if a synapse of the same type already exists at the x location on the selected branch
                    existing_synapse = None
                    for pp in selected_branch(x).point_processes():
                        if synapse_properties_match(synapse, pp, PP_params_dict):
                            existing_synapse = pp
                            break

                    # If synapse doesn't exist, duplicate it
                    if not existing_synapse:
                        num_duplicated_synapses+=1
                        duplicated_synapse = duplicate_synapse(synapse, seg, PP_params_dict)
                        duplicated_synapse.loc(selected_branch(x))
                        synapses_for_netcons.append(duplicated_synapse)
                        synapses_list.append(duplicated_synapse)
                    else:
                        duplicated_synapse = existing_synapse

                    # Point the netcon to the chosen/duplicated synapse
                    netcon.setpost(duplicated_synapse)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Finish distributing synapses to branches and duplicating synapse. {num_duplicated_synapses} new synapses. {len(synapses_list)} total synapses.")
    return synapses_list



def duplicate_synapse(synapse,seg,PP_params_dict):
	'''
	creates a new synapse object with the same parameters as the given synapse object
	uses the dictionary for:
	'''
	syn_type = synapse.hname().split('[')[0]  # Remove index from syn_type
	new_synapse = getattr(h, syn_type)(seg)
	params_were_same = []
	for param_name in PP_params_dict[syn_type]:
			param_value = getattr(synapse, param_name)
			if getattr(new_synapse, param_name) != param_value:
				#if param_name in params_were_same:
				#print("Keep",param_name)
				try: setattr(new_synapse, param_name, param_value)
				except: raise AttributeError('Cannot set',new_synapse,'attribute',param_name,'to',param_value,'may try including attribute in skipped_params for PP_params_dict')
			else:
				if param_name not in params_were_same: params_were_same.append(param_name)
	return new_synapse
		   
def redistribute_netcons(synapse, target_synapses, syn_to_netcon, random_state):
	'''randomly chooses a new synapse among the original and new choices to point the netcon to
	target_synapses: list of new synapses
	'''
	for netcon in syn_to_netcon[synapse]: # redistribute netcons
		rand_index = random_state.randint(0, len(target_synapses)+1) #choose random branch to move point netcon to
		if rand_index==0: #if 0, keep netcon on original synapse
			continue
		else:
			netcon.setpost(target_synapses[rand_index-1]) #find corresponding synapse #point netcon toward synapse

def get_syn_to_netcons(netcons_list):
	syn_to_netcon = {} # dictionary mapping netcons to their synapse
	for netcon in netcons_list: #fill in dictionary
		syn = netcon.syn() # get the synapse that netcon points to
		if syn in syn_to_netcon:
			syn_to_netcon[syn].append(netcon) #add netcon to existing synapse key
		else:
			syn_to_netcon[syn] = [netcon] #create new synapse key using netcon as an item
	return syn_to_netcon
	   
		
		
def add_PP_properties_to_dict(PP, PP_params_dict):
	"""
	add the properties of a point process to PP_params_dict.
	The only properties added to the dictionary are those worth comparing
	attributes not worth comparing are not synapse properties or do not differ in value.
	"""
	skipped_params = {
		"Section", "allsec", "baseattr", "cas", "g", "get_loc", "has_loc", "hname",
		'hocobjptr', "i", "loc", "next", "ref", "same", "setpointer", "state",
		"get_segment", "DA1", "eta", "omega", "DA2", "NEn", "NE2", "GAP1", "unirand", "randGen", "sfunc", "erand", 
		"randObjPtr", "A_AMPA", "A_NMDA", "B_AMPA", "B_NMDA", "D1", "D2", "F", "P", "W_nmda", "facfactor", "g_AMPA", "g_NMDA", "iampa", "inmda", "on_ampa", "on_nmda", "random",  "thr_rp","AlphaTmax_gaba", "Beta_gaba", "Cainf", "Cdur_gaba", "Erev_gaba", "ICag", "Icatotal", "P0g", "W", "capoolcon", "destid", "fCag", "fmax", "fmin", "g_gaba", "gbar_gaba", "igaba", "limitW", "maxChange", "neuroM", "normW", "on_gaba", "pooldiam", "postgid", "pregid", "r_gaba", "r_nmda", "scaleW", "srcid", "tauCa", "type", "z",
		"d1", "gbar_ampa", "gbar_nmda","tau_d_AMPA","tau_d_NMDA","tau_r_AMPA","tau_r_NMDA","Erev_ampa","Erev_nmda", "lambda1", "lambda2", "threshold1", "threshold2",
	}

	syn_params_list = {
		"tau_r_AMPA", "tau_r_NMDA", "Use", "Dep", "Fac", "e", "u0", "initW", "taun1", "taun2", "gNMDAmax", "enmda", "taua1", "taua2", "gAMPAmax", "eampa", "AlphaTmax_ampa", "Beta_ampa", "Cdur_ampa", "AlphaTmax_nmda", "Beta_nmda", "Cdur_nmda", "initW_random", "Wmax", "Wmin", "tauD1", "tauD2", "f", "tauF", "P_0", "d2",
	}

	PP_params = [param for param in dir(PP) if not (param.startswith("__") or callable(getattr(PP, param)))]

	PP_params = list(filter(lambda x: x not in skipped_params, PP_params))

	syn_params = list(filter(lambda x: x in syn_params_list, PP_params))


	PP_params_dict[type_of_point_process(PP)] = syn_params
	# print(PP_params_dict)
