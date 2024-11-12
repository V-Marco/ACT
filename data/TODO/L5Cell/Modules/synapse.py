from neuron import h

Exp2Syn_syn_params = {
    'e': 0., 
    'tau1': 1.0, 
    'tau2': 3.0
}

LTS_syn_params = {
    'e_GABAA': -90.,
    'Use': 0.3,
    'Dep': 25.,
    'Fac': 100.
}

# Inh perisomatic
FSI_syn_params = {
    'e_GABAA': -90.,
    'Use': 0.3,
    'Dep': 400.,
    'Fac': 0.
}

# Exc choice of two:
CS2CP_syn_params = {
    'tau_d_AMPA': 5.2,
    'Use': 0.41,
    'Dep': 532.,
    'Fac': 65.
}

CP2CP_syn_params = {
    'tau_d_AMPA': 5.2,
    'Use': 0.37,
    'Dep': 31.7,
    'Fac': 519.
}


class Synapse:

    def __init__(self, segment, syn_mod, syn_params, gmax, neuron_r, name):
        
        self.syn_mod = syn_mod

        self.gmax_var = None # Variable name of maximum conductance (uS)
        self.current_type = None
        self.set_gmax_var_and_current_type_based_on_syn_mod(syn_mod)

        self.h_syn = getattr(h, self.syn_mod)(segment)
        self.syn_params = None
        self.set_syn_params(syn_params)
        self.gmax_val = None
        self.set_gmax_val(gmax)

        self.random_generator = None
        self.set_random_generator(neuron_r)

        self.name = name

        # Presynaptic cell
        self.pc = None
        self.netcons = []
    
    def set_spike_train_for_pc(self, mean_fr, spike_train):
        self.pc.set_spike_train(mean_fr, spike_train)
        nc = h.NetCon(self.pc.vecstim, self.h_syn, 1, 0, 1)
        self.netcons.append(nc)

    def set_random_generator(self, r: h.Random) -> None:				 
        if self.syn_mod in ['pyr2pyr', 'int2pyr']:
            r.uniform(0, 1)
            self.h_syn.setRandObjRef(r)
            self.random_generator = r

    def set_syn_params(self, syn_params) -> None:
        self.syn_params = syn_params
        for key, value in syn_params.items():
            if callable(value):
                setattr(self.h_syn, key, value(size = 1))
            else:
                setattr(self.h_syn, key, value)

    def set_gmax_val(self, gmax: float) -> None:
        self.gmax_val = gmax
        setattr(self.h_syn, self.gmax_var, self.gmax_val)

    def set_gmax_var_and_current_type_based_on_syn_mod(self, syn_mod: str) -> None:

        syn_params_map = {
            'AlphaSynapse1': 'gmax',
            'Exp2Syn': '_nc_weight',
            'pyr2pyr': 'initW',
            'int2pyr': 'initW',
            'AMPA_NMDA': 'initW',
            'AMPA_NMDA_STP': 'initW',
            'GABA_AB': 'initW',
            'GABA_AB_STP': 'initW'
        }
        
        current_type_map = {
            'AlphaSynapse1': "i",
            'Exp2Syn': "i",
            'GABA_AB': "i",
            'GABA_AB_STP': "i",
            'pyr2pyr': "iampa_inmda",
            'AMPA_NMDA': 'i_AMPA_i_NMDA',
            'AMPA_NMDA_STP': 'i_AMPA_i_NMDA',
            'int2pyr': 'igaba'
        }
    
        # Set gmax_var based on syn_mod
        if syn_mod in syn_params_map:
            self.gmax_var = syn_params_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")
        
        # Set current_type based on syn_mod
        if syn_mod in current_type_map:
            self.current_type = current_type_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")