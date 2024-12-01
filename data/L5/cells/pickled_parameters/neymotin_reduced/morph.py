from neuron import h, gui
from math import * #exp,log, sqrt
from axonMorph import *
h.load_file('import3d.hoc')

####################################################################
# Setting up params
####################################################################
# h-current
Vrest       = -90.220509483
h.v_init = -70.0432010302
h.celsius = 34.0 # same as experiment
h.tstop       = 3000
curr        = 0.300 # stimulus amplitude
# passive properties (optimization proxies)
cap         = 0.700445928608
rall        = 137.494564931
rm          = 38457.4393085
spinecapfactor = 1.48057846279
#Na, K reversal potentials calculated from BenS internal and external solutions via Nernst equation
p_ek          = -104
p_ena         = 42
# h-current
h_erev      = -37.0
h_gbar      = 6.6129403774e-05 # mho/cm^2
h_gbar_tuft = 0.00565 # mho/cm^2 (based on Harnett 2015 J Neurosci)
# d-current
kdmc_gbar   = 0.00110907315064
kdmc_gbar_axonm = 20
# spiking currents
nax_gbar    = 0.0153130368342
nax_gbar_axonm = 5
kdr_gbar    = 0.0084715576279
kdr_gbar_axonm = 5
# A few kinetic params changed vis-a-vis kdr.mod defaults:
# These param values match the default values in kdr.mod (except where indicated).
kdr_vhalfn  = 13
kdr_a0n     = 0.0075 # def 0.02
kdr_nmax    = 20 # def 2
kap_gbar    = 0.0614003391814
kap_gbar_axonm = 5
# These param values match the default values in kap.mod (except where indicated).
kap_vhalfn  = 35 # def 11
kap_vhalfl  = -56
kap_nmin    = 0.4 # def 0.1
kap_lmin    = 5 # def 2
kap_tq      = -45 # def -40
# new ion channel parameters
cal_gcalbar = 5.73945708921e-06
can_gcanbar = 4.74427101753e-06
calginc = 1.0
kBK_gpeak = 7.25128017201e-05 # original value of 268e-4 too high for this model
kBK_caVhminShift = 46.9679440782 # shift upwards to get lower effect on subthreshold
cadad_depth = 0.102468419281 # original 1; reduced for tighter coupling of ica and cai
cadad_taur = 16.0181691392
h.erev_ih   = h_erev       # global
# K delayed rectifier current
h.a0n_kdr     = kdr_a0n  # global
h.nmax_kdr    = kdr_nmax # global
# K-A current
h.nmin_kap    = kap_nmin # global
h.lmin_kap    = kap_lmin # global

###############################################################################
# Morph Cell
###############################################################################
class Cell:
    def __init__(self,morph):
        morph = str(morph)
        self.axon = []
        cell = h.Import3d_Neurolucida3()
        cell.input(morph)
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)
        self.add_axon()
        self.init_once()


    def add_axon(self):
        self.axon.append(h.Section(name="axon[0]"))
        self.all.append(self.axon[0])
        self.axon[0].connect(self.soma[0], 0.0, 0.0)
        # clears 3d points
        h.pt3dclear()
        # define a logical connection point relative to the first 3-d point
        h.pt3dstyle(axonPts[0][0], axonPts[0][1], axonPts[0][2], axonPts[0][3], sec=self.axon[0])
        # add axon points after first logical connection point
        for x, y, z, d in axonPts[1:]:
            h.pt3dadd(x, y, z, d, sec=self.axon[0])


    def init_once(self):
        for sec in self.all: #forall within an object just accesses the sections belonging to the object
            sec.insert('pas')   # passive
            sec.insert('savedist')  # mechanism to keep track of distance from soma even after multisplit
            sec.insert('ih')    # h-current in Ih_kole.mod
            sec.insert('nax')   # Na current
            sec.insert('kdr')       # K delayed rectifier current
            sec.insert('kap')       # K-A current
            sec.insert('k_ion')
            sec.insert('na_ion')
            if not (h.ismembrane('ih', sec= sec)):
                print(sec)
        self.setallprop()
        self.addapicchan()
        self.apicchanprop()
        self.addbasalchan()
        self.basalchanprop()
        # spiny:
        #       for i=2, 102 apic[i] spiny.append()
    #       for i=0, 68 dend[i] spiny.append()
    # Skips first two apicals
        for i in range(len(self.apic)):
            if (i != 0) and (i != 1):
                self.apic[i].cm = spinecapfactor * self.apic[i].cm       # spinecapfactor * cm
                self.apic[i].g_pas = spinecapfactor / rm  # spinecapfactor * (1.0/rm)
        for sec in self.dend:
            sec.cm = spinecapfactor * sec.cm
            sec.g_pas = spinecapfactor / rm
        self.optimize_nseg()
        self.setgbarnaxd()
        self.setgbarkapd() # kap in apic,basal dends
        self.setgbarkdrd() # kdr in apic,basal dends
        self.sethgbar()    # distributes HCN conductance
        for sec in self.soma:
            sec.insert('kdmc')
            sec.insert('ca_ion')
            sec.insert('cadad')
            sec.insert('cal')
            sec.insert('can')
            sec.insert('kBK')
        self.setsomag()
        # axon has I_KD, and more I_Na, I_KA, I_KDR, and no I_h
    # K-D current in soma and axon only
        for sec in self.axon:
            sec.insert('kdmc')
        self.setaxong()


    def addapicchan(self):
        for sec in self.apic:
            sec.insert('ca_ion')
            sec.insert('cadad') # cadad.mod
            sec.insert('cal')   # cal.mod
            sec.insert('can')   # can.mod
            sec.insert('kBK')   #kBK.mod


    def apicchanprop(self):
        for sec in self.apic:
            sec.gcalbar_cal = cal_gcalbar
            sec.gcanbar_can = can_gcanbar
            sec.gpeak_kBK = kBK_gpeak
            sec.caVhmin_kBK = -46.08 + kBK_caVhminShift
            sec.depth_cadad = cadad_depth
            sec.taur_cadad = cadad_taur


    def addbasalchan(self):
        # basal == dend
        for sec in self.dend:
            sec.insert('ca_ion')
            sec.insert('cadad') # cadad.mod
            sec.insert('cal')   # cal.mod
            sec.insert('can')   # can.mod
            sec.insert('kBK')   #kBK.mod


    def basalchanprop(self):
        # basal == dend
        for sec in self.dend:
            sec.gcalbar_cal = cal_gcalbar
            sec.gcanbar_can = can_gcanbar
            sec.gpeak_kBK = kBK_gpeak
            sec.caVhmin_kBK = -46.08 + kBK_caVhminShift
            sec.depth_cadad = cadad_depth
            sec.taur_cadad = cadad_taur



    def geom_nseg(self):
        # local freq, d_lambda, before, after, tmp
        # these are reasonable values for most models
        freq = 100 # Hz, frequency at which AC length constant will be computed
        d_lambda = 0.1
        before = 0
        after = 0
        for sec in self.all:
            before += sec.nseg
        #soma area(0.5) # make sure diam reflects 3d points
        for sec in self.all:
            # creates the number of segments per section
            # lambda_f takes in the current section
            sec.nseg = int((sec.L/(d_lambda*self.lambda_f(sec))+0.9)/2)*2 + 1
        for sec in self.all:
            after += sec.nseg

        print("geom_nseg: changed from ", before, " to ", after, " total segments")


    def lambda_f(self, section):
        # these are reasonable values for most models
        freq = 100      # Hz, frequency at which AC length constant will be computed
        d_lambda = 0.1
        # The lowest number of n3d() is 2
        if (section.n3d() < 2):
            return 1e5*sqrt(section.diam/(4*pi*freq*section.Ra*section.cm))
            # above was too inaccurate with large variation in 3d diameter
            # so now we use all 3-d points to get a better approximate lambda
        x1 = section.arc3d(0)
        d1 = section.diam3d(0)
        self.lam = 0
        #print section, " n3d:", section.n3d(), " diam3d:", section.diam3d(0)
        for i in range(section.n3d()): #h.n3d()-1
            x2 = section.arc3d(i)
            d2 = section.diam3d(i)
            self.lam += (x2 - x1)/sqrt(d1 + d2)

            x1 = x2
            d1 = d2

        #  length of the section in units of lambda
        self.lam *= sqrt(2) * 1e-5*sqrt(4*pi*freq*section.Ra*section.cm)
        return section.L/self.lam


    def optimize_nseg(self):
        # Ra, cm, spinecapfactor
        # use worst case Ra, cm values
        for sec in self.all:
            sec.Ra = rall
            sec.cm = cap
        # spiny:
        #       for i=2, 102 apic[i] spiny.append()
    #       for i=0, 68  dend[i] spiny.append()
    # Skips the first two apicals
        for i in range(len(self.apic)):
            if (i != 0) and (i != 1):
                self.apic[i].cm = spinecapfactor * self.apic[i].cm       # spinecapfactor * cm
                self.apic[i].g_pas = spinecapfactor / rm  # spinecapfactor * (1.0/rm)
        for sec in self.dend:
            sec.cm = spinecapfactor * sec.cm
            sec.g_pas = spinecapfactor / rm

        # optimize # of segments per section (do this afer setting Ra, cm)
        self.geom_nseg()
        # need to reassign distances after changing nseg
        self.recalculate_x_dist()


    def setallprop(self):
        for sec in self.all:
            # passive
            sec.g_pas       = 1 / rm
            sec.Ra          = rall
            sec.cm          = cap
            sec.e_pas       = Vrest
            # Na current
            sec.gbar_nax    = nax_gbar
            # K delayed rectifier current
            sec.gbar_kdr    = kdr_gbar
            sec.vhalfn_kdr  = kdr_vhalfn
            # K-A current
            sec.gbar_kap    = kap_gbar
            sec.vhalfn_kap  = kap_vhalfn
            sec.vhalfl_kap  = kap_vhalfl
            sec.tq_kap      = kap_tq
            # reversal potentials
            sec.ena         = p_ena
            sec.ek          = p_ek


    def setaxong(self):
        # axon has I_KD, and more I_Na, I_KA, I_KDR, and no I_h
        for sec in self.axon:
            # axon has no Ih
            sec.gbar_ih = 0
            # increase the I_Na, I_Ka, I_KD and I_KDR
            sec.gbar_nax    = nax_gbar_axonm * nax_gbar
            sec.gbar_kap    = kap_gbar_axonm * kap_gbar
            sec.gbar_kdmc   = kdmc_gbar_axonm * kdmc_gbar
            sec.gbar_kdr    = kdr_gbar_axonm * kdr_gbar


    def setgbarnaxd(self):
        # alldend:
        #       for i=0, 68 dend[i] alldend.append()
        #       for i=0, 102 apic[i] alldend.append()
        for sec in self.dend:
            sec.gbar_nax = nax_gbar
        for sec in self.apic:
            sec.gbar_nax = nax_gbar


    def setgbarkapd(self):
        # alldend:
        #       for i=0, 68 dend[i] alldend.append()
        #       for i=0, 102 apic[i] alldend.append()
        for sec in self.dend:
            sec.gbar_kap = kap_gbar

        for sec in self.apic:
            sec.gbar_kap = kap_gbar


    def setgbarkdrd(self):
        # alldend:
        #       for i=0, 68 dend[i] alldend.append()
        #       for i=0, 102 apic[i] alldend.append()
        for sec in self.dend:
            sec.gbar_kdr = kdr_gbar
        for sec in self.apic:
            sec.gbar_kdr = kdr_gbar


    def setsomag(self):
        for sec in self.soma:
            sec.gbar_kdmc = kdmc_gbar
            sec.gcalbar_cal = cal_gcalbar
            sec.gcanbar_can = can_gcanbar
            sec.gpeak_kBK = kBK_gpeak
            sec.caVhmin_kBK = -46.08 + kBK_caVhminShift
            sec.depth_cadad = cadad_depth
            sec.taur_cadad = cadad_taur


    def recalculate_x_dist(self):
        # set the center of the soma as origin
        h.distance(0,0.5,sec=self.soma[0])
        for sec in self.all:
            for seg in sec.allseg():
                seg.x_savedist = h.distance(seg.x,sec=sec)
        # stays nested to compare soma to apicals
        # list of apicals that are Upper Trunk
        self.nexusdist = nexusdist = 1e9 # calculate distance to tuft here
        apicUpperTrunk =  [23, 24, 25, 32, 33, 46, 51, 52, 53, 78]
        for apicIndx in apicUpperTrunk:
            # excludes any apicals not in the list
            for seg in self.apic[apicIndx].allseg():
                nd = h.distance(seg.x,sec=self.apic[apicIndx])
                if (nd < self.nexusdist):
                    self.nexusdist = nd


    def reconfig (self):
        self.setallprop() # set initial properties, including g_pas,e_pas,cm,Ra,etc.
        self.optimize_nseg() # set nseg based on passive properties
        self.apicchanprop() #  initial apic dend properties
        self.basalchanprop() # initial basal dend properties
        self.setgbarnaxd() # nax in apic,basal dends
        self.setgbarkapd() # kap in apic,basal dends
        self.setgbarkdrd() # kdr in apic,basal dends
        self.sethgbar() # h in all locations
        self.setsomag() # soma-specific conductance values
        self.setaxong() # axon-specific conductance values


    def sethgbar(self):
        h.distance(0,0.5,sec=self.soma[0]) # middle of soma is origin for distance
        self.h_gbar_tuftm = h_gbar_tuftm = h_gbar_tuft / h_gbar
        self.h_lambda = h_lambda = self.nexusdist / log(h_gbar_tuftm)
        for sec in self.soma:
            sec.gbar_ih = h_gbar
        # basal:
        #       basal == all dends
        for sec in self.dend:
            sec.gbar_ih = h_gbar
        # apical_oblique:
    #       for i=79, 102 apic[i]
        for i in range(79,103):
            self.apic[i].gbar_ih = h_gbar
        #apical_maintrunk:
        #       for i=0, 22 apic[i]
        for i in range (0,23):
            for seg in self.apic[i]:
                xd = h.distance(seg.x,sec=sec)
                if (xd <= self.nexusdist):
                    seg.gbar_ih = h_gbar * exp(seg.x_savedist/self.h_lambda)
                else:
                    seg.gbar_ih = h_gbar_tuft
                if (seg.gbar_ih < 0):
                    seg.gbar_ih = 0
        # apical upper trunk list
        apicUpperTrunk =  [23,24,25,32,33,46,51,52,53,78]
        for i in apicUpperTrunk:
            self.apic[i].gbar_ih = h_gbar * h_gbar_tuftm
        # apical tuft:
        #       i=26, 31 apic[i]
            #       i=34, 45 apic[i]
            #       i=47, 50 apic[i]
            #       i=54, 77 apic[i]
        for i in range(26,32):
            self.apic[i].gbar_ih = h_gbar * h_gbar_tuftm
        for i in range(34,46):
            self.apic[i].gbar_ih = h_gbar * h_gbar_tuftm
        for i in range(47,50):
            self.apic[i].gbar_ih = h_gbar * h_gbar_tuftm
        for i in range(54,77):
            self.apic[i].gbar_ih = h_gbar * h_gbar_tuftm



# Initializing cells
#c1 = Cell('BS0284')
#c2 = Cell('BS0409')
# Showing 3D Plot of Cells
#shape_window = h.PlotShape()