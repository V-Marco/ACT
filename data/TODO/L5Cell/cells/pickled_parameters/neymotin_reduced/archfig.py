# code by sam neymotin & ernie forzano
from neuron import h
h.load_file("stdrun.hoc")
from pylab import *
import sys
import pickle
import numpy
h.install_vecst() # for samp and other NQS/vecst functions
from conf import *
import os
from scipy.stats.stats import pearsonr
from utils import dtrans
import shutil

ion()
rcParams['lines.markersize'] = 15
rcParams['lines.linewidth'] = 4
tl = tight_layout

useRMP = False # True # use RMP for fitness calculation?
useVoltDiff = False
useISI = False # this is for evaluation of full isi voltage
useISIFeat = False # this is for evaluation of isi voltage features
useISIDepth = False # this is for evaluation of isi voltage depth (min voltage)
useISIDur = False # this is for evaluation of isi voltage duration
useSag = False # whether to use sag for fitness
useSpikeAmp = False # spike amplitude (peak - treshold voltage) - do not need when using SpikeThresh and SpikePeak
useSpikePeak = False # spike peak (absolute voltage)
useSpikeW = False # spike widths at 25% and 50%
useSpikeSlope = False # min,max dv/dt
useSpikeThresh = False # spike threshold voltage
useSpikeShape = False # overall spike shape - uses features (peak,width,slope,thresh)
useSpikeTimes = useSpikeCoinc = False
useSFA = False # spike-frequency adaptation measure
useLVar = False
useInstRate = False
useTTFS = False # use time-to-first-spike for fitness

#
def getfitdims ():
    fitdims = []
    if useRMP: fitdims.append('RMP')
    if useSag: fitdims.append('Sag')
    if useFI: fitdims.append('FI')
    if useISI: fitdims.append('ISIVolt')
    if useISIFeat: fitdims.append('ISIFeat')
    if useSFA: fitdims.append('SFA')
    if useLVar: fitdims.append('LVar')
    if useInstRate: fitdims.append('InstRate')
    if useTTFS: fitdims.append('TTFS')
    if useSpikeTimes: fitdims.append('SpikeTimes')
    if useSpikeCoinc: fitdims.append('SpikeCoinc')
    if useSpikeAmp: fitdims.append('SpikeAmp')
    if useSpikePeak: fitdims.append('SpikePeak')
    if useSpikeW: fitdims.append('SpikeW')
    if useSpikeSlope: fitdims.append('SpikeSlope')
    if useSpikeThresh: fitdims.append('SpikeThresh')
    if useSpikeShape: fitdims.append('SpikeShape')
    if useVoltDiff: fitdims.append('VoltDiff')
    if useISIDepth: fitdims.append('ISIDepth')
    if useISIDur: fitdims.append('ISIDur')
    return fitdims

# determine config file name
def setfcfg ():
    fcfg = "PTcell.BS0284.cfg" # default config file name
    for i in range(len(sys.argv)):
        if sys.argv[i].endswith(".cfg") and os.path.exists(sys.argv[i]):
            fcfg = sys.argv[i]
    #print "config file is " , fcfg
    return fcfg

dmod = {}

fcfg=setfcfg() # config file name
dconf = readconf(fcfg)
dprm = dconf['params']
dfixed = dconf['fixed']
sampr = dconf['sampr'] # sampling rate
I = numpy.load(dconf['lstimamp'])
evolts = numpy.load(dconf['evolts']) # experimental voltage traces
tte = linspace(0, 1e3*evolts.shape[0]/sampr, evolts.shape[0])
evolts = numpy.load(dconf['evolts']) # experimental voltage traces

useFI=useInstRate=useISI=useSpikeShape=useVoltDiff=True
fitdims=getfitdims()

#
def geterramp (nqa,row,lc):
    err = 0.0
    for c in lc:
        if nqa.fi(c) != -1:
            err += (nqa.getcol(c).x[row] / nqa.getcol(c).mean())**2
    return sqrt(err)

#
def adderrampcol (nqa,lc):
    nqa.tog('DB')
    if nqa.fi('erramp')== -1.0: nqa.resize('erramp'); nqa.pad()
    for i in range(int(nqa.v[0].size())): nqa.getcol('erramp').x[i] = geterramp(nqa,i,lc)
    nqa.stat('erramp') #

# convert population to NQS
def pop2nq (fpop,fitdims=None):
    if fitdims == None: fitdims=getfitdims()
    nqa = None
    try:
        nqa = h.NQS()
    except:
        h.load_file("nqs.hoc"); #h.load_file("decnqs.hoc")
        nqa = h.NQS()
    # first setup the fitness dimensions
    for s in fitdims: nqa.resize(s)
    nqa.clear(len(fpop))
    for m in fpop:
        fit = m.fitness
        for i,val in enumerate(fit): nqa.v[i].append(val)
    # then setup the parameter values
    for k in list(dprm.keys()): nqa.resize(k)
    nqa.pad()
    for i,m in enumerate(fpop):
        idx = len(fitdims)
        prm = m.candidate
        jdx = idx; kdx = 0
        while jdx < nqa.m[0]:
            nqa.v[jdx].x[i] = prm[kdx]
            jdx += 1; kdx += 1
    adderrampcol(nqa,fitdims)
    return nqa

# print out param values (nqa is table, idx is row)
def rowprmstr (nq,idx):
    s = ''
    for i in range(len(fitdims),int(nq.m[0]),1): s += str(nq.v[i].x[idx]) + ' '
    return s

# loads model archive and stores in global ark and nqa objects
def loadark (fn):
    global ark,nqa
    ark = pickle.load(open(fn))
    print(len(ark), ' models in ', fn, ' archive.')
    nqa = pop2nq(ark,fitdims)

if fcfg == 'SPI6.cfg': # simplified model
    useVoltDiff=useFI=useInstRate=useSpikeW=useSpikeSlope=useSpikeThresh=useSpikePeak=useISI=True
    useSpikeShape=False
    fitdims=getfitdims() # reset fitness dimensions(fitdims), which differ from detailed model
    loadark(os.path.join('data','simparch.pkl')) # load simple model archive
else: # detailed model
    loadark(os.path.join('data','detarch.pkl')) # load detailed model archive

# add text to a plot
def addtext (row,col,lgn,ltxt,tx=-0.025,ty=1.03,c='k'):
    for gn,txt in zip(lgn,ltxt):
        ax = subplot(row,col,gn)
        text(tx,ty,txt,fontweight='bold',transform=ax.transAxes,color=c);

def naxbin (ax,nb): ax.locator_params(nbins=nb);

# print full row (fitness and param values) at the given row (idx) from table (nqa)
def rowstr (nq,idx):
    s = ''
    for i in range(int(nq.m[0])): s += nq.s[i].s + ':' + str(nq.v[i].x[idx]) + "\n"
    return s

# print param values at the given row (idx) from table (nqa)
def rowprmvals (nq,idx):
    lval = []
    for i in range(len(fitdims),int(nq.m[0]),1): lval.append((nq.v[i].x[idx]))
    return lval

# find index of f in a (if not there return -1)
def indexof (a,f):
    for i,val in enumerate(a):
        if abs(val-f) < 0.01: return i
    return -1

ISubth = I[0:6] # subthreshold current injections
ISup = I[6:] # current injections for subthresh right before threshold & superthreshold traces
IAll = list(ISubth); IAll.extend(list(ISup))

# draw traces from experiment (uses black color)
def drawexptraces ():
    tx,ty=-.05,1.02; offy = amin(tte[0]) - 30
    ax=gca(); ax.set_xticks([]); ax.set_yticks([]);
    plot([1420,1520],[590,590],'k',linewidth=4)
    plot([1520,1520],[580,590],'k',linewidth=4)
    ypos = offy
    for j,i in enumerate(IAll):
        idx = indexof(I,i)
        plot(tte,evolts[:,idx] + ypos,'k')
        if j > len(ISubth): ypos += 95
        else: ypos += 15

cdx=0 # index into color list
# draw traces from the model (cycles through colors)
def drawtraces (model):
    global cdx
    lclr = ['r','g','b','c','m','y']
    tt = numpy.array(dmod[model]['vt'])
    tx,ty=-.05,1.02; offy = amin(tt[0]) - 30
    if len(get_fignums())==0: drawexptraces()
    mdx=0; m=model
    ax=gca()
    ypos = offy
    for j,i in enumerate(IAll):
        plot(tt, dmod[m][i] + ypos,lclr[cdx%len(lclr)])
        if j > len(ISubth): ypos += 95
        else: ypos += 15
    ax.set_xticks([]); ax.set_yticks([]);
    xlim((400,1600));
    ylim((-125,680));
    cdx+=1

# run model idx using params in ark/nqa, then load/draw the data
def runmodel (idx):
    global lastmodel
    # should move pkl file to arch index location so dont have to rerun
    fnew = os.path.join('data', fcfg.split('.cfg')[0] + '_' + str(idx) + '.pkl')
    if os.path.exists(fnew):
        print('model ' + str(idx) + ' already ran, data in', fnew)
    else:
        cmd = 'python sim.py ' + fcfg + ' ' + rowprmstr(nqa,idx)
        print(cmd)
        os.system(cmd)
        if fcfg.startswith('PTcell'):
            shutil.move(os.path.join('data','morph.pkl'),fnew)
        else:
            shutil.move(os.path.join('data','SPI6.pkl'),fnew)
        if not os.path.exists(fnew):
            print('ERROR: could not run model!')
            return
    lastmodel = (fcfg,idx)
    dmod[lastmodel] = pickle.load(open(fnew)) # load the data
    print('model fitness error/params:', rowstr(nqa,idx))
    drawtraces((fcfg,idx))

#
def drtxt (ax,lett,tx=-0.075,ty=1.03,fsz=45): text(tx,ty,lett,fontweight='bold',transform=ax.transAxes,fontsize=fsz)

# draw archive figure showing param values of bottom/top percentiles
def drawarchfig ():
    if fcfg == 'SPI6.cfg':
        lprm = ['SPI6.gbar_kdmc','SPI6.cal_gcalbar','SPI6.can_gcanbar','SPI6.kBK_gpeak','SPI6.gbar_kap','SPI6.gbar_kdr','SPI6.gbar_nax','SPI6.kBK_caVhminShift','SPI6.cadad_taur','SPI6.cadad_depth','h.vhalfn_kdr','h.vhalfn_kap','h.vhalfl_kap','h.tq_kap']
    else:
        lprm = ['morph.nax_gbar', 'morph.kdmc_gbar','morph.kdr_gbar','morph.kap_gbar','morph.kBK_gpeak','morph.kBK_caVhminShift','morph.cal_gcalbar','morph.can_gcanbar','morph.cadad_taur','morph.cadad_depth']
    draw1dfig(nqa,'erramp',0.01,lprm,nrow=2,ncol=2,gdx=1,stxt='a')
    xlim((0.5,10.5)); ylim((-3,4.5))
    mbotAMP,mtopAMP = getprct(nqa,'erramp',0.01,lprm)
    mcAMP = getprmcors(nqa,'erramp',0.01,lprm)
    ax = subplot(2,2,2)
    imshow(mcAMP,interpolation='None',origin='lower',aspect='auto',extent=(0,mcAMP.shape[0]-1,0,mcAMP.shape[0]-1))
    colorbar(); ax.set_xticks([]); ax.set_yticks([])
    mytxt = 'Worst                         Best'; xlabel(mytxt); ylabel(mytxt);
    text(-0.025,1.03,'b',fontweight='bold',transform=ax.transAxes,color='k');
    title('Parameter correlations')
    draw1dfig(nqa,'FI',0.01,lprm,nrow=2,ncol=2,gdx=3,stxt='c')
    xlim((0.5,10.5)); ylim((-3,4.5))
    mbotFI,mtopFI = getprct(nqa,'FI',0.01,lprm)
    mcFI = getprmcors(nqa,'FI',0.01,lprm)
    ax = subplot(2,2,4)
    imshow(mcFI,interpolation='None',origin='lower',aspect='auto',extent=(0,mcFI.shape[0]-1,0,mcFI.shape[0]-1))
    colorbar(); ax.set_xticks([]); ax.set_yticks([])
    mytxt = 'Worst                         Best'; xlabel(mytxt); ylabel(mytxt);
    text(-0.025,1.03,'d',fontweight='bold',transform=ax.transAxes,color='k');
    title('Parameter correlations')
    subplot(2,2,1); title('Rank by Error Amplitude');
    subplot(2,2,3); title('Rank by FI Error')

#
def draw1dfig (nq,scc,prct,lprm,nrow=1,ncol=1,gdx=1,stxt='a'):
    tx,ty=-0.025,1.03;
    nqt = h.NQS()
    nqt.cp(nq)
    nqt.sort(scc)
    botsidx,boteidx = 0,int(prct*nqt.v[0].size()) # good
    topsidx,topeidx = int(nqt.v[0].size()*(1.0-prct)),int(nqt.v[0].size()-1) # bad
    ax = subplot(nrow,ncol,gdx)
    for pdx,prm in enumerate(lprm):
        dat = numpy.array(nqt.getcol(prm).to_python())
        dat = dat - mean(dat)
        dat = dat / std(dat)
        plot([pdx+1 for j in range(boteidx-botsidx)],dat[botsidx:boteidx],'^',markeredgecolor='m',markerfacecolor='none',markersize=60,linewidth=8)
        plot([pdx+1 for j in range(topeidx-topsidx)],dat[topsidx:topeidx],'v',markeredgecolor='c',markerfacecolor='none',markersize=60,linewidth=8)
    ax.set_xticklabels([dtrans[prm] for prm in lprm])
    ax.set_xticks(linspace(1,len(lprm),len(lprm)))
    ylabel('Normalized parameter value'); #ylim((-4.2,4.2))
    text(tx,ty,stxt,fontweight='bold',transform=ax.transAxes,color='k');
    h.nqsdel(nqt)

# get bottom/top percentile from nq using column scc
def getprct (nq,scc,prct,lprm):
    nqt = h.NQS()
    nqt.cp(nq)
    nqt.sort(scc)
    botsidx,boteidx = 0,int(prct*nqt.v[0].size()) # good
    topsidx,topeidx = int(nqt.v[0].size()*(1.0-prct)),int(nqt.v[0].size()-1) # bad
    mtop = zeros((topeidx-topsidx,len(lprm)))
    mbot = zeros((boteidx-botsidx,len(lprm)))
    for pdx,prm in enumerate(lprm):
        dat = numpy.array(nqt.getcol(prm).to_python())
        dat = dat - mean(dat)
        dat = dat / std(dat)
        mbot[:,pdx] = dat[botsidx:boteidx]
        mtop[:,pdx] = dat[topsidx:topeidx]
    h.nqsdel(nqt)
    return mbot,mtop

# get parameter correlations
def getprmcors (nq,scc,prct,lprm):
    mbot,mtop = getprct(nq,scc,prct,lprm)
    nrow,ncol = mbot.shape
    mprct = zeros((nrow*2,ncol))
    mprct[0:nrow,:] = mbot
    mprct[nrow:,:] = mtop
    mc = ones((nrow*2,nrow*2))
    for i in range(nrow*2):
        for j in range(i+1,nrow*2,1):
            mc[i,j]=mc[j,i]=pearsonr(mprct[i,:],mprct[j,:])[0]
    return mc
