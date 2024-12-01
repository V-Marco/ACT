# written by sam neymotin, modified by ernie forzano
from neuron import h
h.load_file("stdrun.hoc")
from pylab import *
import time
from time import time
import datetime # to format time of run
import sys
import pickle
import numpy
#h.install_vecst() # for samp and other NQS/vecst functions
from Build_M1_cell.conf import *
import os

# determine config file name
def setfcfg ():
    fcfg = "PTcell.BS0284.cfg" # default config file name
    for i in range(len(sys.argv)):
        if sys.argv[i].endswith(".cfg") and os.path.exists(sys.argv[i]):
            fcfg = sys.argv[i]
    print("config file is " , fcfg)
    return fcfg

def build_m1_cell():
    fcfg=setfcfg() # config file name
    dconf = readconf(fcfg)
    dprm = dconf['params']
    dfixed = dconf['fixed']
    sampr = dconf['sampr'] # sampling rate

    exec('from ' + dconf['cellimport'] + ' import ' + dconf['cellfunc']) # import cell creation function
    if fcfg.startswith('PTcell'):
        exec('cell = ' + dconf['cellfunc'] + '(' + str(dconf['cellfuncargs']) + ')') # create the cell - can use different morphologies)
    else:
        exec('cell = ' + dconf['cellfunc'] + '()') # create the cell

    exec('import ' + dconf['cellimport']) # import the file's variables too, so can access them

    for p in list(dfixed.values()): exec(p.assignstr(p.origval)) # fixed values
    
    return cell