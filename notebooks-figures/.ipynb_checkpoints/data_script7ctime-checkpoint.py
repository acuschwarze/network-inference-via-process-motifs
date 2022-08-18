#Script for generating data for the pfreq notebook.

import sys, dill, time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sys.path.append('../utils/')
from utils import *

t=time.time()
ctime = float(sys.argv[1])
T = int(sys.argv[2])
idis = float(sys.argv[3])

num_trials = 1000
pfvalues = (1 / T) * np.arange(1, 101, 1)
methods = ['CCO','NTE','GC','CCM','OUI','SCC','SRC']
output = np.zeros((len(pfvalues), num_trials, len(methods)))

t = time.time()
A = make_adjacency(10, 0.5, model='RR')
for i, pf in enumerate(pfvalues):
    for j in range(num_trials):
        for k, me in enumerate(methods):
            res = siminf(A=A, pfreq=pf, 
                initial_displacement=idis, T=T, 
                inference_method=me, ctime=ctime)
            output[i,j,k] = res['accuracy']
np.save('../data/extreme_events7_ctime'+str(ctime)+'_T'+str(T)+'_idis'+str(int(idis))+'.npy', output)
print('Completed in', time.time()-t)


