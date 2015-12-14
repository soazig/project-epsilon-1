
"""
Purpose:
-----------------------------------------------------------------------------------
We seek the activated voxel positionsi through multi-comparison of beta values across
subjects

Step
-----------------------------------------------------------------------------------
1. calculate the mean of each single beta values across subject and plot them
2. calculate the variance of each single beta values across subject and plot them
3. calculate the t-stat of each single beta values across subject and plot them
4. calculate the p-value of each single betav values across subject and plot them
"""


import sys, os
##sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import numpy as np
from glm import *
import nibabel as nib
import matplotlib.pyplot as plt
#from scipy.stats import sem
from smoothing import *
from scipy.stats import t as t_dist
#from visualization import *


dirs = ['../../../txt_output/multi_beta']

task = dict()
gain = dict()
loss = dict()
dist = dict()

#load all of them
for x in range(1,17):
	task[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_task.txt')

for x in range(1,17):
	gain[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_gain.txt')

for x in range(1,17):
	loss[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_loss.txt')

for x in range(1,17):
	dist[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_dist.txt')

#calculate mean and plot (let's try for task)
task_sum = task[1]
for x in range(2,17):
	task_sum +=task[x]

task_mean = task_sum/16
#task_mean.shape -> (902629,0)

#calculate variance and plot
stdlst = []
for x in range(1,17):
	stdlst.append(task[x])
stdarray = np.array(stdlst)
task_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
task_std_reshape = task_std.reshape(91,109,91)


#calculate t-stat and plot
task_tstat = task_mean/task_std
#task_tstat -> (902629,0)
task_tstat_reshape = task_tstat.reshape(91,109,91)
	#plot task_tstat


#calculate p-value and plot
task_pval = t_dist.cdf(abs(task_tstat), 15)
task_pval_reshape = task_pval.reshape(91,109,91)
1-task_pval
	#plot task







