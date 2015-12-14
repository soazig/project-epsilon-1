
"""
Purpose:
-----------------------------------------------------------------------------------
We generate convolved hemodynamic neural prediction into seperated txt files for 
all four conditions (task, gain, lost, distance), and also generate plots for 4 
BOLD signals over time for each of them too. We use events2neural_high() in stimuli
because the onsets of condition do not start at TR.

Steps:
-----------------------------------------------------------------------------------
1. Extract 4 conditions of subject __'s all run
2. Gain higher time resolutions
3. Convolve with hrf
4. Plot sampled HRFs with the high resolution neural time course
5. Save to txt files
"""


from __future__ import absolute_import, division, print_function
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from stimuli import *
from scipy.stats import gamma
from organize_behavior_data import *

# Create the necessary directories if they do not exist
dirs = ['../../../txt_output', '../../../txt_output/conv_high_res',\
        '../../../fig','../../../fig/conv_high_res']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
#TODO: the current location for this file project-epsilon/code/scripts
project_path = '../../../'
# TODO: change it to relevant path
data_path = project_path+'data/ds005/'

#change here to get your subject !
subject_list = ['11', '5', '1']
#change here to get your run number !
run_list = [str(i) for i in range(1,4)]
cond_list = [str(i) for i in range(1,5)]

condition_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv_'+ c.zfill(3), \
	data_path + 'sub' + s.zfill(3) + '/model/model001/onsets/task001_run' \
	+ r.zfill(3) + '/cond'+ c.zfill(3) + '.txt') for c in cond_list \
	for r in run_list \
	for s in subject_list]

condition = ['task','gain','loss','dist']
hrf_times = np.arange(0,24,1.0/100)
hrf_at_hr = hrf(hrf_times)

for cond_path in condition_paths:
	name = cond_path[0]
	path = cond_path[1]
	cond = np.loadtxt(path, skiprows = 1)
	# Gain higher time resolutions
	high_res_times, high_cond = events2neural_high(cond)
	# Convolve with hrf
	high_res_hemo = np.convolve(high_cond, hrf_at_hr)[:len(high_cond)]
	tr_indices = np.arange(240)
	tr_times = tr_indices * 2
	# Plot sampled HRFs with the high resolution neural time course
	hr_tr_indices = np.round(tr_indices * 100).astype(int)
	tr_hemo = high_res_hemo[hr_tr_indices]
	plt.plot(tr_times, tr_hemo, label="convolved")
	plt.title(name+'_%s'%(condition[int(name[25])-1]))
	plt.xlabel('Time (seconds)')
	plt.ylabel('Convolved values at TR onsets (condition: %s)'%(condition[int(name[25])-1]))
	plt.legend(loc='lower right')
	plt.savefig(dirs[3]+'/'+ name +'_high_res_.png')
	plt.clf()
	#save the txt file
	np.savetxt(dirs[1] +'/'+ name +'_high_res.txt', tr_hemo)
