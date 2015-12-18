
"""
Purpose:
-----------------------------------------------------------------------------------
We generate convolved hemodynamic neural prediction into seperated txt files for 
all four conditions (task, gain, lost, distance), and also generate plots for 4 
BOLD signals over time for each of them too. 

Steps:
-----------------------------------------------------------------------------------
1. Extract 4 conditions of each subject's run
2. Load the data to get the 4th dimension shape
3. Convolve with hrf
4. Plot sampled HRFs with the high resolution neural time course
5. Save the convolved data into txt files
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
dirs = ['../../../txt_output', '../../../txt_output/conv_normal',\
        '../../../fig','../../../fig/conv_normal']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
project_path = '../../../'
data_path = project_path+'data/ds005/'

#change here to get your subject !
subject_list = [str(i) for i in range(1,17)]
#subject_list = ['1','5']
#change here to get your run number !
run_list = [str(i) for i in range(1,4)]
cond_list = [str(i) for i in range(1,5)]
# Loop through conditions by subject and by run
condition_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv'+ c.zfill(3), \
	data_path + 'sub' + s.zfill(3) + '/model/model001/onsets/task001_run' \
	+ r.zfill(3) + '/cond'+ c.zfill(3) + '.txt') for c in cond_list \
	for r in run_list \
	for s in subject_list]

condition = ['task','gain','loss','dist']

#Use the first image to get the data dimensions
image_path = data_path + 'sub001/BOLD/task001_run001/bold.nii.gz'
img = nib.load(image_path)
data_int = img.get_data()
data = data_int.astype(float)
#set the TR
TR = 2.0

#get canonical hrf
tr_times = np.arange(0, data.shape[2], TR)
hrf_at_trs = hrf(tr_times)
n_vols = data.shape[-1]
vol_shape = data.shape[:-1]
all_tr_times = np.arange(data.shape[-1]) * TR

for cond_path in condition_paths:
    name = cond_path[0]
    path = cond_path[1]
    cond = np.loadtxt(path, skiprows = 1)
    neural_prediction = events2neural(cond,TR,n_vols)
    convolved = np.convolve(neural_prediction, hrf_at_trs)
    convolved = convolved[:-(len(hrf_at_trs)-1)]
    #plot
    plt.plot(all_tr_times, neural_prediction, label="neural_prediction")
    plt.plot(all_tr_times, convolved, label="convolved")
    plt.title(name+'_%s'%(condition[int(name[24])-1]))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Convolved values at TR onsets (condition: %s)'%(condition[int(name[24])-1]))
    plt.legend(loc='lower right')

    plt.savefig(dirs[3]+'/'+ name +'_canonical.png')

    plt.close()
    #save the txt file
    np.savetxt(dirs[1] +'/'+ name +'_canonical.txt', convolved)




