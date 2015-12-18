"""
Purpose:
-----------------------------------------------------------
This script creates graphs for t-test for 4 conditions
For each subject each run each condition, plot the t statistics
-----------------------------------------------------------

"""


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

from t_stat import *
from smoothing import *
from matplotlib import colors
from plot_mosaic import * 

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

# Create the necessary directories if they do not exist
dirs = ['../../../fig','../../../fig/t-test']
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# locate the different paths
project_path = '../../../'
data_path = project_path + 'data/'
txt_path = project_path + 'txt_output/conv_high_res/'
#txt_path = project_path + 'txt_output/conv_normal/'
path_dict = {'data_filtered':{
	   		      'folder' : 'ds005/', 
			      'bold_img_name' : 'filtered_func_data_mni.nii.gz',
			      'run_path' : 'model/model001/',
			      'feat' : '.feat/'
			     },
             'data_original':{
			      'folder' : 'ds005/', 
                              'bold_img_name' : 'bold.nii.gz',
                              'run_path' : 'BOLD/',
			      'feat' : '/'
			     }}

# TODO: uncomment for final version
subject_list = [str(i) for i in range(1,17)]
#subject_list = ['1','5']
run_list = [str(i) for i in range(1,4)]
cond_list = [str(i) for i in range(1,5)]

#TODO: Change to relevant path for data or other thing
d = path_dict['data_original']
#OR
#d =  path_dict['data_filtered']
images_paths = [('ds005' +'_sub' + s.zfill(3) + '_t1r' + r, \
                 data_path + d['folder'] + 'sub%s/'%(s.zfill(3)) + d['run_path'] \
                 + 'task001_run%s'%(r.zfill(3))+d['feat']+'%s'%( d['bold_img_name'])) \
                 for r in run_list \
                 for s in subject_list]

print("\n=====================================================")

thres = 375 #from analysis of the histograms
for image_path in images_paths:
    name = image_path[0]
    print("Starting t-test analysis and plot for subject "+name[9:12])
    pdb.set_trace()
    img = nib.load(image_path[1])
    data_int = img.get_data()
    data = data_int.astype(float)
    vol_shape = data.shape[:-1]
    n_trs = data.shape[-1]
    #get the mean value
    mean_data = np.mean(data, axis = -1)
    #build the mask
    in_brain_mask = mean_data > 375
    #smooth the data set
    smooth_data = smoothing(data, 1, range(n_trs))
    #initialize design matrix for t test
    p = 7
    X_matrix = np.ones((data.shape[-1], p))
    #build our design matrix
    for cond in range(1,5):
        convolved = np.loadtxt(txt_path + name + '_conv_' + str(cond).zfill(3) + '_high_res.txt')
	#convolved = np.loadtxt(txt_path + name + '_conv_' + str(cond).zfill(3) + '_canonical.txt')
        X_matrix[:,cond] = convolved
    linear_drift = np.linspace(-1, 1, n_trs)
    X_matrix[:,5] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X_matrix[:,6] = quadratic_drift
    beta, t, df, p = t_stat(smooth_data, X_matrix)
    for cond in range(0,4):
        print("Starting test for condition " + str(i))
        t_newshape = np.reshape(t[cond,:],vol_shape)
        t_newshape[~in_brain_mask]=np.nan
        t_T = np.zeros(vol_shape)
        for z in range(vol_shape[2]):
            t_T[:, :, z] = t_newshape[:,:, z].T
        t_plot = plot_mosaic(t_T)
        plt.imshow(t_plot,interpolation='nearest', cmap='seismic')
        zero_out=max(abs(np.nanmin(t_T)),np.nanmax(t_T))
        plt.title(name+'_t_statistics'+'_cond_'+'_%s'%(cond+1))
        plt.clim(-zero_out,zero_out)
        plt.colorbar()
        plt.savefig(dirs[1]+'/'+ name +'_t-test_'+'cond'+str(cond+1)+'.png')
        plt.close()
print("\nT-test analysis and plots done for selected subjects")
print("See mosaic plots in project-epsilon/fig/t-test/")

