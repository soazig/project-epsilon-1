
"""
Purpose:
-----------------------------------------------------------------------------------
We generate beta values for each single voxels for each subject and save them to 
files for multi-comparison test.
-----------------------------------------------------------------------------------

"""


import sys, os, pdb
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
from glm_func import *
from smoothing import *
from t_stat import *

# Create the necessary directories if they do not exist
dirs = ['../../../txt_output/multi_beta']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
project_path = '../../../'
# TODO: change it to relevant path
conv_path = project_path + 'txt_output/conv_normal/'
conv_high_res_path = project_path + 'txt_output/conv_high_res/'

# select your own subject
subject_list = [str(i) for i in range(1,17)]
#subject_list = ['1','5']

conv_list = [str(i) for i in range(1,5)]

txt_paths = [('ds005_sub' + s.zfill(3) + '_t1r1' +'_cond'+ c.zfill(3),\
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv001_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv002_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv003_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv004_canonical.txt', \
              '../../../data/ds005/sub' + s.zfill(3) + '/model/model001/task001_run002' \
              + '.feat/filtered_func_data_mni.nii.gz',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_001_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_002_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_003_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_004_high_res.txt') \
                for s in subject_list \
                for c in conv_list]

print("\n=======================================================================")
print("Starting multi_betas analysis")
print("Generating the beta values for run 1 of each subject for each condition\n")
for txt_path in txt_paths:
# get 4_d image data
    name = txt_path[0] 
    print("Starting multi_betas analysis for subject " + name[9:12] + " condition " + name[24])
    img = nib.load(txt_path[5])
    data_int = img.get_data()
    data =  data_int.astype(float)
    p = 7
    # p is the number of columns in our design matrix
    # it is the number of convolved column plus 1 (a column of 1's)
    
    X_matrix1 = np.loadtxt(txt_path[1])
    X_matrix2 = np.loadtxt(txt_path[2])
    X_matrix3 = np.loadtxt(txt_path[3])
    X_matrix4 = np.loadtxt(txt_path[4])
    X_matrix = np.ones((len(X_matrix1),p))
    X_matrix[...,1] = X_matrix1
    X_matrix[...,2] = X_matrix2
    X_matrix[...,3] = X_matrix3
    X_matrix[...,4] = X_matrix4
    linear_drift = np.linspace(-1, 1, 240)
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X_matrix[...,5] = linear_drift
    X_matrix[...,6] = quadratic_drift

    # smooth the data
    # use high resolution matrix and re-run the regression
    data_smooth = smoothing(data,1,range(data.shape[-1]))
    beta_3d_smooth, t, df, p = t_stat(data_smooth,X_matrix)
    beta_3d_smooth_task = beta_3d_smooth[...,1]
    beta_3d_smooth_gain = beta_3d_smooth[...,2]
    beta_3d_smooth_loss = beta_3d_smooth[...,3]
    beta_3d_smooth_dist = beta_3d_smooth[...,4]

    location_of_txt= dirs[0]
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_task.txt",beta_3d_smooth_task.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_gain.txt",beta_3d_smooth_gain.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_loss.txt",beta_3d_smooth_loss.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_dist.txt",beta_3d_smooth_dist.ravel())
print("\nAll betas generated from the multi glm analysis")
print("See project-epsilon/" + location_of_txt + " to for the txt files containing the betas")

