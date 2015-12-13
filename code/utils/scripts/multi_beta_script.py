
"""
Purpose:
-----------------------------------------------------------------------------------
We generate beta values for each single voxels for each subject and save them to 
files for multi-comparison test.
-----------------------------------------------------------------------------------

"""


import sys, os
##sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import numpy as np
from glm import *
#from convolution_normal_script import X_matrix
#from convolution_high_res_script import X_matrix_high_res
import nibabel as nib
import matplotlib.pyplot as plt
from smoothing import *

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

conv_list = [str(i) for i in range(1,5)]

txt_paths = [('ds005_sub' + s.zfill(3) + '_t1r1' +'_cond'+ c.zfill(3),\
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv001_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv002_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv003_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv004_canonical.txt', \
              '../../../data/ds005_2/sub' + s.zfill(3) + '/model/model001/task001_run002' \
              #for me (Min) it's ds005_2... the filtered data set
              + '.feat/filtered_func_data_mni.nii.gz',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_001_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_002_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_003_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r1' +'_conv_004_high_res.txt') \
                for s in subject_list \
                for c in conv_list]

for txt_path in txt_paths:
# get 4_d image data
    name = txt_path[0]
    
    img = nib.load(txt_path[5])
    data = img.get_data()
    
    p = 5
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
    
    X_matrix_high_res1 = np.loadtxt(txt_path[6])
    X_matrix_high_res2 = np.loadtxt(txt_path[7])
    X_matrix_high_res3 = np.loadtxt(txt_path[8])
    X_matrix_high_res4 = np.loadtxt(txt_path[9])
    X_matrix_high_res = np.ones((len(X_matrix1),p))
    X_matrix_high_res[...,1] = X_matrix_high_res1
    X_matrix_high_res[...,2] = X_matrix_high_res2
    X_matrix_high_res[...,3] = X_matrix_high_res3
    X_matrix_high_res[...,4] = X_matrix_high_res4

    beta_4d = glm_beta(data,X_matrix)

    # smooth the data
    # use high resolution matrix and re-run the regression
    data_smooth = smoothing(data,1,range(data.shape[-1]))
    beta_4d_smooth_high_res = glm_beta(data_smooth,X_matrix_high_res)
    beta_4d_smooth_high_res_task = beta_4d_smooth_high_res[...,1]
    beta_4d_smooth_high_res_gain = beta_4d_smooth_high_res[...,2]
    beta_4d_smooth_high_res_loss = beta_4d_smooth_high_res[...,3]
    beta_4d_smooth_high_res_dist = beta_4d_smooth_high_res[...,4]

    location_of_txt= dirs[0]
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_task.txt",beta_4d_smooth_high_res_task.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_gain.txt",beta_4d_smooth_high_res_gain.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_loss.txt",beta_4d_smooth_high_res_loss.ravel())
    np.savetxt(location_of_txt + '/' +name[0:17]+ "_beta_dist.txt",beta_4d_smooth_high_res_dist.ravel())
    
