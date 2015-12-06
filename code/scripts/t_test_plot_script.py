"""
ADD DESCRIPTION
"""

import sys, os
sys.path.append("../utils")
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

from t_test import *
from find_activated_voxel_functions import *
from convolution_normal_script import X_matrix
from scipy.ndimage import gaussian_filter
from matplotlib import colors


# Create the necessary directories if they do not exist
dirs = ['../../txt_output', '../../txt_output/FOLDERS_NAME',\
        '../../fig','../../fig/FOLDER_NAME']
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# Locate the different paths
project_path = '../../'
data_path = project_path + 'data/ds005/'

# TODO: uncomment for final version
#subject_list = [str(i) for i in range(1,17)]
subject_list = ['1']
run_list = [str(i) for i in range(1,2)]

#TODO: Change to relevant path for data or other thing
images_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r, \
                 dir_path + 'sub' + s.zfill(3) + '/BOLD/task001_run' \
                 + r.zfill(3) + '/bold.nii.gz') for r in run_list \
                 for s in subject_list]

#TODO: remove the following
location_of_data = "../../data/ds005/sub001/BOLD/task001_run001/"
location_of_plot = "../../plots/"


#TODO:See the script hist-mosaic for plot

img = nib.load(location_of_data+ "bold.nii")
data = img.get_data()
data = data[4:,]
smooth_data = gaussian_filter(data, [2, 2, 2, 0])
beta, t, df,p=t_test(smooth_data,X_matrix)
vol_shape, n_trs = data.shape[:-1], data.shape[-1]

#find mask boolean vectors
mean_data = np.mean(data,axis=-1)
in_brain_mask = mean_data > 400

#nice map
nice_cmap_values = np.loadtxt('actc.txt')
nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')

#set up the label font size
matplotlib.rc('xtick', labelsize=5) 
matplotlib.rc('ytick', labelsize=5) 

#draw heat map
t1 = np.reshape(t[1,:],vol_shape)
t1[~in_brain_mask]=np.nan
for i in range(34):
 plt.subplot(5,7,i+1)
 plt.imshow(t1[:,:,i],cmap = nice_cmap, alpha=0.5)
 plt.title("Slice"+str(i+1), fontsize=5)
 plt.tight_layout()

plt.suptitle("Subject 1 Run 1 T Statistics in Condition 1 for different Slices\n")
plt.colorbar()
plt.savefig(location_of_plot+"t_statistics_for_condition_1")
plt.close()

t2 = np.reshape(t[2,:], vol_shape)
t2[~in_brain_mask]=np.nan
for i in range(34):
 plt.subplot(5,7,i+1)
 plt.imshow(t2[:,:,i])
 plt.title("Slice"+str(i+1), fontsize=5)
 plt.tight_layout()

plt.suptitle("Subject 1 Run 1 T Statistics in Condition 2 for different Slices\n")
plt.colorbar()
plt.savefig(location_of_plot+"t_statistics_for_condition_2")
plt.close()

t3 = np.reshape(t[3,:],vol_shape)
t3[~in_brain_mask]=np.nan
for i in range(34):
 plt.subplot(5,7,i+1)
 plt.imshow(t3[:,:,i])
 plt.title("Slice"+str(i+1), fontsize=5)
 plt.tight_layout()

plt.suptitle("Subject 1 Run 1 T Statistics in Condition 3 for different Slices\n")
plt.colorbar()
plt.savefig(location_of_plot+"t_statistics_for_condition_3")
plt.close()

t4 = np.reshape(t[4,:],vol_shape)
t4[~in_brain_mask]=np.nan
for i in range(34):
 plt.subplot(5,7,i+1)
 plt.imshow(t4[:,:,i])
 plt.title("Slice"+str(i+1), fontsize=5)
 plt.tight_layout()

plt.suptitle("Subject 1 Run 1 T Statistics in Condition 4 for different Slices\n")
plt.colorbar()
plt.savefig(location_of_plot+"t_statistics_for_condition_4")
plt.close()

