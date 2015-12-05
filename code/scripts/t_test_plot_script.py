import numpy as np
import sys
sys.path.append("../.utils")
from t_test import *
from find_activated_voxel_functions import *
from convolution_normal_script import X_matrix
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import colors
import matplotlib


location_of_data = "../../data/ds005/sub001/BOLD/task001_run001/"
location_of_plot = "../../plots/"

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

