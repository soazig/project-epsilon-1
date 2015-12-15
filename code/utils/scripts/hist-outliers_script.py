"""
This script creates some graphs for QA analysis for each subjects
 - Outliers plots
 - histograms of the mean voxels

Run with:
    python hist-outliers_script.py
    from this directory

"""

from __future__ import division
from scipy import ndimage

import sys, os
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import scipy
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
import diagnostics

# set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'


# Locate the path
project_path = '../../../'
data_path = project_path+'data/ds005/'

# Create a directory for the figures if it does not exist
dirs = [project_path+'fig',\
        project_path+'fig/histograms',\
	project_path+'fig/outliers',\
        project_path+'fig/BOLD',\
	project_path+'txt_output/',\
	project_path+'txt_output/outliers']

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

dir_path = project_path+'data/ds005/'
# Uncomment to run all subjects
#subject_list = [str(i) for i in range(1,17)]
subject_list = ['1','5']

# Choose run here (1,2 and/or 3)
run_list = [str(i) for i in range(1,2)]
images_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r, \
                 dir_path + 'sub' + s.zfill(3) + '/BOLD/task001_run' \
		 + r.zfill(3) + '/bold.nii.gz') for r in run_list \
		 for s in subject_list]


# Loop through subjects 
for image_path in images_paths:
    name = image_path[0]
    img = nib.load(image_path[1])
    data = img.get_data()
    vol_shape = data.shape[-1]
    mean_vol = np.mean(data, axis=-1)
#    Take the mean value over time and plot an histogram
    mean_vol = np.mean(data, axis=-1)
    plt.hist(np.ravel(mean_vol), bins=100)
    plt.title('Histogram of mean values of voxels accross time\n' + str(name))
    plt.savefig(project_path+'fig/histograms/' + str(name) + '_hist.png')
    plt.xlabel('Mean value of voxels accross time')
    plt.close()
    # Outliers
    vol_stds = diagnostics.vol_std(data)
    assert len(vol_stds) == vol_shape

    #Plot data to see outliers
    np.savetxt(project_path+'txt_output/outliers/vol_std_' \
         + str(name) + '.txt', vol_stds)
    outliers, thresholds = diagnostics.iqr_outliers(vol_stds)
    np.savetxt(project_path+'txt_output/outliers/vol_std_outliers_' \
         + str(name) + '.txt', outliers)
    N = len(vol_stds)
    x = np.arange(N)
    plt.plot(vol_stds, label='voxels std ' + str(name))
    plt.plot(x[outliers], vol_stds[outliers], 'o', label='outliers')
    plt.plot([0, N], [thresholds[0], thresholds[0]], ':', label='IQR lo')
    plt.plot([0, N], [thresholds[1], thresholds[1]], ':', label='IQR hi')
    plt.title('voxels std ' + str(name))
    plt.xlabel('time')
    plt.legend(fontsize=11, \
    ncol=2, loc=9, borderaxespad=0.2)
    plt.savefig(project_path+'fig/outliers/%s_vol_std.png' %str(name))
    plt.close()

    #RMS difference values
    rms_dvals = diagnostics.vol_rms_diff(data)
    rms_outliers, rms_thresholds = diagnostics.iqr_outliers(rms_dvals)
    #Plot
    N = len(rms_dvals)
    x = np.arange(N)
    plt.plot(rms_dvals, label='vol RMS differences ' + str(name))
    plt.plot(x[rms_outliers], rms_dvals[rms_outliers], 'o', label='outliers')
    plt.plot([0, N], [rms_thresholds[0], rms_thresholds[0]], ':', label='IQR lo')
    plt.plot([0, N], [rms_thresholds[1], rms_thresholds[1]], ':', label='IQR hi')
    plt.title('voxels rms difference ' + str(name))
    plt.xlabel('time')
    plt.legend(fontsize=11, \
    ncol=2, loc=9, borderaxespad=0.2)
    plt.savefig(project_path+'fig/outliers/%s_vol_rms_outliers.png'%str(name))
    plt.close()
    #Label the outliers
    T = data.shape[-1]
    ext_outliers = diagnostics.extend_diff_outliers(rms_outliers)
    np.savetxt(project_path+'txt_output/outliers/%s_extended_vol_rms_outliers.png' \
      %str(name), ext_outliers)
    x = np.arange(T)
    rms_dvals_ext = np.concatenate((rms_dvals, (0,)), axis=0)
    plt.plot(rms_dvals_ext, label='vol RMS differences ' + str(name))
    plt.plot(x[ext_outliers], rms_dvals_ext[ext_outliers], 'o', label='outliers')
    plt.plot([0, N], [rms_thresholds[0], rms_thresholds[0]], ':', label='IQR lo')
    plt.plot([0, N], [rms_thresholds[1], rms_thresholds[1]], ':', label='IQR hi')
    plt.xlabel('time')
    plt.legend(fontsize=11, \
    ncol=2, loc=9, borderaxespad=0.2)
    plt.savefig(project_path+\
    'fig/outliers/%s_extended_vol_rms_outliers.png'%str(name))
    plt.close()

print("=============================")
print("\nHistograms and outliers plots generated")
print("See project-epsilon/fig/histograms")
print("See project-epsilon/fig/outliers\n\n")

