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

from __future__ import print_function, division
import sys, os
import pdb
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
sys.path.append(os.path.join(os.path.dirname('__file__'), "./"))
from glm_func import *
from matplotlib import colors
from smoothing import *
from plot_mosaic import * 
from scipy.stats import t as t_dist


nice_cmap_values = np.loadtxt('actc.txt')
nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')
dirs = ['../../../txt_output/multi_beta','../../../fig/multi_beta']
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
project_path='../../../'

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)


#Template to plot brain images
template = nib.load(project_path+\
           'data/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii')
template_data_int = template.get_data()
template_data = template_data_int.astype(float)

task = dict()
gain = dict()
loss = dict()
print("\n================================================================================")
print("Starting multi comparison analysis for the selected subjects")

subject_list = [str(i) for i in range(1,17))
#subject_list = ['1','5']
#load all of them
#for x in range(1,17):
for sub in subject_list:
	task[sub] = np.loadtxt(dirs[0]+'/ds005_sub'+str(sub).zfill(3)+'_t1r1_beta_task.txt')
#for x in range(1,17):
	gain[sub] = np.loadtxt(dirs[0]+'/ds005_sub'+str(sub).zfill(3)+'_t1r1_beta_gain.txt')
#for x in range(1,17):
	loss[sub] = np.loadtxt(dirs[0]+'/ds005_sub'+str(sub).zfill(3)+'_t1r1_beta_loss.txt')
# for x in range(1,17):
# 	dist[x] = np.loadtxt(dirs[0]+'/ds005_sub'+str(x).zfill(3)+'_t1r1_beta_dist.txt')

##################################### MEAN plot #########################################
print("\n================================================================================")
print("Starting analysis for the mean voxels values accross subjects")
#calculate mean and plot (let's try for task)
#task_sum = task[1]
#for x in range(2,17):
#	task_sum +=task[x]

#task_mean = task_sum/16
#task_mean = task_sum/len(subject_list)
#task_mean_reshape = task_mean.reshape(91,109,91)

#gain_sum = gain[1] 
#for x in range(2,17):
#	gain_sum +=gain[x]

#gain_mean = gain_sum/16
#gain_mean_reshape = gain_mean.reshape(91,109,91)


#loss_sum = gain[1]
#for x in range(2,17):
#	loss_sum +=loss[x]

#loss_mean = loss_sum/16
#loss_mean_reshape = loss_mean.reshape(91,109,91)

for mydict in [task, gain, loss]:
    mydict['mean'] = sum([task[sub] for sub in subject_list])/(len(subject_list)) 
    mydict['mean_reshape'] = mydict['mean'].reshape(91,109,91)
    mydict['mean_mask'] = (mydict['mean_reshape'] - 0.0) < 0.01
    mydict['mean_reshape_plot'] = mydict['mean_reshape']
    mydict['mean_reshape_plot'][~mydict['mean_mask']] = np.nan
print("Creating plots for the voxel mean accross subjects analysis")
task_mean = task['mean']
gain_mean = gain['mean']
loss_mean = loss['mean']
in_brain_task = task['mean_mask']
in_brain_gain = gain['mean_mask']
in_brain_loss = loss['mean_mask']

plt.title('In brain activated voxels - \nmean across ' + str(len(subject_list)) +' subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task['mean_reshape_plot'], transpose=False), \
cmap='seismic', alpha=1, vmin=task['mean_reshape_plot'].min(), vmax= task['mean_reshape_plot'].max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_task.png')
#plt.show()
plt.clf()
print("  Plot for the TASK condition saved in " + dirs[1]+'/mean_task.png')

plt.title('In brain activated voxels - \nmean across ' + str(len(subject_list)) + ' subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain['mean_reshape_plot'], transpose=False), \
cmap='seismic', alpha=1, vmin=gain['mean_reshape_plot'].min(), vmax= gain['mean_reshape_plot'].max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_gain.png')
#plt.show()
plt.clf()
print("  Plot for the GAIN condition saved in " + dirs[1] + '/mean_gain.png')

plt.title('In brain activated voxels - \nmean across ' + str(len(subject_list)) + ' subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss['mean_reshape_plot'], transpose=False), \
cmap='seismic', alpha=1, vmin=loss['mean_reshape_plot'].min(), vmax= loss['mean_reshape_plot'].max())
plt.colorbar()
plt.savefig(dirs[1]+'/mean_loss.png')
#plt.show()
plt.clf()
print("  Plot for the LOSS condition saved in " + dirs[1] + '/mean_loss.png')

#################################### SD plot #########################################
print("\n================================================================================")
print("Starting analysis and plot for the mean variance accross subjects")

#calculate variance and plot
stdlst = []
for x in subject_list:
	stdlst.append(task[x])

stdarray = np.array(stdlst)
task_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
task_std_reshape = task_std.reshape(91,109,91)
task_std_reshape_plot = task_std_reshape
task_std_reshape_plot[~in_brain_gain] = np.nan

stdlst = []
#for x in range(1,17):
for x in subject_list:
	stdlst.append(gain[x])

stdarray = np.array(stdlst)
gain_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
gain_std_reshape = gain_std.reshape(91,109,91)
gain_std_reshape_plot = gain_std_reshape
gain_std_reshape_plot[~in_brain_gain] = np.nan

stdlst = []
for x in subject_list:
	stdlst.append(loss[x])

stdarray = np.array(stdlst)
loss_std = stdarray.std(axis=0)
#task_std.shape -> (902629,0)
loss_std_reshape = loss_std.reshape(91,109,91)
loss_std_reshape_plot = loss_std_reshape
loss_std_reshape_plot[~in_brain_loss] = np.nan
print("Creating plots for the voxel mean variance accross subjects analysis")

plt.title('In brain activated voxels - \nStandard Deviation across ' + str(len(subject_list)) + ' subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_std_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=task_std_reshape_plot.min(), vmax= task_std_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_task.png')
#plt.show()
plt.clf()
print("  Plot for the TASK condition saved in " + dirs[1] + '/std_task.png')

plt.title('In brain activated voxels - \nStandard Deviation across ' + str(len(subject_list)) + ' subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_std_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=gain_std_reshape_plot.min(), vmax= gain_std_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_gain.png')
#plt.show()
plt.clf()
print("  Plot for the GAIN condition saved in " + dirs[1] + '/std_task.png')

plt.title('In brain activated voxels - \nStandard Deviation across ' + str(len(subject_list)) + ' subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss_std_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=loss_std_reshape_plot.min(), vmax= loss_std_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/std_loss.png')
#plt.show()
plt.clf()
print("  Plot for the LOSS condition saved in " + dirs[1] + '/std_loss.png')

##################################### stat plot #########################################
print("\n================================================================================")
print("Starting analysis and plot for the t-test accross subjects")


#calculate t-stat and plot
task_tstat = task_mean/(task_std/np.sqrt(15))
task_tstat = np.nan_to_num(task_tstat)
task_tstat_reshape = task_tstat.reshape(91,109,91)
task_tstat_reshape_plot = task_tstat_reshape
task_tstat_reshape_plot[~in_brain_task] = np.nan

gain_tstat = gain_mean/(gain_std/np.sqrt(15))
gain_tstat = np.nan_to_num(gain_tstat)
gain_tstat_reshape = gain_tstat.reshape(91,109,91)
gain_tstat_reshape_plot = gain_tstat_reshape
gain_tstat_reshape_plot[~in_brain_gain] = np.nan

loss_tstat = loss_mean/(loss_std/np.sqrt(15))
loss_tstat = np.nan_to_num(loss_tstat)
loss_tstat_reshape = loss_tstat.reshape(91,109,91)
loss_tstat_reshape_plot = loss_tstat_reshape
loss_tstat_reshape_plot[~in_brain_loss] = np.nan

plt.title('In brain activated voxels - \nT-statistics across ' + str(len(subject_list)) + ' subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_tstat_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=task_tstat_reshape_plot.min(), vmax=task_tstat_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_task.png')
#plt.show()
plt.clf()
print("  Plot for the TASK condition saved in " + dirs[1] + '/tstat_task.png')

plt.title('In brain activated voxels - \nT-statistics across ' + str(len(subject_list)) + ' subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_tstat_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=gain_tstat_reshape_plot.min(), vmax=gain_tstat_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_gain.png')
#plt.show()
plt.clf()
print("  Plot for the GAIN condition saved in " + dirs[1] + '/tstat_gain.png')

plt.title('In brain activated voxels - \nT-statistics across ' + str(len(subject_list)) + ' subjects on LOSS condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(loss_tstat_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=loss_tstat_reshape_plot.min(), vmax=loss_tstat_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/tstat_loss.png')
#plt.show()
plt.clf()
print("  Plot for the LOSS condition saved in " + dirs[1] + '/tstat_loss.png')

##################################### P-value plot #########################################
print("\n================================================================================")
print("Starting analysis and plot for the p-values accross subjects")

#calculate p-value and plot
task_pval = t_dist.cdf(abs(task_tstat), 15)
task_pval_reshape_old = task_pval.reshape(91,109,91)
task_pval_reshape = np.ones((91,109,91))-task_pval_reshape_old
task_pval_reshape_plot = task_pval_reshape
mask = task_pval_reshape_plot < 0.10 
task_pval_reshape_plot[~mask] = np.nan

gain_pval = t_dist.cdf(abs(gain_tstat), 15)
gain_pval_reshape_old = gain_pval.reshape(91,109,91)
gain_pval_reshape = np.ones((91,109,91))-gain_pval_reshape_old
gain_pval_reshape_plot = gain_pval_reshape
mask = gain_pval_reshape_plot < 0.10 
gain_pval_reshape_plot[~mask] = np.nan

loss_pval = t_dist.cdf(abs(loss_tstat), 15)
loss_pval_reshape_old = loss_pval.reshape(91,109,91)
loss_pval_reshape = np.ones((91,109,91))-loss_pval_reshape_old
loss_pval_reshape_plot = loss_pval_reshape
#loss_pval_reshape_plot[~in_brain_task] = np.nan

p_value_thres = 0.05/(91*109*91)

plt.title('In brain activated voxels - \nP-value across ' + str(len(subject_list)) + ' subjects on TASK condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(task_pval_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=task_pval_reshape_plot.min(), vmax=task_pval_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/pval_task.png')
plt.show()
plt.clf()
print("  Plot for the TASK condition saved in " + dirs[1] + '/pval_task.png')

plt.title('In brain activated voxels - \nP-value across ' + str(len(subject_list)) +' subjects on GAIN condition', fontsize=12)
plt.imshow(plot_mosaic(template_data, transpose=False), cmap='gray', alpha=1)
plt.imshow(plot_mosaic(gain_pval_reshape_plot, transpose=False), cmap='seismic', alpha=1, vmin=gain_pval_reshape_plot.min(), vmax=gain_pval_reshape_plot.max())
plt.colorbar()
plt.savefig(dirs[1]+'/pval_gain.png')
plt.show()
plt.clf()
print("  Plot for the GAIN condition saved in " + dirs[1] + '/pval_gain.png')
print("Multi comparison analysis done")

