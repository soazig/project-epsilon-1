"""
Purpose:
-----------------------------------------------------------------------------------
We try to capture the significance of gain and loss amount condition for each subjects.
We fit the logistic regression line based on their responses on the experiment. The slope 
of the fitted line illustrates the subject's sensitivity on either gain or loss amount. 

This script outputs plots for each subject and combine them into one image of subplots. 
-----------------------------------------------------------------------------------
"""

from __future__ import absolute_import, division, print_function
import sys, os
#TODO : later change this
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from logistic_reg import *
from organize_behavior_data import *



# Create the necessary directories if they do not exist
dirs = ['../../fig','../../fig/log_reg_behav']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
#TODO: the current location for this file project-epsilon/code/scripts
project_path = '../../../'
# TODO: change it to relevant path
data_path = project_path+'data/ds005/'
subject_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']


fig = plt.figure()
for i,subject in enumerate(subject_list):
	behav_df = load_in_dataframe(subject)
	add_lambda = add_gainlossratio(behav_df)
	columns_changed = organize_columns(add_lambda)
	logit_pars = log_regression(columns_changed)

	### START TO PLOT ###
	# calculate intercept and slope
	intercept = -logit_pars['Intercept'] / logit_pars['gain']
	slope = -logit_pars['loss'] / logit_pars['gain']
		#fig = plt.figure(figsize = (10, 8))   
	ax = fig.add_subplot(4, 4, i+1) 
	ax.set_title("Subject_%s_run001"%(str(i+1)), fontsize =10)
	ax.set_axis_bgcolor('white')

	# plot gain and loss for respcat = 1(decides to gamble)
	l1, = ax.plot(behav_df[behav_df['respcat'] == 1].values[:,2], behav_df[behav_df['respcat'] == 1].values[:,1], '.', label = "Gamble", mfc = 'None', mec='red')

	# plot gain and loss for respcat = 0(decides to not gamble)
	l2, = ax.plot(behav_df[behav_df['respcat'] == 0].values[:,2], behav_df[behav_df['respcat'] == 0].values[:,1], '.', label = "Not gamble", mfc = 'None', mec='blue')

	# draw regression line
	ax.plot(behav_df['loss'], intercept + slope * behav_df['loss'],'-', color = 'green') 

	ax.set_xlabel('Loss ($)', fontsize =10)
	ax.set_ylabel('Gain ($)', fontsize =10)
	ax.set_xlim([2,23])
	ax.set_ylim([8,41])
	ax.tick_params(axis='x', labelsize=10)
	ax.tick_params(axis='y', labelsize=10)
	
fig.legend((l1,l2), ('Gamble','Not Gamble'), loc = 'lower right', labelspacing = 0.5, fontsize = 10)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
fig.suptitle("Fitted Logistic Regression Line (1(gamble) 0(not gamble) with gain and loss values\n", fontsize=12)
fig.savefig(dirs[1]+'/log_regression_behav_subplots.png',facecolor='white', edgecolor='white')


