"""
This script is used to design the design matrix for our linear regression.
We explore the influence of linear and quadratic drifts on the model 
performance.


This script is for the raw data.
Run with:
    python noise-pca_script.py 
"""
from __future__ import print_function, division
import sys, os, pdb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from matplotlib import colors
from os.path import splitext
from scipy.stats import t as t_dist

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import scipy
import pprint as pp
import json

#Specicy the path for functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from smoothing import *
from diagnostics import *
from glm import *
from plot_mosaic import * 
from mask_filtered_data import *

# Locate the paths
project_path = '../../../'
data_path = project_path+'data/ds005/' 
path_dict = {'data_filtered':{ 
			      'type' : 'filtered',
			      'feat' : '.feat/',
			      'bold_img_name' : 'filtered_func_data_mni.nii.gz',
			      'run_path' : 'model/model001/'
			     },
             'data_original':{
		       	      'type' : '',
			      'feat': '',
                              'bold_img_name' : 'bold.nii.gz',
                              'run_path' : 'BOLD/'
			     }}
			
# TODO: uncomment for final version
#subject_list = [str(i) for i in range(1,17)]
#run_list = [str(i) for i in range(1,4)]

# Run only for subject 1 and 5 - run 1
run_list = [str(i) for i in range(1,2)]
subject_list = ['1','5']

d_path = path_dict['data_original'] #OR original or filtered 

images_paths = [('ds005' + '_sub' + s.zfill(3) + '_t1r' + r, \
                 data_path + 'sub%s/'%(s.zfill(3)) + d_path['run_path'] \
                 + 'task001_run%s%s/%s' %(r.zfill(3),d_path['feat'],\
		 d_path['bold_img_name'])) \
                 for r in run_list \
                 for s in subject_list]
# set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

# Mask	
# To be used with the normal data
thres = 375 #From analysis of the histograms   
# To be used with the filtered data
mask_path = project_path+'data/mni_icbm152_t1_tal_nlin_asym_09c_mask_2mm.nii'

sm = ''
#sm='not_smooth/'
project_path = project_path + sm

# Create the needed directories if they do not exist
dirs = [project_path+'fig/',\
        project_path+'fig/BOLD',\
        project_path+'fig/drifts',\
        project_path+'fig/pca',\
	project_path+'fig/mosaic',\
        project_path+'txt_output/',\
	project_path+'txt_output/MRSS/',\
	project_path+'txt_output/pca/',\
	project_path+'txt_output/drifts/']

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

MRSS_dict = {}
MRSS_dict['ds005' + d_path['type']] = {}

for image_path in images_paths:
    name = image_path[0]
    MRSS_dict['ds005' + d_path['type']][name] = {}
    MRSS_dict['ds005' + d_path['type']][name]['drifts'] = {}
    MRSS_dict['ds005' + d_path['type']][name]['pca'] = {}
    if d_path['type']=='filtered':
        in_brain_img = nib.load('../../../'+
	    'data/ds005/sub001/model/model001/task001_run001.feat/'\
	    + 'masked_filtered_func_data_mni.nii.gz')
	# Image shape (91, 109, 91, 240)
	#in_brain_img = make_mask_filtered_data(image_path[1],mask_path)
	data = in_brain_img.get_data()
        mean_data = np.mean(data, axis=-1)
        in_brain_mask = (mean_data - 0.0) < 0.01	
	Transpose = False
    else:
        img = nib.load(image_path[1])
        data = img.get_data()
        mean_data = np.mean(data, axis=-1)
        in_brain_mask = mean_data > thres
	Transpose = True
    # Smoothing with Gaussian filter
    smooth_data = smoothing(data,1,range(data.shape[-1])) 

    # Selecting the voxels in the brain
    in_brain_tcs = smooth_data[in_brain_mask, :] 
    #in_brain_tcs = data[in_brain_mask, :] 
    vol_shape = data.shape[:-1]
    # Plotting the voxels in the brain
    plt.imshow(plot_mosaic(mean_data, transpose=Transpose), cmap='gray', alpha=1)
    plt.colorbar()
    plt.contour(plot_mosaic(in_brain_mask, transpose=Transpose),colors='blue')
    plt.title('In brain voxel mean values' + '\n' +  (d_path['type'] + str(name)))
    plt.savefig(project_path+'fig/BOLD/%s_mean_voxels_countour.png'\
                %(d_path['type'] + str(name)))
    #plt.show()
    plt.clf()
    
    # Convolution with 1 to 4 conditions
    convolved = np.zeros((240,5))
    for i in range(1,5):
        #convolved = np.loadtxt(\
	#   '../../../txt_output/conv_normal/%s_conv_00%s_canonical.txt'\
	#   %(str(name),str(i)))
        convolved[:,i] = np.loadtxt(\
	    '../../../txt_output/conv_high_res/%s_conv_00%s_high_res.txt'\
	    %(str(name),str(i)))
    reg_str = ['Intercept','Task', 'Gain', 'Loss', 'Distance', 'Linear Drift',\
                'Quadratic drift', 'PC1', 'PC2', 'PC3', 'PC4']
    

    # Create design matrix X - Including drifts
    P = 7 #number of regressors of X including the ones for intercept 
    n_trs = data.shape[-1]
    X = np.ones((n_trs, P))
    for i in range(1,5):
        X[:,i] = convolved[:,i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X[:,5] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X[:,6] = quadratic_drift
    # Save the design matrix
    np.savetxt(project_path+\
     'txt_output/drifts/%s_design_matrix_with_drift.txt'\
     %(d_path['type'] + str(name)), X)

    # Linear Model - Including drifts
    Y = in_brain_tcs.T

    betas = npl.pinv(X).dot(Y)
    # Save the betas for the linear model including drifts 
    np.savetxt(project_path+\
      'txt_output/drifts/%s_betas_with_drift.txt'%(d_path['type'] + str(name)), betas)
    betas_vols = np.zeros(vol_shape + (P,))
    betas_vols[in_brain_mask] = betas.T
    
    # Plot
    # Set regions outside mask as missing with np.nan
    mean_data[~in_brain_mask] = np.nan
    betas_vols[~in_brain_mask] = np.nan
    nice_cmap_values = np.loadtxt('actc.txt')
    nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')
    # Plot each slice on the 3rd dimension of the image in a mosaic
    for k in range(1,P):
	plt.imshow(plot_mosaic(mean_data, transpose=Transpose), cmap='gray', alpha=1)
       	#plt.imshow(plot_mosaic(betas_vols[...,k], transpose=Transpose), cmap='gray', alpha=1)
       	plt.imshow(plot_mosaic(betas_vols[...,k], transpose=Transpose), cmap=nice_cmap, alpha=1)
	plt.colorbar()
        plt.title('Beta (with drift) values for brain voxel related to ' \
	    + str(reg_str[k]) + '\n' + d_path['type'] + str(name))
	plt.savefig(project_path+'fig/mosaic/%s_withdrift_%s'\
	            %(d_path['type'] + str(name), str(reg_str[k]))+'.png')
        plt.close()
	#plt.show()
	plt.clf()
	#Show the middle slice only
        plt.imshow(betas_vols[:, :, 18, k], cmap='gray', alpha=0.5)
        plt.colorbar()
        plt.title('In brain voxel - Slice 18 Projection on %s\n%s'\
	          %(str(reg_str[k]), d_path['type'] + str(name)))
	plt.savefig(\
	project_path+'fig/drifts/%s_withdrift_middleslice_%s'\
	%(d_path['type'] + str(name), str(k))+'.png')
	#plt.show()
        plt.clf()
	plt.close()

    # PCA Analysis
    Y_demeaned = Y - np.mean(Y, axis=1).reshape([-1, 1])
    unscaled_cov = Y_demeaned.dot(Y_demeaned.T)
    U, S, V = npl.svd(unscaled_cov)
    projections = U.T.dot(Y_demeaned)
    projection_vols = np.zeros(data.shape)
    projection_vols[in_brain_mask, :] = projections.T
    # Plot the projection of the data on the 5 first principal component
    # from SVD
    for i in range(1,5):
        plt.plot(U[:, i])
	plt.title('U' + str(i) + ' vector from SVD \n' + str(name))
        plt.imshow(projection_vols[:, :, 18, i])   
        plt.colorbar()
        plt.title('PCA - 18th slice projection on PC' + str(i) + ' from SVD \n ' +\
	          d_path['type'] + str(name))
	plt.savefig(project_path+'fig/pca/%s_PC%s.png' \
	%((d_path['type'] + str(name),str(i))))
        #plt.show()
	plt.clf()
	plt.close()

    # Variance Explained analysis
    s = []
    #S is diag -> trace = sum of the elements of S
    for i in S:
        s.append(i/np.sum(S))
    np.savetxt(project_path+\
       'txt_output/pca/%s_variance_explained' % (d_path['type'] + str(name)) +\
       '.txt', np.array(s[:40]))
    ind = np.arange(len(s[1:40]))
    plt.bar(ind, s[1:40], width=0.5)
    plt.xlabel('Principal Components indices')
    plt.ylabel('Explained variance in percent')
    plt.title('Variance explained graph \n' + (d_path['type'] + str(name)))
    plt.savefig(project_path+\
        'fig/pca/%s_variance_explained.png' %(d_path['type'] + str(name)))
    #plt.show()
    plt.close()

    # Linear Model - including PCs from PCA analysis
    PC = 3 # Number of PCs to include in the design matrix
    P_pca = P + PC 
    X_pca = np.ones((n_trs, P_pca))
    for i in range(1,5):
        X_pca[:,i] = convolved[:,i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X_pca[:,5] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X_pca[:,6] = quadratic_drift
    for i in range(3):
        X_pca[:,7+i] = U[:, i]
    # Save the design matrix - with PCs
    np.savetxt(project_path+'txt_output/pca/%s_design_matrix_pca.txt'\
    %(d_path['type'] + str(name)), X_pca)
    #plt.imshow(X_pca, aspect=0.25)
    B_pca = npl.pinv(X_pca).dot(Y)
    np.savetxt(project_path+'txt_output/pca/%s_betas_pca.txt'\
    %(d_path['type'] + str(name)), B_pca)
    b_pca_vols = np.zeros(vol_shape + (P_pca,))
    b_pca_vols[in_brain_mask, :] = B_pca.T
    # Save betas as nii files


    # Plot - with PCs
    # Set regions outside mask as missing with np.nan
    mean_data[~in_brain_mask] = np.nan
    b_pca_vols[~in_brain_mask] = np.nan
    # Plot each slice on the 3rd dimension of the image in a mosaic
    for k in range(1,P_pca):
        fig = plt.figure(figsize = (8, 5))
        #plt.imshow(plot_mosaic(b_pca_vols[...,k], transpose=Transpose), cmap='gray', alpha=0.5)
	plt.imshow(plot_mosaic(mean_data, transpose=Transpose), cmap='gray', alpha=1)
       	plt.imshow(plot_mosaic(b_pca_vols[...,k], transpose=Transpose), cmap=nice_cmap, alpha=1)
	plt.colorbar()
        plt.title('Beta (with PCA) values for brain voxel related to ' \
	    + str(reg_str[k]) + '\n' + d_path['type'] + str(name))
	plt.savefig(project_path+'fig/mosaic/%s_withPCA_%s'\
	            %(d_path['type'] + str(name), str(reg_str[k]))+'.png')
        #plt.show()
	plt.close()
	#Show the middle slice only
        plt.imshow(b_pca_vols[:, :, 18, k], cmap='gray', alpha=0.5)
        plt.colorbar()
        plt.title('In brain voxel - Slice 18 Projection on %s \n %s'\
	          %(str(reg_str[k]),d_path['type'] + str(name)))
	plt.savefig(\
	project_path+\
	'fig/pca/%s_withPCA_middle_slice_%s'\
	%(d_path['type'] + str(name), str(k))+'.png')
	#plt.show()
        plt.clf()
        plt.close()

    # Residuals    
    residuals = Y - X.dot(betas)
    df = X.shape[0] - npl.matrix_rank(X)
    MRSS = np.sum(residuals ** 2 , axis=0) / df
    residuals_pca = Y - X_pca.dot(B_pca)
    df_pca = X_pca.shape[0] - npl.matrix_rank(X_pca)
    MRSS_pca = np.sum(residuals_pca ** 2 , axis=0) / df_pca
    MRSS_dict['ds005' + d_path['type']][name]['drifts']['mean_MRSS'] = []
    MRSS_dict['ds005' + d_path['type']][name]['pca']['mean_MRSS'] = []
    MRSS_dict['ds005' + d_path['type']][name]['drifts']['mean_MRSS'] = np.mean(MRSS)
    MRSS_dict['ds005' + d_path['type']][name]['pca']['mean_MRSS'] = np.mean(MRSS_pca)
with open(project_path+'txt_output/MRSS/ds005%s_MRSS.json'\
        %(d_path['type']), 'w') as file_out:
    json.dump(MRSS_dict, file_out)
    # Save the mean MRSS values to compare the performance 
    # of the design matrices
#    for design_matrix, beta, mrss, name in \
#        [(X, betas, MRSS, 'drifts'), (X_pca, B_pca, MRSS_pca, 'pca')]:
#	MRSS_dict['ds005' + d_path['type']][name]['p-values'] = []
#	MRSS_dict['ds005' + d_path['type']][name]['t-test'] = []
#
#
#        SE = np.zeros(beta.shape)
#        for i in range(design_matrix.shape[-1]):
#            c = np.zeros(design_matrix.shape[-1])
#            c[i]=1
#            c = np.atleast_2d(c).T
#            SE[i,:]= np.sqrt(\
#	        mrss* c.T.dot(npl.pinv(design_matrix.T.dot(design_matrix)).dot(c)))
#            zeros = np.where(SE==0)
#            SE[zeros] = 1
#            t = beta / SE
#            t[:,zeros] = 0
#            # Get p value for t value using CDF of t didstribution
#            ltp = t_dist.cdf(abs(t), df)
#            p = 1 - ltp # upper tail
#            t_brain = t[in_brain_mask]
#            p_brain = p[in_brain_mask]
#
#	    # Save 3D data in .nii files
#	    for k in range(1,4):
#                t_nib = nib.Nifti1Image(t_brain[..., k], affine)
#	        nib.save(t-test, project_path+'txt_output/%s/%s_t-test_%s.nii.gz'\
#		%(name, d_path['type'] + str(name),str(reg_str[k])))
#                p_nib = nib.Nifti1Image(p_brain[..., k], affine)
#	        nib.save(p-values,project_path+'txt_output/%s/%s_p-values_%s.nii.gz'\
#		%(name, d_path['type'] + str(name),str(reg_str[k])))
#        pdb.set_trace() 
#    pdb.set_trace() 


print("======================================")
print("\n Noise and PCA analysis done")
print("Design Matrix including drift terms \
      stored in project_epsilon/txt_output/drifts/ \n\n")
print("Design Matrix including PCs terms \
      stored in project_epsilon/txt_output/pca/\n\n")


