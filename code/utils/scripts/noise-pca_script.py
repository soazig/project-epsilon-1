"""
This script is used to design the design matrix for our linear regression.
We explore the influence of linear and quadratic drifts on the model 
performance.

"""
import sys, os, pdb
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from matplotlib import colors

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import scipy
import pprint as pp

#Specicy the path for functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from diagnostics import *
from glm import *
from plot_mosaic import * 
#from mask_functions import *

# Locate the paths
project_path = '../../../'
data_path = project_path+'data/ds005/' 
path_dict = {'data_filtered':{ 
			      'type' : 'filtered',
			      'bold_img_name' : 'filtered_func_data_mni.nii.gz',
			      'run_path' : 'model/model001/'
			     },
             'data_original':{
		       	      'type' : '',
                              'bold_img_name' : 'bold.nii.gz',
                              'run_path' : 'BOLD/'
			     }}
			
# TODO: uncomment for final version
#subject_list = [str(i) for i in range(1,17)]
#run_list = [str(i) for i in range(1,4)]
run_list = [str(i) for i in range(1,3)]
subject_list = ['1']

d = path_dict['data_original'] #OR path_dict['data_filtered']

images_paths = [('ds005' + d['type'] +'_sub' + s.zfill(3) + '_t1r' + r, \
                 data_path + 'sub%s/'%(s.zfill(3)) + d['run_path'] \
                 + 'task001_run%s/%s' %(r.zfill(3), d['bold_img_name'])) \
                 for r in run_list \
                 for s in subject_list]
sm = ''
#sm='not_smooth/'
project_path = project_path + sm
# Create the needed directories if they do not exist
dirs = [project_path+'fig/',\
        project_path+'fig/BOLD',\
        project_path+'fig/linear_model',\
        project_path+'fig/pca',\
	project_path+'fig/pca/projections/',\
	project_path+'fig/linear_model/projections',\
	project_path+'fig/linear_model/mosaic',\
	project_path+'fig/linear_model/mosaic/middle_slice',\
        project_path+'txt_output/',\
	project_path+'txt_output/pca/',\
	project_path+'txt_output/design_matrix/',\
	project_path+'txt_output/betas/']

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
# set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

# Create the mask				                
thres = 375 #From analysis of the histograms   

for image_path in images_paths:
    name = image_path[0]
    img = nib.load(image_path[1])
    data = img.get_data()
    vol_shape = data.shape[:-1]
    mean_data = np.mean(data, axis=-1)
    in_brain_mask = mean_data > thres
    # Select the voxels in the brain
    plt.imshow(plot_mosaic(mean_data), cmap='gray', alpha=1)
    plt.colorbar()
    plt.contour(plot_mosaic(in_brain_mask),colors='blue')
    plt.title('In brain voxel mean values' + '\n' +  (str(name)))
    plt.savefig(project_path+'fig/BOLD/%s_mean_voxels_countour.png'\
                %(str(name)))
    #plt.show()
    #plt.clf()
    
    n_trs = data.shape[-1]
    # Smoothing with Gaussian filter
    # Smooth by 2 voxel SD in all three spatial dimensions
    smooth_data = gaussian_filter(data, [2, 2, 2, 0])
    # Convolution with 1 to 4 conditions
    convolved = []
    for i in range(1,5):
        convolved = np.loadtxt('../../../txt_output/conv_normal/%s_conv001_canonical.txt'%(str(name)))
    #Create design matrix X
    P = 7 #number of regressors of X 
    X = np.ones((n_trs, P))
    for i in range(1,5):
        X[:,i] = convolved[i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X[:,5] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X[:,6] = quadratic_drift
    # Save the design matrix
    np.savetxt(project_path+\
     'txt_output/design_matrix/%s_design_matrix_with_drift.txt'\
     %(str(name)), X)

#    X = np.loadtxt('ds005_sub001_t1r1_design_matrix.txt')
#    plt.imshow(X, aspect=0.05)
#    plt.colorbar()
#    plt.title('Design matrix subject 1 r1t1')
#    plt.show()
#    plt.savefig('../fig/linear_model/design_matrix/ds005_sub001_t1r1_conv1.png')
#    plt.close()
    # Model
    reg_str = ['Intercept','Task', 'Gain', 'Loss', 'Distance', 'Linear Drift',\
                'Quadratic drift', 'PC#1', 'PC#2', 'PC#3', 'PC#4']
    in_brain_tcs = data[in_brain_mask, :]
    #TODO: uncomment if don't smooth
    in_brain_tcs = smooth_data[in_brain_mask, :]
    Y = in_brain_tcs.T
    betas = npl.pinv(X).dot(Y)
    #Save the betas 
    np.savetxt(project_path+\
      'txt_output/betas/%s_betas_with_drift.txt'%(str(name)), betas)
##    betas = np.loadtxt('%s_betas.txt'%(str(name)))
    betas_vols = np.zeros(vol_shape + (P,))
    betas_vols[in_brain_mask] = betas.T
#    # set regions outside mask as missing with np.nan
    mean_data[~in_brain_mask] = np.nan
    betas_vols[~in_brain_mask] = np.nan
    nice_cmap_values = np.loadtxt('actc.txt')
    nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')
    for k in range(1,P):
       	#plt.imshow(plot_mosaic(betas_vols[...,k]), cmap='gray', alpha=1)
	plt.imshow(plot_mosaic(mean_data), cmap='gray', alpha=1)
       	plt.imshow(plot_mosaic(betas_vols[...,k]), cmap=nice_cmap, alpha=1)
	plt.colorbar()
        plt.title('Beta (with drift) values for brain voxel related to ' \
	    + str(reg_str[k]) + '\n' + (str(name)))
	plt.savefig(project_path+'fig/linear_model/mosaic/%s_withdrift_gray_%s'\
	            %(str(name), str(reg_str[k]))+'.png')
        plt.close()
	#plt.show()
	#plt.clf()
	#Show the middle slice only
        plt.imshow(betas_vols[:, :, 18, k], cmap='gray', alpha=0.5)
        plt.colorbar()
        plt.title('In brain voxel - Slice 18 \n' \
	          'Projection on %s - %s'\
	          %(str(reg_str[k]), str(name)))
	plt.savefig(\
	project_path+'fig/linear_model/mosaic/middle_slice/%s_withdrift_gray_middleslice_%s'\
	%(str(name), str(k))+'.png')
	#plt.show()
        #plt.clf()
	plt.close()
    
    # PCA
    Y_demeaned = Y - np.mean(Y, axis=1).reshape([-1, 1])
    unscaled_cov = Y_demeaned.dot(Y_demeaned.T)
    U, S, V = npl.svd(unscaled_cov)
    projections = U.T.dot(Y_demeaned)
    projection_vols = np.zeros(data.shape)
    projection_vols[in_brain_mask, :] = projections.T
    PC = 3
    for i in range(1,PC):
        plt.plot(U[:, i])
	plt.title('U' + str(i) + ' vector from SVD \n' + str(name))
        plt.imshow(projection_vols[:, :, 18, i])   
        plt.colorbar()
        plt.title('PCA - Projection on PC#' + str(i) + ' vector from SVD \n ' + str(name))
	plt.savefig(project_path+'fig/pca/projections/PC%s_' %i + str(name) + '.png')
        #plt.show()
	#plt.clf()
	plt.close()
    #TODO: remove the mean
    s = []
    for i in S:
        s.append(i/np.sum(S))
    np.savetxt(project_path+\
       'txt_output/pca/%s_variance_explained' %str(name) +\
       '.txt', np.array(s[:50]))
    ind = np.arange(len(s[1:50]))
    plt.bar(ind, s[1:50], width=0.25)
    plt.xlabel('S indices')
    plt.ylabel('Explained variance in percent')
    plt.title('Variance explained graph \n' + str(name))
    plt.savefig(project_path+\
        'fig/pca/%s_variance_explained' %str(name) + '.png')
    #plt.show()
    plt.close()

    P_pca = P + PC 
    X_pca = np.ones((n_trs, P_pca))
    for i in range(1,5):
        X_pca[:,i] = convolved[i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X_pca[:,5] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X_pca[:,6] = quadratic_drift
    for i in range(3):
        X_pca[:,7+i] = U[:, i]

    np.savetxt(project_path+'txt_output/pca/%s_design_matrix_pca.txt'%(str(name)), X_pca)
#    X_pca = np.loadtxt('ds005_sub001_t1r1_design_matrix_pca.txt')
    #plt.imshow(X_pca, aspect=0.25)
    B_pca = npl.pinv(X_pca).dot(Y)
    np.savetxt(project_path+'txt_output/pca/%s_betas_pca.txt'%(str(name)), B_pca)
    #B_pca = np.loadtxt('ds005_sub001_t1r1_betas_pca.txt')
    b_pca_vols = np.zeros(vol_shape + (P_pca,))
    b_pca_vols[in_brain_mask, :] = B_pca.T
#    # set regions outside mask as missing with np.nan
    mean_data[~in_brain_mask] = np.nan
    b_pca_vols[~in_brain_mask] = np.nan
    for k in range(1,P_pca):
       	plt.imshow(plot_mosaic(b_pca_vols[...,k]), cmap='gray', alpha=0.5)
	#plt.imshow(plot_mosaic(mean_data), cmap='gray', alpha=1)
       	#plt.imshow(plot_mosaic(b_pca_vols[...,k]), cmap=nice_cmap, alpha=1)
	plt.colorbar()
        plt.title('Beta (with PCA) values for brain voxel related to ' \
	    + str(reg_str[k]) + '\n' + str(name))
	plt.savefig(project_path+'fig/linear_model/mosaic/%s_withPCA_%s'\
	            %(str(name), str(reg_str[k]))+'.png')
        #plt.show()
	#plt.clf()
	plt.close()
	#Show the middle slice only
        plt.imshow(b_pca_vols[:, :, 18, k], cmap='gray', alpha=0.5)
        plt.colorbar()
        plt.title('In brain voxel model - Slice 18 \n' \
	          'Projection on X%s \n %s'\
	          %(str(reg_str[k]),str(name)))
	plt.savefig(\
	project_path+\
	'fig/linear_model/mosaic/middle_slice/%s_withPCA_middle_slice_%s'\
	%(str(name), str(k))+'.png')
	#plt.show()
        #plt.clf()
        plt.close()
    
#    # Residuals
#    df = X.shape[0] - npl.matrix_rank(X)
#    MRSS = np.sum(residuals ** 2 , axis=0) / df
#    residuals_pca = Y - X_pca.dot(B_pca)
#    df_pca = X_pca.shape[0] - npl.matrix_rank(X_pca)
#    MRSS_pca = np.sum(residuals_pca ** 2 , axis=0) / df_pca
#    np.savetxt(project_path+\
#       'txt_output/linear_model/%s_mean_mrss_vals.txt'%(str(name)),\
#       (np.mean(MRSS), np.mean(MRSS_pca)))
#pdb.set_trace()
print("done")
