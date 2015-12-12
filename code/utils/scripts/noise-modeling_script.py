"""
This script is used for the design matric for our linear regression.
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
from diagnostics import *

# Create the needed directories if they do not exist
#dirs = ['../../fig/','../../fig/linear_projection',\
#        '../../txt_output','../../txt_output/linear_model']
#for d in dirs:
#    if not os.path.exists(d):
#        os.makedirs(d)

# Locate the paths
project_path = '../../../'
bold_img_name = 'filtered_func_data_mni.nii.gz' #OR 'bold.nii.gz'
data_path = '../../data/ds005/' 
data_state = 'filtered' #OR ''
run_path = 'model/model001/' #OR '/BOLD/'
# TODO: uncomment for final version
#subject_list = [str(i) for i in range(1,17)]
#run_list = [str(i) for i in range(1,4)]
run_list = [str(i) for i in range(1,3)]
subject_list = ['1']
images_paths = [('ds005_' + data_state +'_sub' + s.zfill(3) + '_t1r' + r, \
                 data_path + 'sub%s/'%(s.zfill(3)) + run_path \
                 + 'task001_run%s/%s' %(r.zfill(3), bold_img_name)) \
                 for r in run_list \
                 for s in subject_list]

# set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
nice_cmap_values = np.loadtxt('scripts/actc.txt')
nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')


#Load the image				                
thres = 375 #From analysis of the histograms   
for image_path in images_paths:
    name = image_path[0]
    img = nib.load(image_path[1])
    data = img.get_data()
    vol_shape = data.shape[:-1]
    mean_data = np.mean(data, axis=-1)
    # Select the voxels in the brain
    in_brain_mask = mean_data > thres 
    n_trs = data.shape[-1]
    # Smoothing with Gaussian filter
    # Smooth by 2 voxel SD in all three spatial dimensions
    smooth_data = gaussian_filter(data, [2, 2, 2, 0])
    # Convolution with 1 to 4 conditions
    convolved = []
    for i in range(1,5):
        convolved.append(np.loadtxt(\
 	      'txt_files/ds005_sub001_t1r1_conv' + str(i) + '_high_res.txt'))
#        convolved = np.loadtxt('txt_files/ds005_sub001_t1r1_conv0.txt')
     
    #Create design matrix X
    P = 7 #number of regressors of X 
    X = np.ones((n_trs, P))
    for i in range(4):
        X[:,i] = convolved[i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X[:,4] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X[:,5] = quadratic_drift
    # Save the design matrix
    np.savetxt('../../txt_output/linear_model/%s_design_matrix.txt'%(str(name)), X)
#    X = np.loadtxt('ds005_sub001_t1r1_design_matrix.txt')
#    plt.imshow(X, aspect=0.05)
#    plt.colorbar()
#    plt.title('Design matrix subject 1 r1t1')
#    plt.show()
#    plt.savefig('../fig/linear_model/design_matrix/ds005_sub001_t1r1_conv1.png')
#    plt.close()

    # Model
    reg_str = ['Task', 'Gain', 'Loss', 'Distance', 'Linear Drift',\
                'Quadratic drift']
    in_brain_tcs = smooth_data[in_brain_mask, :]
    Y = in_brain_tcs.T
    betas = npl.pinv(X).dot(Y)
    np.savetxt('../../txt_output/linear_model/%s_betas.txt'%(str(name)), betas)
##    betas = np.loadtxt('%s_betas.txt'%(str(name)))

    betas_vols = np.zeros(vol_shape + (P,))
    betas_vols[in_brain_mask] = betas.T
#    # set regions outside mask as missing with np.nan
    mean_data[~in_brain_mask] = np.nan
    betas_vols[~in_brain_mask] = np.nan
    for k in range(5):
        grid_1D = np.zeros((6*6, vol_shape[0], vol_shape[1]))
        grid_1D_beta = grid_1D
        grid_2D = np.zeros((6*vol_shape[0], 6*vol_shape[1]))
        grid_2D_beta  = grid_2D
        for z in range(vol_shape[2]):
            grid_1D[z, :, :] = mean_data[:, ::-1, z].T
            grid_1D_beta[z, :, :] = betas_vols[:, :, z, k].T
            #plt.title('Slice #%s (z axis) - mean values of voxels \n' %(z+1)\
            #          + str(name))
            d = '../plots/fig/linear_model/model/%s/'%str(name)
            if not os.path.exists(d): os.makedirs(d)
            #plt.savefig(d + str(name) + '_mean_vol_mask_slice' \
       	#            + str(z+1) + '.png')
            z = 0
       	    for i in range(6):
	        for j in range(6):
		    if z < vol_shape[2]:
		        grid_2D[i*vol_shape[0]:(i+1)*vol_shape[0],\
		        j*vol_shape[1]:(j+1)*vol_shape[1]] = grid_1D[z]
		        grid_2D_beta[i*vol_shape[0]:(i+1)*vol_shape[0],\
		        j*vol_shape[1]:(j+1)*vol_shape[1]] = grid_1D_beta[z]
                        z += 1
	plt.imshow(grid_2D, cmap='gray', alpha=0.5)
       	plt.imshow(grid_2D_beta, cmap=nice_cmap, alpha=0.5)
	plt.colorbar()
        plt.title('In brain voxel activation related to ' \
	    + str(cond_str[k]) + '\n' + (str(name)))
	plt.savefig('../plots/fig/linear_model/model/%s_noise_modelX%s'\
	            %(str(name), str(k))+'.png')
        #plt.show()
                         
#`    for i in range(P):
#`        plt.imshow(betas_vols[:, :, 18, i], cmap='gray', alpha=0.5)
#`        plt.colorbar()
#`        plt.title('In brain voxel model - projection on X%s \n %s'\
#`	          %(i,str(name)))
#`	plt.savefig('../plots/fig/linear_model/pca/%s_noise_modelX%s'\
#`	            %(str(name), str(i))+'.png')
#`	plt.show()
#`        plt.close()
#`        pdb.set_trace()
#    # PCA
    Y_demeaned = Y - np.mean(Y, axis=1).reshape([-1, 1])
    pdb.set_trace()
    unscaled_cov = Y_demeaned.dot(Y_demeaned.T)
    U, S, V = npl.svd(unscaled_cov)
    projections = U.T.dot(Y_demeaned)
    projection_vols = np.zeros(data.shape)
    projection_vols[in_brain_mask, :] = projections.T
    for i in range(4):
        plt.plot(U[:, i])
	plt.title('U' + str(i) + ' vector from SVD \n' + str(name))
	plt.savefig('../plots/fig/linear_model/pca/U%s_' %i + str(name) + '.png')
        plt.imshow(projection_vols[:, :, 18, i])   
        plt.colorbar()
        plt.title('PCA - Projection on U' + str(i) + ' vector from SVD \n ' + str(name))
        plt.savefig('../plots/fig/linear_model/pca/%s_projection_U%s' %(str(name), str(i))\
	+ '.png')
        plt.close()
    #TODO: remove the mean
    s = []
    for i in S:
        s.append(i/np.sum(S))
    np.savetxt('../plots/fig/linear_model/pca/%s_variance_explained' %str(name) + '.txt', np.array(s[:50]))
    ind = np.arange(len(s[1:50]))
    plt.bar(ind, s[1:50], width=0.25)
    plt.xlabel('S indices')
    plt.ylabel('Explained variance in percent')
    plt.title('Variance explained graph \n' + str(name))
    plt.savefig('../plots/fig/linear_model/pca/%s_variance_explained' %str(name) + '.png')
    #plt.show()
    
    X_pca = np.ones((n_trs, P+1))
    for i in range(3):
        X_pca[:,i] = convolved[i]
    linear_drift = np.linspace(-1, 1, n_trs)
    X_pca[:,3] = linear_drift
    quadratic_drift = linear_drift ** 2
    quadratic_drift -= np.mean(quadratic_drift)
    X_pca[:,4] = quadratic_drift
    X_pca[:,5] = U[:, 0]
#    np.savetxt('%s_design_matrix_pca.txt'%(str(name)), X_pca)
#    X_pca = np.loadtxt('ds005_sub001_t1r1_design_matrix_pca.txt')
    plt.imshow(X_pca, aspect=0.25)
    B_pca = npl.pinv(X_pca).dot(Y)
#    np.savetxt('%s_betas_pca.txt'%(str(name)), B_pca)
#    B_pca = np.loadtxt('ds005_sub001_t1r1_betas_pca.txt')
    b_pca_vols = np.zeros(vol_shape + (P+1,))
    b_pca_vols[in_brain_mask, :] = B_pca.T
    for i in range(5):
        plt.imshow(b_pca_vols[:, :, 18, i])
        plt.colorbar()
	plt.title('In brain voxel model - projection on X%s - PCA \n %s'%(i,str(name)))
	plt.savefig('../plots/fig/linear_model/pca/%s_noise_modelX%s_PCA'%(str(name), str(i))+\
	'.png')
	#plt.show()
        plt.close()
    residuals = Y - X.dot(betas) 
    df = X.shape[0] - npl.matrix_rank(X)
    MRSS = np.sum(residuals ** 2 , axis=0) / df
    residuals_pca = Y - X_pca.dot(B_pca)
    df_pca = X_pca.shape[0] - npl.matrix_rank(X_pca)
    MRSS_pca = np.sum(residuals_pca ** 2 , axis=0) / df_pca
    np.savetxt('../plots/fig/linear_model/pca/%s_mean_mrss_vals.txt'%(str(name)),\
               (np.mean(MRSS), np.mean(MRSS_pca)))
    pdb.set_trace()

