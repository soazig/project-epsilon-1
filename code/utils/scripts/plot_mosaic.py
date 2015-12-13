"""

"""
from __future__ import division, print_function
import sys, os, pdb
import numpy as np
import nibabel as nib

def plot_mosaic(img_data):
    """ Return a mosaic plot for each slice of
        the 3rd dimension of img_data

    Parameters:
    ----------
    img_data = 3D array
    
    Returns:
    -------
    grid_2D : a 2D image with each slice of
        the 3rd dimension of img_data plotted
	in a mosaic
    """
    n_slices = img_data.shape[2]
    # Dimensions of the mosaic grid
    n_rows = int(np.ceil(float(np.sqrt(n_slices))))
    n_cols = int(np.ceil(float(n_slices)/float(n_rows)))
    # Define the 2D mosaic
    grid_2D = np.zeros((n_rows*img_data.shape[0], n_cols*img_data.shape[1]))
    z = 0
    for i in range(n_rows):
        for j in range(n_cols):
	    if z < n_slices:
	        grid_2D[i*img_data.shape[0]:(i+1)*img_data.shape[0],\
	        j*img_data.shape[1]:(j+1)*img_data.shape[1]] = img_data[:,::-1,z].T
            z += 1
    return grid_2D

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    img = nib.load(\
    '../../../data/ds005/sub001/BOLD/task001_run001/bold.nii.gz') 
    img_data = img.get_data()
    mean_data = np.mean(img_data, axis=-1)
    plt.imshow(plot_mosaic(mean_data), cmap='gray', alpha=1)
    plt.colorbar()
    plt.show()
