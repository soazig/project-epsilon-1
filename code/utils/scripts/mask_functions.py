"""mask_functions.py

A collection of functions to make masks on data.
See test_* functions in this directory for nose tests
"""
import sys, os, pdb
import numpy as np
import nibabel as nib

def get_mask_filtered_data(img_path, mask_path):
    """Return the masked filtered data

    Parameters
    ----------
    img_path: string
    	path to the 4D data
    
    mask_path: string
        path to the mask function

    Return
    ------
    masked_func: 4D array
        masked filtered data
    
    """
    func_img = nib.load(img_path)
    mask_img = nib.load(mask_path)
    mask = mask_img.get_data()
    func_data = func_img.get_data()
    # Make data 4D to prepare for "broadcasting"
    mask = np.reshape(mask, mask.shape + (1,))
    # "Broadcasting" expands the final length 1 dimension to match the func data
    masked_func = nib.Nifti1Image(func_data, func_img.affine, func_img.header)
    # nib.save(masked_func, 'masked_' + img_name )
    return masked_func


def make_mask_img(data_3d, mask_3d):
    """Apply mask on a 3D image and return the masked data

    Parameters
    ----------
    data_3d: 3D numpy array
        The subject's run image data
    mask_3d: 3D numpy array    
        The mask for the corresponding data_3d
	has values 1 inside the brain and values 0 outside
    
    Return
    ------
    masked_func: 3D numpy array
        masked data
    
    """
    # Make sure the data_3d and mask_3d have same shape
    assert data_3d.shape == mask_3d.shape, \
           "Data and Mask shapes differ \n" \
           + "data shape: %s" %(data_3d.shape) \
	   + "maske shape: %s" %(mask_3d.shape)
    return data_3d * mask_3d

