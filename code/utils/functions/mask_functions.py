"""mask_functions.py

A collection of functions to make masks on data.
See test_* functions in this directory for nose tests
"""
from __future__ import print_function, division
import sys, os, pdb
import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal

def make_mask_filtered_data(img_path, mask_path):
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


def apply_mask(data_3d, mask_3d):
    """Apply mask on a 3D image and return the masked data

    Parameters
    ----------
    data_3d: 3D numpy array
        The subject's run image data
    mask_3d: 3D numpy array   
        The mask for the corresponding data_3d
        has 1 for the positions to select and 
	0 for the positions to filter out.
    Return
    ------
    masked_func: 3D numpy array
        masked data with the values 
    
    """

    a = data_3d.shape
    b = mask_3d.shape
    assert(a == b), "Data and mask shape differ \n" \
    + "Data shape is: %s\nMask shape is: %s" %(data_3d.shape, mask_3d.shape)
    return data_3d * mask_3d 

