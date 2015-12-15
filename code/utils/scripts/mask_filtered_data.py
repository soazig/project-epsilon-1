"""mask_functions.py

A collection of functions to make masks on data.
See test_* functions in this directory for nose tests
"""
from __future__ import print_function, division

import sys, os, pdb
import numpy as np
import nibabel as nib
import numpy.linalg as npl

from os.path import splitext
from numpy.testing import assert_array_equal
from scipy.ndimage import affine_transform

def make_bool_mask(mask_path):
    """Return a bool type mask from binary mask
    Parameters
    ----------
    mask_path : 3D float type numpy array
        the binary array to create the mask from
	has 0 and 1

    Returns
    -------
    mask_bool : 3D bool type numpy array
        the bool array created from the binary array
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_data()
    mask_bool = mask_data == 1
    return mask_bool


def make_mask_filtered_data(func_path, mask_path):
    """Return the masked filtered data

    Parameters
    ----------
    func_path: string
    	  path to the 4D data
    
    mask_path: string
        path to the mask function

    Return
    ------
    masked_func: 4D array
        masked filtered data
    
    """
    func_img = nib.load(func_path)
    mask_img = nib.load(mask_path)
    mask = mask_img.get_data()
    func_data = func_img.get_data()
    # Make data 4D to prepare for "broadcasting"
    mask = np.reshape(mask, mask.shape + (1,))
    # "Broadcasting" expands the final length 1 dimension to match the func data
    masked_name = 'masked_' + str(func_path.split('/')[-1])
    masked_func_path = os.path.join(\
       str('/'.join(func_path.split('/')[:-1])), masked_name)
    masked_func = nib.Nifti1Image(func_data, func_img.affine, func_img.header)
    nib.save(masked_func, masked_func_path)
    return masked_func

if __name__=='__main__':
    project_path='../../../'
    func_path = \
    project_path+\
    'data/ds005/sub001/model/model001/task001_run001.feat/filtered_func_data_mni.nii.gz'
    img = nib.load(func_path) 
    mask_path = project_path+\
       'data/mni_icbm152_t1_tal_nlin_asym_09c_mask_2mm.nii'
    masked_func = make_mask_filtered_data(func_path, mask_path)
