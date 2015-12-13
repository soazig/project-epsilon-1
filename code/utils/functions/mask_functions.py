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


def make_binary_mask(data, mask_bool):
    """Return a numpy array with 0 and 1 
    Parameters
    ----------

    Return
    ------
    """
    data = np.asarray(data)
    mask_bool = np.asarray(mask_bool)
    assert(len(data.shape) == len(mask_bool.shape)),\
    "Data and mask shape differ \n" \
    + "Data dim is: %s\nMask dim is: %s" \
    %(len(data.shape), len(mask_bool.shape))
    assert(all(data.shape[i] >= mask_bool.shape[i] \
           for i in range(len(data.shape)))),\
    "Data and mask shape are not compatible"\
    +"Data shape is: %s\nMask shape is: %s"\
    %(data.shape, mask_bool.shape)
    new_mask = np.zeros(data.shape)
    new_mask[mask_bool] = 1
    return new_mask 



def apply_mask(data, mask):
    """Apply mask on an image and return the masked data

    Parameters
    ----------
    data: numpy array
        The subject's run image data
    mask: numpy array same shape as data  
        The mask for the corresponding data_3d
        has 1 for the positions to select and 
      	0 for the positions to filter out.
    Return
    ------
    masked_data: numpy array 
        Array with the values of the data at the selected positions
	and 0 for the position filtered out by the mask.
    """
    assert(data.shape == mask.shape), "Data and mask shape differ \n" \
    + "Data shape i: %s\nMask shape is: %s" %(data.shape, mask.shape)
    return data * mask 

