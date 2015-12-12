"""mask_functions.py

This script is using the data provided is used for the design matric for our linear regression.
We explore the influence of linear and quadratic drifts on the model 
performance.

Template from:
http://nipy.bic.berkeley.edu/rcsds/mni_icbm152_nlin_asym_09c_2mm/mni_icbm152_t1_tal_nlin_asym_09c_mask_2mm.nii

Filtered functional from:
	http://nipy.bic.berkeley.edu/rcsds/ds005/

Run with:
	python make_masked_filtered_data.py
"""
import sys, os, pdb
import numpy as np
import nibabel as nib

def make_mask_filtered_data(img_path, mask_path):
    """

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


def make_mask():
    """

    """
    return NotImplemented


