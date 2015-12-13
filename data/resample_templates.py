""" Resample template to example filtered functional dataset

Templates from:
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip

Filtered functional from:
    http://nipy.bic.berkeley.edu/rcsds/ds005/sub001/model/model001/task001_run001.feat/filtered_func_data_mni.nii.gz
"""
from os.path import splitext
from glob import glob

import numpy.linalg as npl

from scipy.ndimage import affine_transform

import nibabel as nib

filtered_func = nib.load('filtered_func_data_mni.nii.gz')
filtered_shape = filtered_func.shape[:3]  # First 3 dimensions

for mni_fname in glob('mni_icbm*'):
    template_img = nib.load(mni_fname)
    template_data = template_img.get_data()
    vox2vox = npl.inv(template_img.affine).dot(filtered_func.affine)
    M, trans = nib.affines.to_matvec(vox2vox)
    resampled = affine_transform(template_data, M, trans,
                                 output_shape=filtered_shape)
    froot, ext = splitext(mni_fname)
    new_name = froot + '_2mm' + ext
    new_img = nib.Nifti1Image(resampled, filtered_func.affine,
                              template_img.header)
    nib.save(new_img, new_name)
