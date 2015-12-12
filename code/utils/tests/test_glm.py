""" Tests for glm function in glm module
This checks the glm function.

Run at the tests directory with:
    nosetests code/utils/tests/test_glm.py

"""
# Loading modules.
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal


# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


# Load our GLM functions. 
from glm import glm_beta, glm_mrss

def test_glm_beta():
    # Read in the image data.
    img = nib.load('data/ds114/sub009/BOLD/task002_run001/ds114_sub009_t2r1.nii')
    data = img.get_data()
    # Read in the convolutions.
    p = 2
    convolved1 = np.loadtxt('data/ds114/sub009/behav/task002_run001/ds114_sub009_t2r1_conv.txt')
    # Create design matrix. 
    X_matrix = np.ones((len(convolved1), p))
    X_matrix[:, 1] = convolved1
    
    # Calculate betas, copied from the exercise. 
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    B = npl.pinv(X_matrix).dot(data_2d.T)
    B_4d = np.reshape(B.T, img.shape[:-1] + (-1,))
    
    # Run function.
    test_B_4d = glm_beta(data, X_matrix)
    assert_almost_equal(B_4d, test_B_4d)


def test_glm_mrss():
    img = nib.load('data/ds114/sub009/BOLD/task002_run001/ds114_sub009_t2r1.nii')
    data = img.get_data()
    convolved1 = np.loadtxt('data/ds114/sub009/behav/task002_run001/ds114_sub009_t2r1_conv.txt')
    X_matrix = np.ones((len(convolved1), 2))
    X_matrix[:, 1] = convolved1
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    B = npl.pinv(X_matrix).dot(data_2d.T)
    B_4d = np.reshape(B.T, img.shape[:-1] + (-1,))
    test_B_4d = glm_beta(data, X_matrix)
    
    # Pick a single voxel to check mrss functiom.
    # Calculate actual fitted values, residuals, and MRSS of voxel.
    fitted = X_matrix.dot(B_4d[12, 22, 10])
    residuals = data[12, 22, 10] - fitted
    MRSS = np.sum(residuals**2)/(X_matrix.shape[0] - npl.matrix_rank(X_matrix))
    
    # Calculate using glm_diagnostics function.
    test_MRSS, test_fitted, test_residuals = glm_mrss(test_B_4d, X_matrix, data)
    assert_almost_equal(MRSS, test_MRSS[12, 22, 10])
    assert_almost_equal(fitted, test_fitted[12, 22, 10])
    assert_almost_equal(residuals, test_residuals[12, 22, 10])

    
    


