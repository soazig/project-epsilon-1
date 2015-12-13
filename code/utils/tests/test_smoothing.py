""" Tests for smoothvoxels in smooth module
Run the test with:
    nosetests test_smoothing.py
"""

import os
import sys
import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
from numpy.testing import assert_almost_equal
from nose.tools import assert_not_equals


# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load smoothing function.
from smoothing import smoothing
project_path = os.path.join(os.path.dirname(__file__), '../../../')

def test_smooth():
    # Read in the image data.
    img = nib.load(project_path + \
          'data/ds114/sub009/BOLD/task002_run001/ds114_sub009_t2r1.nii')
    data = img.get_data()

    # Run the smoothing function with sigma 0 at time 12
    non_smoothed_data = smoothing(data, 0, 12)

	# assert that data at time 12 and non_smoothed_data are equal since sigma = 0
    assert_almost_equal(data[..., 12], non_smoothed_data)

	# Run the smoothvoxels function with sigma 5 at time 100
    smoothed_data = smoothing(data, 5, 100)
	# assert that data at time 16 and smoothed_data are not equal
    assert_not_equals(data[..., 100].all(), smoothed_data.all())
