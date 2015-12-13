""" Tests functions in outliers module
Largely lifted straight from the tests provided for diagnostics.py in HW2 and janewliang's diagnosis_script.py

Run the test with:
    nosetests test_outliers.py
"""

# Loading modules.
import os
import sys
import numpy as np
import nibabel as nib
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal


# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

from outlier import *
project_path=os.path.join(os.path.dirname(__file__), '../../../')

def test_vol_std():
    # create the test data
    shape = (2, 3, 4)
    L = np.prod(shape)
    t = 10
    data_2d = np.random.normal(size=(L, t))
    test_stds = np.std(data_2d, axis=0)
    data_4d = np.reshape(data_2d, shape + (t,))
    stds = vol_std(data_4d)
    assert_almost_equal(test_stds, stds)

def test_iqr_outliers():
    data = np.arange(101)
    # the data is from 0 to 100, so the iqr = 50
    exp_lo = 25 - 1.5*50
    exp_hi = 75 + 1.5*50
    indices, thresholds = iqr_outliers(data)
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reverse the data
    # check the results will not change
    indices, thresholds = iqr_outliers(data[::-1])
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Add outliers
    data[0] = -100
    data[1] = 200
    data[100] = 1
    indices, thresholds = iqr_outliers(data)
    assert_array_equal(indices, [0, 1])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reversed the data
    indices, thresholds = iqr_outliers(data[::-1])
    assert_array_equal(indices, [99, 100])
    assert_equal(thresholds, (exp_lo, exp_hi))

