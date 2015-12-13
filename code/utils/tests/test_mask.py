"""test_mask.py
Tests for the functions in the mask_functions.py

Run with:
    nosetests test_mask.py
"""
from __future__ import print_function
import os, sys
import numpy as np
from numpy.testing import assert_array_equal

#Append path to functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
from mask_functions import *


def test_apply_mask():
    # We make a 3D array of shape (3,3,2)
    slab0 = np.reshape(np.arange(9), (3, 3))
    slab1 = np.reshape(np.arange(100, 109), (3, 3))
    arr_3d = np.zeros((2, 3, 3))
    arr_3d[0, :, :] = slab0
    arr_3d[1, :, :] = slab1
    # We make a mask as a 3D array of shape (2,3,3)
    # with zeros on the 2nd component of the 1st dimension
    mask_3d = np.zeros((2, 3, 3))
    mask_3d[0] = np.ones((3,3))
    # Defined the resulting masked array
    masked_arr = np.zeros((2,3,3))
    masked_arr[0, :, :] = slab0
    assert_array_equal(apply_mask(arr_3d, mask_3d),masked_arr)


def test_make_binary_mask():
    # We make a 3D array of shape (3,3,2)
    slab0 = np.reshape(np.arange(9), (3, 3))
    slab1 = np.reshape(np.arange(100, 109), (3, 3))
    arr_3d = np.zeros((2, 3, 3))
    arr_3d[0, :, :] = slab0
    arr_3d[1, :, :] = slab1
    # We make a mask boolean as a 3D array of shape (2,3,3)
    # that filtered the values below 100
    mask_bool = arr_3d < 100
    mask_3d = np.zeros((2, 3, 3))
    mask_3d[0] = np.ones((3,3))
    assert_array_equal(make_binary_mask(arr_3d,mask_bool), mask_3d)
    arr_2d = np.arange(9).reshape((3,3))
    mask_bool2d = arr_2d < 10
#    make_binary_mask(arr_3d,mask_bool2d)


