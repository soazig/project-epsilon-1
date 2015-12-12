import numpy as np
import itertools
import scipy
from scipy.ndimage.filters import gaussian_filter

def smoothing(data, sigma, time):
	'''
	Return a smoothed array that has the same shape as our input.
	
	Parameters
	----------
	data: A numpy array. It is the image data for one subject, one run.
    
	sigma: Scalar or sequence of scalars. Standard deviation for Gaussian
    kernel. The standard deviations of the Gaussian filter are given for
    each axis as a sequence, or as a single number, in which case it is equal
    for all axes.
    
    time: The time slice, which is the fourth dimension of our image data.
    
	Returns
	-------
    ndarray
    Returned array of same shape as `input`.
	'''
	timeslice = data[...,time]
	smoothed = scipy.ndimage.filters.gaussian_filter(timeslice, sigma)
	return smoothed
