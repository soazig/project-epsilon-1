from __future__ import division, print_function, absolute_import
import numpy as np
from functools import reduce
from itertools import product
from t_test import *

def size(shape, axis=None):
    """
    Return the number of elements along a given axis.
    Parameters
    ----------
    shape : tuple
        The shape of the input array
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.
    Returns
    -------
    element_count : int
        Number of elements along the specified axis.
    """
    if axis is None:
        size = 1
        for e in shape:
            size *= e
        return size
    return shape[axis]


def get_increment(shape):
    """
    Return the increments corresponding to each axis or dimension in the shape.
    Parameters
    ----------
    shape : tuple
        The shape of the input array.
    Returns
    -------
    increment_per_axis : list
        The number of positions in the linear order that you need to move
        to retrieve the element specified by incrementing each axis
        in the corresponding index.
    """
    inc = [reduce(lambda x, y: x * y, shape[1 + i:], 1)
           for i in range(len(shape[1:]))]
    return inc + [1]


def get_index(shape, position):
    """
    Return the index in the multidimensional array that is corresponds to the
    element in the given position in the linear ordered.
    Parameters
    ----------
    shape : tuple
        The shape of the input array
    position : int
        The position in the linear ordering
    Returns
    -------
    index : tuple (same number of elements as shape)
        The index in the multidimensional array specified by the
        position in linear order
    """
    assert position < size(shape)
    inc = get_increment(shape)
    index = list()
    for i in inc:
        x = position // i
        position -= x * i
        index.append(x)
    return tuple(index)

def find_activated_voxel(shape, p):
    """
    Return the positions of the activated voxels based on the p-values we get from the t-test
    Parameters
    ----------
    shape : tuple
            The shape of the first dimension of our data
    p: numpy.ndarray
       The p values for our predictors and voxels
    Returns
    -------
    position : numpy.ndarray
        The index of our activated voxels
        position1...4 correspond to the cond001...4 .txt in model_one
    """
    # get the number of predictors(betas) first
    n = p.shape[0]-1
    loc={}
    lst=[]
    position=[]
    for t in range(1,n+1):
        loc["loc{0}".format(t)] = [i for i,j in enumerate(p[t,...]) if j < 0.05]
        for i in loc["loc{0}".format(t)]:
            position.append(get_index(shape,i))
        position = np.asarray(position)
        lst.append(position)
        position=[]

    return lst

