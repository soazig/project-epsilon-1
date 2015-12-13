import numpy as np


def vol_std(data):
    """ Return standard deviation across voxels for 4D array `data`
    Parameters
    ----------
    data : 4D array
        4D array from FMRI run with last axis indexing volumes. 
    Returns
    -------
    std_values : array shape (T,)
        One dimensonal array where ``std_values[i]`` gives the standard
        deviation of all voxels contained in ``data[..., i]``.
    """
    T = data.shape[-1]
    data_2d = np.reshape(data, (-1, T))
    return np.std(data_2d, axis=0)


def iqr_outliers(arr_1d, iqr_scale=1.5):
    """ Return indices of outliers identified by interquartile range
    Parameters
    ----------
    arr_1d : 1D array
        One-dimensional numpy array, from which we will identify outlier
        values.
    iqr_scale : float, optional
        Scaling for IQR to set low and high thresholds.  Low threshold is given
        by 25th centile value minus ``iqr_scale * IQR``, and high threshold id
        given by 75 centile value plus ``iqr_scale * IQR``.
    Returns
    -------
    outlier_indices : array
        Array containing indices in `arr_1d` that contain outlier values.
    lo_hi_thresh : tuple
        Tuple containing 2 values (low threshold, high thresold) as described
        above.
    """
    pct_25, pct_75 = np.percentile(arr_1d, [25, 75])
    iqr = pct_75 - pct_25
    lo_thresh = pct_25 - iqr * iqr_scale
    hi_thresh = pct_75 + iqr * iqr_scale
    is_outlier = (arr_1d < lo_thresh) | (arr_1d > hi_thresh)
    return np.nonzero(is_outlier)[0], (lo_thresh, hi_thresh)