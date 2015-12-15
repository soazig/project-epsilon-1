from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
from __future__ import absolute_import
import scipy.stats
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib



def hrf(times):
     """ Return values for HRF at given times """
     # Gamma pdf for the peak
     peak_values = gamma.pdf(times, 6)
     # Gamma pdf for the undershoot
     undershoot_values = gamma.pdf(times, 12)
     # Combine them
     values = peak_values - 0.35 * undershoot_values
     # Scale max to 0.6
     return values / np.max(values) * 0.6


""" 
Functions to work with standard OpenFMRI stimulus files
The functions have docstrings according to the numpy docstring standard - see:
https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""


def events2neural(task, tr, n_trs):
    """ Return predicted neural time course from event file `task_fname`
    Parameters
    ----------
    task_fname : str
        Filename of event file
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run
    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    #if task.ndim != 2 or task.shape[1] != 3:
    #    raise ValueError("Is {0} really a task file?", task_fname)
    # Convert onset, duration seconds to TRs
    task[:, :2] = task[:, :2] / tr
    # Neural time course from onset, duration, amplitude for each event
    time_course = np.zeros(n_trs)
    for onset, duration, amplitude in task:
        time_course[onset:onset + duration] = amplitude
    return time_course

def events2neural_high(cond_data, TR=2, n_trs=240, tr_div=100):
    """Return predicted neural time course in the case when onsets 
    are not equally spaced and do not start on a TR.
    
    Parameters:
    ----------
    cond_data : np.array
        np.array of the condition
    TR: float (default: 2)
        TR in seconds
    n_trs: int (default: 240)
        number of TRs 
    tr_div: int (default value is 10)
        step per TR
        (We want a resolution to the 10th between each TR)
    
    Return
    -------
    high_res_neural: np.array
        predicted neural time course
    """
    onsets_seconds = cond_data[:, 0] # the time when a task starts
    durations_seconds = cond_data[:, 1] # duration of a task
    amplitudes = cond_data[:, 2] # amplitudes for each different task
    onsets_in_scans = onsets_seconds / TR 
    high_res_times = np.arange(0, n_trs, 1.0/tr_div) * TR
    high_res_neural = np.zeros(high_res_times.shape)
    high_res_onset_indices = onsets_in_scans * tr_div
    high_res_durations = durations_seconds / TR * tr_div

    for hr_onset, hr_duration, amplitude \
        in list(zip(high_res_onset_indices,high_res_durations,amplitudes)):
        hr_onset = int(round(hr_onset))
        hr_duration = int(round(hr_duration))
        high_res_neural[hr_onset:hr_onset+hr_duration] = amplitude
    
    return high_res_times, high_res_neural
