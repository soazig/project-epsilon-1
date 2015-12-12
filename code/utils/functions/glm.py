import numpy as np
import numpy.linalg as npl

def glm_beta(data_4d, X):
    """
    Return a tuple of the estimated coefficients in 4 dimensions and 
    the design matrix. 
    
    Parameters
    ----------
    data_4d: numpy array of 4 dimensions 
        The image data of one subject one run
    X: numpy array of 2 dimensions
        design matrix of four columns, each for one condition, one run, one subject
        
    Note that the fourth dimension of `data_4d` (time or the number 
    of volumes) must be the same as the number of rows of `X`. 
    
    Returns
    -------
    glm_results : 
        Estimated coefficients in 4 dimensions
    """
    assert(X.shape[0] == data_4d.shape[-1])
    data_2d = np.reshape(data_4d, (-1, data_4d.shape[-1]))
    B = npl.pinv(X).dot(data_2d.T)

    B_4d = np.reshape(B.T, data_4d.shape[:-1] + (-1,))
    return B_4d

def glm_mrss(B_4d, design, data_4d):
    """
    Return a tuple of the MRSS in 3 dimensions, fitted values in 4 
    dimensions, and residuals in 4 dimensions. 
    
    Parameters
    ----------
    B_4d: numpy array of 4 dimensions
        The estimated coefficients
    design: numpy array
        The design matrix used to get the estimated coefficients
    data_4d: numpy array of 4 dimensions 
        The corresponding image data
    
    Returns
    -------
    diagnostics : tuple
        MRSS (3d), fitted values (4d), and residuals (4d).
    """
    B_2d = np.reshape(B_4d, (-1, B_4d.shape[-1])).T
    data_2d = np.reshape(data_4d, (-1, data_4d.shape[-1]))
    
    fitted = design.dot(B_2d)
    residuals = data_2d.T - fitted
    df = design.shape[0] - npl.matrix_rank(design)
    MRSS = (residuals**2).sum(0)/df
    
    MRSS_3d = np.reshape(MRSS.T, data_4d.shape[:-1])
    fitted_4d = np.reshape(fitted.T, data_4d.shape)
    residuals_4d = np.reshape(residuals.T, data_4d.shape)
    return MRSS_3d, fitted_4d, residuals_4d

