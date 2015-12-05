import sys
sys.path.append("../utils")
import numpy as np
from glm import *
from convolution_normal_script import X_matrix
from convolution_high_res_script import X_matrix_high_res
from load_BOLD import *
import nibabel as nib
import matplotlib.pyplot as plt
from smoothing import *

location_of_plot = "../../plots/"
location_of_data = "../../data/ds005/"

# get 4_d image data
data = load_img(1,1)

beta_4d = glm_beta(data,X_matrix)
MRSS, fitted, residuals = glm_mrss(beta_4d, X_matrix, data)

# smooth the data and re-run the regression
data_smooth = smoothing(data,1,range(data.shape[-1]))

beta_4d_smooth = glm_beta(data_smooth,X_matrix)
MRSS_smooth, fitted_smooth, residuals_smooth = glm_mrss(beta_4d_smooth, X_matrix, data_smooth)


# use high resolution to create our design matrix
beta_4d_high_res = glm_beta(data,X_matrix_high_res)
MRSS_high_res, fitted_high_res, residuals_high_res = glm_mrss(beta_4d_high_res, X_matrix_high_res, data)


print ("MRSS of multiple regression: "+str(np.mean(MRSS)))
print ("MRSS of multiple regression by using high resoultion design matrix: "+str(np.mean(MRSS_high_res)))
print ("MRSS of multiple regression using the smoothed data: "+str(np.mean(MRSS_smooth)))

plt.plot(data[4,22,11], label = "actual")
plt.plot(fitted[4,22,11], label = "fitted")
plt.plot(fitted_high_res[4,22,11], label = "fitted_high_res")

plt.title("Subject 001, voxel (4,22,11) actual vs fitted")
plt.legend(loc = "upper left", fontsize = "smaller")
plt.savefig(location_of_plot + "glm_fitted.png")
plt.close()

location_of_txt="../txt_files/"
file = open(location_of_txt+'ds005_mrss_result.txt', "w")
file.write("MRSS of multiple regression for subject 1, run 1 is: "+str(np.mean(MRSS))+"\n")
file.write("\n")
file.write("MRSS of multiple regression for subject 1, run 1, using the smoothed data is: "+str(np.mean(MRSS_smooth))+"\n")
file.close()
