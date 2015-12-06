import sys, os
sys.path.append("../utils")
import numpy as np
from glm import *
from convolution_normal_script import X_matrix
from convolution_high_res_script import X_matrix_high_res
from load_BOLD import *
import nibabel as nib
import matplotlib.pyplot as plt
from smoothing import *

# Create the necessary directories if they do not exist
dirs = ['../../txt_output', '../../txt_output/FOLDER_NAME',\
        '../../fig','../../fig/FOLDER_NAME']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
project_path = '../../'
# TODO: change it to relevant path
data_path = project_path + 'data/ds005/'

# TODO: uncomment for final version select your own subject
#subject_list = [str(i) for i in range(1,17)]
subject_list = ['1']
#TODO: select your own run id
run_list = [str(i) for i in range(1,2)]
images_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r, \
                 data_path + 'sub' + s.zfill(3) + '/BOLD/task001_run' \
                 + r.zfill(3) + '/bold.nii.gz') for r in run_list \
                 for s in subject_list]
#TODO: remove the following
location_of_plot = "../../plots/"
location_of_data = "../../data/ds005/"

# get 4_d image data
for s in subject_list:
    for r in run_list: 
        data = load_img(s,r)
	name = 'ds005_sub' + s.zfill(3) + '_t1r' + r

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
