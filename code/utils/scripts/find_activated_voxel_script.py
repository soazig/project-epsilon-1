from __future__ import division, print_function, absolute_import
import numpy as np
import sys
sys.path.append(".././utils")
import nibabel as nib
import matplotlib.pyplot as plt
from load_BOLD import *
from t_test import *
from find_activated_voxel_functions import *
from convolution_normal_script import X_matrix
from convolution_high_res_script import X_matrix_high_res

location_of_data = "../../data/ds005/sub001/BOLD/task001_run001/"
data = load_img(1,1)
beta, t, df,p=t_test(data,X_matrix_high_res)
shape = data.shape[:3]

lst = find_activated_voxel(shape, p)
location_of_txt="../txt_files/"

for i in range(1,len(lst)+1):
    np.savetxt(location_of_txt+'ds005_sub001_t1r1_position%s.txt'%(str(i)),lst[i-1].ravel())

