import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
import numpy as np
from glm_func import *
#from convolution_normal_script import X_matrix
#from convolution_high_res_script import X_matrix_high_res
#from load_BOLD import *
import nibabel as nib
import matplotlib.pyplot as plt
from smoothing import *

# Create the necessary directories if they do not exist
dirs = ['../../../txt_output/mrss',\
        '../../../fig/glm_fitted']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

# Locate the different paths
project_path = '../../../'
# TODO: change it to relevant path
conv_path = project_path + 'txt_output/conv_normal/'
conv_high_res_path = project_path + 'txt_output/conv_high_res/'

# select your own subject
#subject_list = [str(i) for i in range(1,17)]
subject_list = ['1','5']
run_list = [str(i) for i in range(1,2)]
conv_list = [str(i) for i in range(1,5)]

txt_paths = [('ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv'+ c.zfill(3),\
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv001_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv002_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv003_canonical.txt', \
              conv_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv004_canonical.txt', \
              '../../../data/ds005/sub' + s.zfill(3) + '/BOLD/task001_run' \
              + r.zfill(3) + '/bold.nii.gz',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv_001_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv_002_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv_003_high_res.txt',\
              conv_high_res_path + 'ds005_sub' + s.zfill(3) + '_t1r' + r +'_conv_004_high_res.txt') \
                for r in run_list \
                for s in subject_list \
                for c in conv_list]

print("\n====================================================")

for txt_path in txt_paths:
# get 4_d image data
    name = txt_path[0]
    print("Starting glm analysis for subject " +name[9:12]+ " condition " + name[24])    
    img = nib.load(txt_path[5])
    data_int = img.get_data()
    data = data_int.astype(float)
    p = 5
    # p is the number of columns in our design matrix
    # it is the number of convolved column plus 1 (a column of 1's)
    
    X_matrix1 = np.loadtxt(txt_path[1])
    X_matrix2 = np.loadtxt(txt_path[2])
    X_matrix3 = np.loadtxt(txt_path[3])
    X_matrix4 = np.loadtxt(txt_path[4])
    X_matrix = np.ones((len(X_matrix1),p))
    X_matrix[...,1] = X_matrix1
    X_matrix[...,2] = X_matrix2
    X_matrix[...,3] = X_matrix3
    X_matrix[...,4] = X_matrix4
    
    X_matrix_high_res1 = np.loadtxt(txt_path[6])
    X_matrix_high_res2 = np.loadtxt(txt_path[7])
    X_matrix_high_res3 = np.loadtxt(txt_path[8])
    X_matrix_high_res4 = np.loadtxt(txt_path[9])
    X_matrix_high_res = np.ones((len(X_matrix1),p))
    X_matrix_high_res[...,1] = X_matrix_high_res1
    X_matrix_high_res[...,2] = X_matrix_high_res2
    X_matrix_high_res[...,3] = X_matrix_high_res3
    X_matrix_high_res[...,4] = X_matrix_high_res4

    beta_4d = glm_beta(data,X_matrix)
    MRSS, fitted, residuals = glm_mrss(beta_4d, X_matrix, data)

    # smooth the data and re-run the regression
    data_smooth = smoothing(data,1,range(data.shape[-1]))
    beta_4d_smooth = glm_beta(data_smooth,X_matrix)
    MRSS_smooth, fitted_smooth, residuals_smooth = glm_mrss(beta_4d_smooth, X_matrix, data_smooth)

    # use high resolution to create our design matrix
    beta_4d_high_res = glm_beta(data,X_matrix_high_res)
    MRSS_high_res, fitted_high_res, residuals_high_res = glm_mrss(beta_4d_high_res, X_matrix_high_res, data)


    plt.plot(data[4,22,11], label = "actual")
    plt.plot(fitted[4,22,11], label = "fitted")
    plt.plot(fitted_high_res[4,22,11], label = "fitted_high_res")

    plt.title(name[0:17]+"voxel (4,22,11) actual vs fitted")
    plt.legend(loc = "upper left", fontsize = "smaller")
    plt.savefig(dirs[1] + '/'+ name[0:17]+ "_glm_fitted.png")
    plt.close()

    location_of_txt= dirs[0]
    file = open(location_of_txt+ '/' + name[0:17] +'_mrss_result.txt', "w")
    file.write("MRSS of multiple regression for" +name[0:17]+ " is: "+str(np.mean(MRSS))+"\n")
    file.write("\n")
    file.write("MRSS of multiple regression for" +name[0:17]+ " using the smoothed data is: "+str(np.mean(MRSS_smooth))+"\n")
    file.write("\n")
    file.write("MRSS of multiple regression for" +name[0:17]+ " using high_res design matrix is: "+str(np.mean(MRSS_high_res))+"\n")
    file.close()
print("GLM analysis done.")
print("See MRSS project-epsilon/txt_output/mrss/" + name[0:17])


