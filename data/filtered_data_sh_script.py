"""
This script create a bash script to be used 
in the Makefile of the data directory.
It will download the filtered data from from 
https://nipy.bic.berkeley.edu/rcsds/ds005/
and locate them in the relevant local directories
The bash script is then used in the Makefile of the
data directory to download the filtered data one
by one.

Run with:
    python filtered_data_sh_script.py
    from this directory
"""
import os
import pdb
import numpy as np

project_id = 'ds005'
subject_N = 16
run_N = 3

project_path = '../'
data_path = project_path + 'data/'

subject_list = [str(i) for i in range(1,subject_N+1)]
#subject_list = ['1','5']
run_list = [str(i) for i in range(1,run_N+1)]
base_url = 'http://nipy.bic.berkeley.edu/rcsds/'
file_name = 'filtered_func_data_mni.nii.gz'

#Write local downloading location and link on txt file
txt_out = []
for s in subject_list:
    for r in run_list:
        start = 'wget -N -P ' + data_path
        location = project_id + '/sub'+ s.zfill(3)\
           +'/model/model001/task001_run%s.feat/'%(r.zfill(3))
        link = base_url + location + file_name
        txt_out.append((start + location + ' ' + link))
np_txt_out = np.array((txt_out))
np.savetxt(data_path + 'dwn_filtered_data_script.sh', \
   np_txt_out, delimiter=" ", fmt="%s")
