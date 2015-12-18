.PHONY: download_data validate_data download_filtered_data download_test_data
download_data:
	wget -P ../code/utils/scripts/ http://www.jarrodmillman.com/rcsds/_downloads/actc.txt
	wget http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz
	tar -xvzf ds005_raw.tgz
	rm ds005_raw.tgz 

download_filtered_data:
	variable=Stop python ../data/filtered_data_sh_script.py
	variable=Stop bash ../data/dwn_filtered_data_script.sh
	wget -N http://nipy.bic.berkeley.edu/rcsds/mni_icbm152_nlin_asym_09c_2mm/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii
	wget -N http://nipy.bic.berkeley.edu/rcsds/mni_icbm152_nlin_asym_09c_2mm/mni_icbm152_t1_tal_nlin_asym_09c_mask_2mm.nii

validate_data:
	wget -N http://openfmri.s3.amazonaws.com/tarballs/ds005_raw_checksums.txt
	python data.py
 	
download_test_data:
	wget -N -P ../data/ds114/sub009/BOLD/task002_run001/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1.nii
	wget -N -P ../data/ds114/sub009/behav/task002_run001/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1_cond.txt
	wget -N -P ../data/ds114/sub009/behav/task002_run001/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1_conv.txt

test: 
	nosetests tests 

coverage:
	nosetests tests --with-coverage --cover-package=data_hashes.py, tests/test_get_check_hashes.py 

download_all:
	make download_data
	make download_filtered_data
	make download_test_data



