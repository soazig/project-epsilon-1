# project-template
[![Build Status](https://travis-ci.org/berkeley-stat159/project-epsilon.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-epsilon?branch=master)
[![Coverage Status](https://coveralls.io/repos/berkeley-stat159/project-epsilon/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-epsilon?branch=master)

# Project Epsilon
## UC Berkeley's Statistics 159/259
### Project Group Epsilon, Fall Term 2015 

_**Topic:**_ [The Neural Basis of Loss Aversion in Decision-Making Under Risk] 

## Overview
This repository attempts to reproduce the original analysis on 
"The Neural Basis of Loss Aversion in Decision-Making Under Risk" 
done by Sabrina M. Tom, Craig R. Fox, Christopher Trepel, Russell A. Poldrack. 
The imaging data were collected using the fMRI method. They were processed 
and analyzed in order to identify the regions of the brain activated by the 
decision making process. This study also investigated the relationship between 
the brain activity and the behavior of the subjects towards the gambling situations 
using a whole-brain robust regression analysis. 
Please follow the insturctions to explore on the repository.

## Directions
1. Clone the repo: `git clone https://github.com/berkeley-stat159/project-epsilon.git'
2. Install python dependencies with pip: `pip install -r requirements.txt` 

### Navigation
 - Data `make data` : Downloads the ds005 dataset including brain scan images of total 
 16 subjects. When rin from this repository, this commend will download the raw data and
 the filerted data provided. The total size of the file is ~17GB.

 - Validate `make validate_data` : Validates the downloaded data 

 - Clean `make clean` : remove compiled python files

 - Test `make test` : Tests the functions in code/utils folder

 - Coverage `make coverage` : Creates a coverage report for the functions in code/utils/ folder

 - Verbose `make verbose` : Tests the functions in code/utils folder via nosetests option

 - Report `make report` : Creates final_report.pdf under paper/ and clean the paper/ directory

 - Analysis for Subject 1 and 5 `make analysis-except-multi` : Executes all analysis (
 except for the multi comparison) and creates relevant  img files under fig/ folder
 	 - NOTICE : `make multi-comparison` will run about for 1 hour because it 
	 has to generate all the beta values for each single voxel for each subject 
	 over time-course.
 - To make all analyses `make analysis-except-multi` and then `make multi-comparison`

 - Each of analysis :
   - `make eda` 
   - `make linear`
   - `make logistic`
   - `make convolution-high`
   - `make convolution-normal`
   - `make t-test`
   - `make glm`
   - `make noise-pca`
   - `make multi-comparison`

- If you want to perform each analysis, please be aware of the following dependencies:
   - linear (prerequisites: convolution-normal, convolution-high)
   - logistic (prerequisites: convolution-normal, convolution-high)
   - t-test (prerequisites: convolution-normal, convolution-high)
   - glm (prerequesites: convolution-normal, convolution-high)
   - noise-pca (prerequisites: convolution-normal, convolution-high)
   - multi-comparison (prerequisites: convolution-normal, convolution-high, glm)

###Special note for Ross
- Please use `make filtered_data_only` to complete the download of the filtered data before runnning : `make analysis-except-multi` and then `make multi-comparison`

## Contributors
- Min Gu Jo ([`mingujo`](https://github.com/mingujo))
- Soazig Kaam ([`soazig`](https://github.com/soazig))
- Zhuangdi Li ([`lizhua`](https://github.com/lizhua))
- Ye Zhi ([`ye-zhi`](https://github.com/ye-zhi))
- Timothy Yu ([`timothy1191xa`](https://github.com/timothy1191xa))
