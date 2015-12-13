"""
Purpose:
-----------------------------------------------------------------------------------
We try to capture the significance of gain and loss amount condition for each subjects.
We fit the logistic regression line based on their responses on the experiment. The slope 
of the fitted line illustrates the subject's sensitivity on either gain or loss amount. 
-----------------------------------------------------------------------------------
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit, ols
import sys

#PREDICT SUBJECT's DECISION TO WHETHER GAMBLE OR NOT GIVEN WITH GAIN AND LOSS AMOUNT using LOGISTIC REGRESSION
#predictors : gain, loss, respnum


def add_gainlossratio(run):
	""" Return the behavioral data of a subject with added 'ratio' column
 
    Parameters
    ----------
    run : data.frame
        Behavior data over 3 runs (combined) of a subject
    
    Returns
    -------
    run : data.frame
        behavioral data added the ratio column (ratio : gain/loss)

    """

	gain = run.ix[:,1]
	loss = run.ix[:,2]
	run['ratio'] = gain/loss

	return run

def organize_columns(run):
	""" move around the columns to get the data frame be ready for logistic regression
 
    Parameters
    ----------
    run : data.frame
        behavioral data added the ratio column (ratio : gain/loss)

    Returns
    -------
    run_final : data.frame
        behavioral data frame with organized columns

    """


	# drop onset column
	a = run.drop('onset', 1) 
	
	# drop PTval column
	run_organized = a.drop('PTval', 1) 

	# get the column names
	cols = run_organized.columns.tolist() 

	# put respcat column into front
	cols.insert(0, cols.pop(cols.index('respcat'))) 

	cols.insert(3, cols.pop(cols.index('ratio')))

	# reorganize
	run_organized = run_organized.reindex(columns= cols) 

	# drop error(rescap=-1) in experiment
	run_final = run_organized.drop(run_organized[run_organized.respcat == -1].index) 
	
	return run_final


def log_regression(run_final):
	"""Do logistic regression on train cols to predict the subject's decision
	
	Parameters
    ----------
    run : data.frame
        behavioral data frame with organized columns

    Returns
    -------
    logit_pars : logistic regression result summary
        the logistic regression result summary

    """
	# do logistic regression
	x = logit("respcat ~ gain + loss", run_final).fit()

	# check the summary
	print(x.summary())

	#store the parameters of logistic regression
	logit_pars = x.params

	return logit_pars

if __name__ == '__main__':
    a=add_gainlossratio(int(sys.argv[1]))
    b=organize_columns(a)
    log_regression(b)









