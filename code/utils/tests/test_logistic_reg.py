"""test_diagnostics.py
Tests for the functions in logistic_reg.py module 

Run with:

    nosetests test_logistic_reg.py
"""


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
from logistic_reg import *
from organize_behavior_data import *
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from scipy import stats
from statsmodels.formula.api import logit, ols

def test_add_gainlossratio():
	"""Tests whether the gain/loss ratio is properly added in the data frame

	"""

	#load the subject 2's combined runs on the dataframe (use organize_behav_data.py)
	run = load_in_dataframe(2)
	gain = run.ix[:,1]
	loss = run.ix[:,2]
	#add the column for gain/loss ratio
	run['ratio'] = gain/loss
	run_mat=run.as_matrix()
	run_ratio=run_mat[:,7]
	#compare with the output from the actual output
	test_run = load_in_dataframe(2)
	test_run_added = add_gainlossratio(test_run).as_matrix()
	test_run_added_ratio = test_run_added[:,7]
	assert_array_equal(run_ratio,test_run_added_ratio)


def test_organize_columns():
	"""Tests whether columns in the data frame are organized or not for logistic regression

	"""

	run = load_in_dataframe(2)
	run_added = add_gainlossratio(run)
	a = run_added.drop('onset', 1) 
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
	test_run_final = organize_columns(run).as_matrix()
	assert_array_equal(run_final.as_matrix(),test_run_final)



def test_log_regression():
	"""Tests the results of logistic regression.
	Explore on the beta coefficient

	"""
	run = load_in_dataframe(2)
	run_added = add_gainlossratio(run)
	run_final = organize_columns(run_added)

	#fit the logistic regression line
	fitted = logit("respcat ~ gain + loss", run_final).fit()
	#get the parameters
	fitted_params = fitted.params.as_matrix()
	test_fitted_params = log_regression(run_final).as_matrix()
	assert_array_equal(fitted_params,test_fitted_params)


