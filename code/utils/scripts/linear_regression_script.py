""" Linear Regression on Begavioral data """

import sys, os
#TODO : later change this
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

from linear_regression import *
from scipy import stats


# Get the data
#all_subjects=['001','005']
all_subjects= ['001', '002' ,'003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016']
# data_dir = "/Users/macbookpro/Desktop/stat159_Project/"
project_path = '../../../'
data_dir = project_path+'data/'

# directory for plot
dirs = ['../../../fig','../../../fig/lin']
for d in dirs:
    if not os.path.exists(d):
            os.makedirs(d)

########################
# combine all the data #
########################


all_data = []

# Loop for 16 subjects to get the data
for i in all_subjects:
	temp = load_data(i, data_dir)
	#temp = load_data(i)
	# if there's no such dataset, then stop running the loop and return nothing
	if temp is None:
		break
	all_data.append(temp)

# Concat the data
all_data = pd.concat(all_data)

# Calculate the difference for future use
all_data['diff'] = all_data['gain'] - all_data['loss']

# Calculate the ratio and other variables for future use
all_data['ratio'] = all_data['gain'] / all_data['loss']
all_data['log_gain'] = np.log(all_data['gain'])
all_data['log_loss'] = np.log(all_data['loss'])
all_data['log_diff'] = np.log(all_data['diff'])
all_data['log_ratio'] = np.log(all_data['ratio'])

# # Remove the rows that have respcat == -1 
#(meaning that the individual chooses to not participate the activity)
all_data = all_data[np.logical_not(all_data['respcat'] == -1)]


########################
#  Load each subject   #
########################

data_each = []

for i in all_subjects:
	data_each.append(load_data(i, data_dir))
	#data_each.append(load_data(i))

for i in range(len(all_subjects)):
	data_each[i]['ratio'] = data_each[i]['gain'] / data_each[i]['loss']



##############################
#  Peform linear regression  #
##############################

data = all_data

# Run the linear_regression function to get the summary
beta1, pvalues1 = linear_regression(data, 'RT', 'gain', 'loss')

beta2, pvalues2 = linear_regression(data, 'RT', 'ratio')

beta3, pvalues3 = linear_regression(data, 'RT', 'diff')


#######################
#        Plot         #
#######################


# PLot the simple regression
# Since the ratio is the most significant predictor

y = data['RT']
x = data['ratio']
n = len(data)
# Design X matrix
X = np.column_stack((np.ones(n), x))
# Get the beta
B = npl.pinv(X).dot(y)

# Get the regression line
#def my_line(x, B):
#    # Best prediction 
#	return B[0] + B[1] * x

x_vals = [0, max(x)] # since the ratio wouldn't be negative
y_vals = [B[0], (B[0] + B[1] * max(x) )]
# Plot the simple linear regression
plt.plot(x, y, '+')
plt.xlabel('ratio (gain/loss)')
plt.ylabel('Response Time')
plt.plot(x_vals, y_vals)
plt.title('Ratio vs Response Time with predicted line')
plt.show()
plt.savefig(dirs[1]+'/linear_regression.png')
plt.close()


