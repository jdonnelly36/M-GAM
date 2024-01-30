#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=missing_data # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=general_acc_%j.out
#SBATCH --ntasks=1                 # Run on a single Node
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=100gb                     # Job memory request
#not SBATCH  -x linux[41-60],gpu-compute[1-7]
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

import os
import sys
import re
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn import metrics
import fastsparsegams
import matplotlib.pyplot as plt
from mice_utils import return_imputation, binarize_according_to_train, eval_model, get_train_test_binarized, binarize_and_augment, errors

num_quantiles = 8
dataset = 'BREAST_CANCER'
train_miss = 0
test_miss = train_miss

### Immutable ###
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]
holdout_set = 0
val_set = 0
s_size=100
###################

### basic hyperparameter processing ###

miss_str = ''
if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
    miss_str = f'_train_missing_{train_miss}_test_missing_{test_miss}'

def path_to_imputed(dataset, holdout_set, val_set, imputation):
    if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
        return  f'../../handling_missing_data/IMPUTED_DATA/{dataset}/MICE/train_per_{train_miss}/test_per_{test_miss}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'
    return f'../../handling_missing_data/IMPUTED_DATA/{dataset}/MICE/train_per_0/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'

def prefix_pre_imputed(dataset):
    standard_prefix = '../../handling_missing_data/DATA'
    if 'SYNTHETIC' in dataset:
        standard_prefix = f'{standard_prefix}/SYNTHETIC_MAR' if 'MAR' in dataset else f'{standard_prefix}/SYNTHETIC'
        if 'CATEGORICAL' in dataset: 
            overall_dataset = 'SYNTHETIC_CATEGORICAL'
        else: 
            overall_dataset = 'SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE' 
        if bool(re.search(r'\d', dataset)): # if dataset contains a number, use our system for MAR missing datasets
            missing_prop = int(dataset[dataset.rfind('_')+1:])/100
            return  f'{standard_prefix}/{overall_dataset}/{missing_prop}/'
        return f'{standard_prefix}/{overall_dataset}/'
    if bool(re.search(r'\d', dataset)): # if dataset contains a number, use our system for MAR missing datasets
        missing_prop = int(dataset[dataset.rfind('_')+1:])/100
        overall_dataset = dataset[:dataset.rfind('_')]
        return  f'{standard_prefix}/{overall_dataset}/{missing_prop}/'
    return f'{standard_prefix}/{dataset}/'

###############################################

#############################################
### measures across trials, for plotting: ###
#############################################

#sparsity for each lambda, holdout, and method
sparsity_aug = np.zeros((len(lambda_grid[0])))
sparsity_indicator = np.zeros((len(lambda_grid[0])))
sparsity_no_missing = np.zeros((len(lambda_grid[0])))


train = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_train_{val_set}{miss_str}.csv')
val = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_val_{val_set}{miss_str}.csv')
test = pd.read_csv(prefix_pre_imputed(dataset) + f'holdout_{holdout_set}{miss_str}.csv')
label = train.columns[-1]
predictors = train.columns[:-1]

#variables for prob preds of indicator methods, for each lambda
train_probs_aug = np.zeros((len(lambda_grid[0]), train.shape[0] + val.shape[0]))
train_probs_indicator = np.zeros((len(lambda_grid[0]), train.shape[0] + val.shape[0]))
train_probs_no_missing = np.zeros((len(lambda_grid[0]), train.shape[0] + val.shape[0]))
test_probs_aug = np.zeros((len(lambda_grid[0]), test.shape[0]))
test_probs_indicator = np.zeros((len(lambda_grid[0]), test.shape[0]))
test_probs_no_missing = np.zeros((len(lambda_grid[0]), test.shape[0]))
######################################
### Missingness Indicator Approach ###
######################################

(train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
y_train_no, y_train_ind, y_train_aug, 
y_test_no, y_test_ind, y_test_aug) = binarize_and_augment(
    pd.concat([train, val]), test, np.linspace(0, 1, num_quantiles + 2)[1:-1], label
    )

model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                            num_lambda=None, num_gamma=None, max_support_size=s_size)
(train_probs_no_missing, _, 
 test_probs_no_missing , _, sparsity_no_missing ) = eval_model(
    model_no, train_no, test_no, y_train_no, y_test_no, lambda_grid[0],
    lambda y, probs: np.exp(np.log(probs/(1 - probs))*-1/2*(2*y-1)).mean()
    )

model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                            num_lambda=None, num_gamma=None, max_support_size=s_size)
(train_probs_indicator, _,
    test_probs_indicator, _, sparsity_indicator) = eval_model(
    model_ind, train_ind, test_ind, y_train_ind, y_test_ind, lambda_grid[0], 
    lambda y, probs: np.exp(np.log(probs/(1 - probs))*-1/2*(2*y-1)).mean()
    )

model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                            lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=s_size)
(train_probs_aug , _,
    test_probs_aug , _, sparsity_aug ) = eval_model(
    model_aug, train_aug, test_aug, y_train_aug, y_test_aug, lambda_grid[0], 
    lambda y, probs: np.exp(np.log(probs/(1 - probs))*-1/2*(2*y-1)).mean()
    )

#save data to csv: 

results_path = f'debug/{dataset}'
if train_miss != 0 or test_miss != 0: 
    results_path = f'{results_path}/train_{train_miss}/test_{test_miss}'
if not os.path.exists(results_path):
    os.makedirs(results_path)

np.savetxt(f'{results_path}/train_probs_aug.csv', train_probs_aug)
np.savetxt(f'{results_path}/train_probs_indicator.csv', train_probs_indicator)
np.savetxt(f'{results_path}/train_probs_no_missing.csv', train_probs_no_missing)
np.savetxt(f'{results_path}/test_probs_aug.csv', test_probs_aug)
np.savetxt(f'{results_path}/test_probs_indicator.csv', test_probs_indicator)
np.savetxt(f'{results_path}/test_probs_no_missing.csv', test_probs_no_missing)
np.savetxt(f'{results_path}/nllambda.csv', -np.log(lambda_grid[0]))

np.savetxt(f'{results_path}/train_labels.csv', pd.concat([train, val])[label].to_numpy())
np.savetxt(f'{results_path}/test_labels.csv', test[label].to_numpy())

np.savetxt(f'{results_path}/sparsity_aug.csv', sparsity_aug)
np.savetxt(f'{results_path}/sparsity_indicator.csv', sparsity_indicator)
np.savetxt(f'{results_path}/sparsity_no_missing.csv',sparsity_no_missing)


#can also try pickle file of all hyperparameters, and save to a folder with corresponding hash


print('successfully finished execution')
