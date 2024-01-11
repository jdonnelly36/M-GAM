#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=missing_data # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=missing_data_%j.out
#SBATCH --ntasks=1                 # Run on a single Node
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=100gb                     # Job memory request
#not SBATCH  -x linux[41-60],gpu-compute[1-7]
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn import metrics
import fastsparsegams
import matplotlib.pyplot as plt
from mice_utils import return_imputation, binarize_according_to_train, eval_model, get_train_test_binarized, binarize_and_augment, errors

dataset = 'FICO'#'BREAST_CANCER'

#load data files from csv: 

train_auc_aug = np.loadtxt(f'experiment_data/{dataset}/train_auc_aug.csv')
train_auc_indicator = np.loadtxt(f'experiment_data/{dataset}/train_auc_indicator.csv')
train_auc_no_missing = np.loadtxt(f'experiment_data/{dataset}/train_auc_no_missing.csv')
test_auc_aug = np.loadtxt(f'experiment_data/{dataset}/test_auc_aug.csv')
test_auc_indicator = np.loadtxt(f'experiment_data/{dataset}/test_auc_indicator.csv')
test_auc_no_missing = np.loadtxt(f'experiment_data/{dataset}/test_auc_no_missing.csv')
imputation_ensemble_train_auc = np.loadtxt(f'experiment_data/{dataset}/imputation_ensemble_train_auc.csv')
imputation_ensemble_test_auc = np.loadtxt(f'experiment_data/{dataset}/imputation_ensemble_test_auc.csv')
nllambda = np.loadtxt(f'experiment_data/{dataset}/nllambda.csv')

sparsity_aug = np.loadtxt(f'experiment_data/{dataset}/sparsity_aug.csv')
sparsity_indicator = np.loadtxt(f'experiment_data/{dataset}/sparsity_indicator.csv')
sparsity_no_missing = np.loadtxt(f'experiment_data/{dataset}/sparsity_no_missing.csv')

no_timeouts_aug = (train_auc_aug > 0).all(axis=0)
sparsity_aug = sparsity_aug[:, no_timeouts_aug]
train_auc_aug = train_auc_aug[:, no_timeouts_aug]
test_auc_aug = test_auc_aug[:, no_timeouts_aug]

plt.title(f'Train AUC vs # Nonzero Coefficients \n for {dataset} dataset')
plt.hlines(imputation_ensemble_train_auc.mean(), 0,
           max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
           label='mean performance, ensemble of 10 MICE imputations') #TODO: add error bars? 
plt.errorbar(sparsity_no_missing.mean(axis=0), train_auc_no_missing.mean(axis=0), 
             yerr = errors(train_auc_no_missing, axis=0),
             xerr = errors(sparsity_no_missing, axis=0),
             label='No missingness handling', fmt='.', lw=1)
plt.errorbar(sparsity_aug.mean(axis=0), train_auc_aug.mean(axis=0), 
             yerr = errors(train_auc_aug, axis=0), 
             xerr = errors(sparsity_aug, axis=0),
             fmt = '.', lw=1,
             label='Missingness with interactions')
plt.errorbar(sparsity_indicator.mean(axis=0), train_auc_indicator.mean(axis=0), 
             yerr = errors(train_auc_indicator, axis=0), 
             xerr = errors(sparsity_indicator, axis=0),
             fmt = '.', lw=1,
             label='Missingness indicators')
plt.xlabel('# Nonzero Coefficients')
plt.ylabel('Train AUC')
plt.legend()
# plt.ylim(0.7, 0.86)
plt.savefig(f'./figs/{dataset}/mice_slurm_train_auc.png')

plt.clf()

plt.title(f'Test AUC vs # Nonzero Coefficients \n for {dataset} dataset')
plt.hlines(imputation_ensemble_test_auc.mean(), 0,
           max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
           label='mean performance, ensemble of 10 MICE imputations') #TODO: add error bars? 
plt.errorbar(sparsity_no_missing.mean(axis=0), test_auc_no_missing.mean(axis=0), 
             yerr = errors(test_auc_no_missing, axis=0),
             xerr = errors(sparsity_no_missing, axis=0),
             label='No missingness handling', fmt='.', lw=1)
plt.errorbar(sparsity_aug.mean(axis=0), test_auc_aug.mean(axis=0), 
             yerr = errors(test_auc_aug, axis=0), 
             xerr = errors(sparsity_aug, axis=0),
             fmt = '.', lw=1,
             label='Missingness with interactions')
plt.errorbar(sparsity_indicator.mean(axis=0), test_auc_indicator.mean(axis=0), 
             yerr = errors(test_auc_indicator, axis=0), 
             xerr = errors(sparsity_indicator, axis=0),
             fmt = '.', lw=1,
             label='Missingness indicators')
plt.xlabel('# Nonzero Coefficients')
plt.ylabel('Test AUC')
plt.legend()
# plt.ylim(0.7, 0.86)
plt.savefig(f'./figs/{dataset}/mice_slurm_test_auc.png')

print('successfully finished execution')
