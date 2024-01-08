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
#load data files from csv: 

train_auc_aug = np.loadtxt('experiment_data/train_auc_aug.csv')
train_auc_indicator = np.loadtxt('experiment_data/train_auc_indicator.csv')
train_auc_no_missing = np.loadtxt('experiment_data/train_auc_no_missing.csv')
test_auc_aug = np.loadtxt('experiment_data/test_auc_aug.csv')
test_auc_indicator = np.loadtxt('experiment_data/test_auc_indicator.csv')
test_auc_no_missing = np.loadtxt('experiment_data/test_auc_no_missing.csv')
imputation_ensemble_train_auc = np.loadtxt('experiment_data/imputation_ensemble_train_auc.csv')
imputation_ensemble_test_auc = np.loadtxt('experiment_data/imputation_ensemble_test_auc.csv')
nllambda = np.loadtxt('experiment_data/nllambda.csv')

no_timeouts_aug = (train_auc_aug > 0).all(axis=0)


plt.title('Train AUC vs Negative Log Lambda_0 \n for BRECA dataset')
plt.errorbar(nllambda,
             imputation_ensemble_train_auc.mean()*np.ones(len(nllambda)),
             yerr = imputation_ensemble_train_auc.std()/len(imputation_ensemble_train_auc), capsize=3,
             label='ensemble of 10 MICE imputations')
plt.errorbar(nllambda, train_auc_no_missing.mean(axis=0), 
             yerr = errors(train_auc_no_missing, axis=0), capsize=3,
             label='No missingness handling (single run)')
plt.errorbar(nllambda, train_auc_indicator.mean(axis=0), 
             yerr = errors(train_auc_indicator, axis=0), capsize=3,
             label='Missingness indicators (single run)')
plt.errorbar(nllambda[no_timeouts_aug], train_auc_aug[:, no_timeouts_aug].mean(axis=0), 
             yerr = errors(train_auc_aug[:, no_timeouts_aug], axis=0), capsize=3,
             label='Missingness with interactions (single run)')
plt.xlabel('Negative Log Lambda_0')
plt.ylabel('Train AUC')
plt.legend()
plt.ylim(0.7, 0.86)
plt.savefig('./figs/mice_slurm_train_AUC.png')

plt.clf()

plt.title('Test AUC vs Negative Log Lambda_0 \n for BRECA dataset')
plt.errorbar(nllambda, 
             imputation_ensemble_test_auc.mean()*np.ones(len(nllambda)), 
             yerr = imputation_ensemble_test_auc.std()/len(imputation_ensemble_test_auc), capsize=3,
             label='ensemble of 10 MICE imputations')
plt.errorbar(nllambda, test_auc_no_missing.mean(axis=0), 
             yerr = errors(test_auc_no_missing, axis=0), capsize=3,
             label='No missingness handling (single run)')
plt.errorbar(nllambda, test_auc_indicator.mean(axis=0), 
             yerr = errors(test_auc_indicator, axis=0), capsize=3,
             label='Missingness indicators (single run)')
plt.errorbar(nllambda[no_timeouts_aug], 
             test_auc_aug[:, no_timeouts_aug].mean(axis=0), 
             yerr = errors(test_auc_aug[:, no_timeouts_aug], axis=0), capsize=3,
             label='Missingness with interactions (single run)')
plt.xlabel('Negative Log Lambda_0')
plt.ylabel('Test AUC')
plt.legend()
plt.ylim(0.7, 0.86)
plt.savefig('./figs/mice_slurm_test_AUC.png')

print('successfully finished execution')
