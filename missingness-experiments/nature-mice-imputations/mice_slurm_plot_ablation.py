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
from mice_utils import errors, uncertainty_bands

# plt.style.use('seaborn-v0_8')

dataset = 'FICO'
metric = 'acc'

METRIC_NAME = {
    'acc': 'Accuracy',
    'auc': 'AUC',
    'auprc': 'AUPRC',
    'loss': 'Exponential Loss'
}

ablation = 'MICE'

#load data files from csv: 

train_auc_aug = np.loadtxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_aug.csv')
train_auc_indicator = np.loadtxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_indicator.csv')
train_auc_no_missing = np.loadtxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_no_missing.csv')
test_auc_aug = np.loadtxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_aug.csv')
test_auc_indicator = np.loadtxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_indicator.csv')
test_auc_no_missing = np.loadtxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_no_missing.csv')
imputation_ensemble_train_auc = np.loadtxt(f'experiment_data/{dataset}/imputation_ensemble_train_{metric}.csv')
imputation_ensemble_test_auc = np.loadtxt(f'experiment_data/{dataset}/imputation_ensemble_test_{metric}.csv')
nllambda = np.loadtxt(f'experiment_data/{dataset}/{ablation}/nllambda.csv')

sparsity_aug = np.loadtxt(f'experiment_data/{dataset}/{ablation}/sparsity_aug.csv')
sparsity_indicator = np.loadtxt(f'experiment_data/{dataset}/{ablation}/sparsity_indicator.csv')
sparsity_no_missing = np.loadtxt(f'experiment_data/{dataset}/{ablation}/sparsity_no_missing.csv')

no_timeouts_aug = (train_auc_aug > 0).all(axis=0)
sparsity_aug = sparsity_aug[:, no_timeouts_aug]
train_auc_aug = train_auc_aug[:, no_timeouts_aug]
test_auc_aug = test_auc_aug[:, no_timeouts_aug]

no_timeouts_indicator = (train_auc_indicator > 0).all(axis=0)
sparsity_indicator = sparsity_indicator[:, no_timeouts_indicator]
train_auc_indicator = train_auc_indicator[:, no_timeouts_indicator]
test_auc_indicator = test_auc_indicator[:, no_timeouts_indicator]

no_timeouts_no_missing = (train_auc_no_missing > 0).all(axis=0)
sparsity_no_missing = sparsity_no_missing[:, no_timeouts_no_missing]
train_auc_no_missing = train_auc_no_missing[:, no_timeouts_no_missing]
test_auc_no_missing = test_auc_no_missing[:, no_timeouts_no_missing]

plt.title(f'Train {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset, with {ablation} imputation')
# plt.hlines(imputation_ensemble_train_auc.mean(), 0,
#            max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
#            label='mean performance, ensemble of 10 MICE imputations',
#            color='grey') #TODO: add error bars? 
uncertainty_bands(sparsity_no_missing, train_auc_no_missing, f'Imputation via {ablation}')
uncertainty_bands(sparsity_aug, train_auc_aug, f'Missingness with interactions + {ablation}')
uncertainty_bands(sparsity_indicator, train_auc_indicator, f'Missingness indicators + {ablation}') 
plt.xlabel('# Nonzero Coefficients')
plt.ylabel(f'Train {METRIC_NAME[metric]}')
plt.legend()
# plt.ylim(0.7, 0.86)
plt.savefig(f'./figs/{dataset}/{ablation}/mice_slurm_train_{metric}.png')

plt.clf()

plt.title(f'Test {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset, with {ablation} imputation')
# plt.hlines(imputation_ensemble_test_auc.mean(), 0,
#            max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
#            label='mean performance, ensemble of 10 MICE imputations', 
#            color='grey') #TODO: add error bars? 
uncertainty_bands(sparsity_no_missing, test_auc_no_missing, f'Imputation via {ablation}')
uncertainty_bands(sparsity_aug, test_auc_aug, f'Missingness with interactions + {ablation}')
uncertainty_bands(sparsity_indicator, test_auc_indicator, f'Missingness indicators + {ablation}')
plt.xlabel('# Nonzero Coefficients')
plt.ylabel(f'Test {METRIC_NAME[metric]}')
plt.legend()
# plt.ylim(0.7, 0.86)
plt.savefig(f'./figs/{dataset}/{ablation}/mice_slurm_test_{metric}.png')

print('successfully finished execution')
