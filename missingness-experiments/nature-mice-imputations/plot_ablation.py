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
from mice_utils import errors, uncertainty_bands, uncertainty_bands_subplot

dataset = 'FICO'
metric = 'acc'
miss_handlings = ['aug', 'indicator', 'no_missing']

METRIC_NAME = {
    'acc': 'Accuracy',
    'auc': 'AUC',
    'auprc': 'AUPRC',
    'loss': 'Exponential Loss'
}

MISS_HANDLING_NAME = {
    'aug': 'with Missingness Interactions',
    'indicator': 'with Missingness Indicators',
    'no_missing': 'without Missingness Indicators'
}


ablations = ['', 'median/', 'MICE/', 'mean/']
ABLATION_NAME = {
    '': 'No Imputation', # (missingness imputed as false for all thresholds)
    'median/': 'Median Imputation', 
    'MICE/': 'MICE',
    'mean/': 'Mean Imputation'
}

fig, axs = plt.subplots(len(miss_handlings), figsize=(7, 14))

for plot_idx, miss_handling in enumerate(miss_handlings): 
    #load data files from csv:
    train_metrics = [] 
    test_metrics = []
    sparsities = []
    for ablation in ablations: 
        train_metric = np.loadtxt(f'experiment_data/{dataset}/{ablation}train_{metric}_{miss_handling}.csv')
        test_metric = np.loadtxt(f'experiment_data/{dataset}/{ablation}test_{metric}_{miss_handling}.csv')
        sparsity = np.loadtxt(f'experiment_data/{dataset}/{ablation}sparsity_{miss_handling}.csv')
        
        no_timeouts = (train_metric> 0).all(axis=0)
        sparsity = sparsity[:, no_timeouts]
        train_metric = train_metric[:, no_timeouts]
        test_metric = test_metric[:, no_timeouts]
        
        train_metrics.append(train_metric)
        test_metrics.append(test_metric)
        sparsities.append(sparsity)

    axs[plot_idx].set_title(f'{MISS_HANDLING_NAME[miss_handling]}')
    for i in range(len(ablations)): 
        uncertainty_bands_subplot(sparsities[i], test_metrics[i], ABLATION_NAME[ablations[i]] if plot_idx == 0 else None, axs[plot_idx])
    axs[plot_idx].set_xlabel('# Nonzero Coefficients')
    axs[plot_idx].set_ylabel(f'Test {METRIC_NAME[metric]}')
fig.suptitle(f'Test {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset...')
fig.legend(loc='lower right')
fig.tight_layout()

fig.savefig(f'./figs/{dataset}/all_ablations_{metric}.png')

print('Successful execution')
