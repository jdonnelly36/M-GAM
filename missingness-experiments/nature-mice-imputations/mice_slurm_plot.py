#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=plot # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=logs/plot_%j.out
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
from mice_utils import errors, uncertainty_bands, uncertainty_bands_subplot, uncertainty_bands_subplot_mice

from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#808080'])
# plt.rcParams.update({'font.size': 16})

dataset = 'FICO'
metric = 'acc'
filetype = 'png'

overall_mi_intercept = False
overall_mi_ixn = False
specific_mi_intercept = True
specific_mi_ixn = True

mgam_imputer = None
mice_augmentation_level = 1 # 0 for no missingness features, 1 for indicators, 2 for interactions

sparsity_metric = 'default'
baseline_imputer = 'MICE'

DATASET_NAME = {
    'FICO': 'FICO',
    'BREAST_CANCER': 'BREAST_CANCER', 
    'FICO_MAR': 'Synthetic',
    'FICO_MAR_25': 'Synthetic Missingness @ rate=0.25, FICO',
    'FICO_MAR_50': 'Synthetic Missingness @ rate=0.5, FICO',
    'BREAST_CANCER_MAR': 'BREAST_CANCER + Synthetic Missingness',
    'BREAST_CANCER_MAR_25': 'Synthetic Missingness 0.25, BREAST_CANCER',
    'BREAST_CANCER_MAR_50': 'Synthetic Missingness 0.5, BREAST_CANCER', 
    'SYNTHETIC_MAR': 'Synthetic MAR',
    'SYNTHETIC_MAR_50': 'Synthetic data, MAR missingness 0.5',
    'SYNTHETIC_MAR_25': 'Synthetic data, MAR missingness 0.5',
    'SYNTHETIC_CATEGORICAL_MAR': 'Synthetic Categorical MAR',
    'SYNTHETIC_CATEGORICAL_MAR_50': 'Synthetic Categorical data, MAR missingness 0.5',
    'SYNTHETIC_CATEGORICAL_MAR_25': 'Synthetic Categorical data, MAR missingness 0.5',
}

METRIC_NAME = {
    'acc': 'Accuracy',
    'auc': 'AUC',
    'auprc': 'AUPRC',
    'loss': 'Exponential Loss'
}

s_size_folder = '100'
train_miss = 0
test_miss = train_miss
s_size_cutoff = 52.5
num_quantiles = 8
quantile_addition = ''

#load data files from csv: 

res_dir = f'experiment_data/{dataset}'
res_dir = f'{res_dir}/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}'

if train_miss != 0 or test_miss != 0: 
    res_dir = f'{res_dir}/train_{train_miss}/test_{test_miss}'
if num_quantiles != 8: 
     res_dir = f'{res_dir}/q{num_quantiles}'

mice_res_dir = f'{res_dir}/{s_size_folder}'+(
     f'/{mice_augmentation_level}' if mice_augmentation_level > 0 else '')+(
        f'/{baseline_imputer}' if baseline_imputer != 'MICE' else ''
     )

if sparsity_metric != 'default': 
    res_dir = f'{res_dir}/sparsity_{sparsity_metric}'
if mgam_imputer != None: 
    res_dir = f'{res_dir}/imputer_{mgam_imputer}'

train_auc_aug = np.loadtxt(f'{res_dir}/train_{metric}_aug.csv')
train_auc_indicator = np.loadtxt(f'{res_dir}/train_{metric}_indicator.csv')
train_auc_no_missing = np.loadtxt(f'{res_dir}/train_{metric}_no_missing.csv')
test_auc_aug = np.loadtxt(f'{res_dir}/test_{metric}_aug.csv')
test_auc_indicator = np.loadtxt(f'{res_dir}/test_{metric}_indicator.csv')
test_auc_no_missing = np.loadtxt(f'{res_dir}/test_{metric}_no_missing.csv')
imputation_ensemble_train_auc = np.loadtxt(f'{mice_res_dir}/imputation_ensemble_train_{metric}.csv')
imputation_ensemble_test_auc = np.loadtxt(f'{mice_res_dir}/imputation_ensemble_test_{metric}.csv')
nllambda = np.loadtxt(f'{res_dir}/nllambda.csv')

sparsity_aug = np.loadtxt(f'{res_dir}/sparsity_aug.csv')
sparsity_indicator = np.loadtxt(f'{res_dir}/sparsity_indicator.csv')
sparsity_no_missing = np.loadtxt(f'{res_dir}/sparsity_no_missing.csv')

#filter out any skipped (0 valued) runs of imputation (TODO: verify this has an effect for breca)
# imputation_ensemble_test_auc = imputation_ensemble_test_auc[imputation_ensemble_test_auc > 0]
imputation_ensemble_train_auc = imputation_ensemble_train_auc[imputation_ensemble_train_auc > 0]
imputation_ensemble_test_auc[imputation_ensemble_test_auc == 0] = np.nan

no_timeouts_aug = (train_auc_aug > 0).all(axis=0)
sparsity_aug = sparsity_aug[:, no_timeouts_aug]
train_auc_aug = train_auc_aug[:, no_timeouts_aug]
test_auc_aug = test_auc_aug[:, no_timeouts_aug]

no_timeouts_indicator = (train_auc_indicator > 0).all(axis=0)
sparsity_indicator = sparsity_indicator[:, no_timeouts_indicator]
train_auc_indicator = train_auc_indicator[:, no_timeouts_indicator]
test_auc_indicator = test_auc_indicator[:, no_timeouts_indicator]

# no_timeouts_no_missing = (train_auc_no_missing > 0).all(axis=0)
# sparsity_no_missing = sparsity_no_missing[:, no_timeouts_no_missing]
# train_auc_no_missing = train_auc_no_missing[:, no_timeouts_no_missing]
# test_auc_no_missing = test_auc_no_missing[:, no_timeouts_no_missing]

fig_dir = f'./figs/{dataset}/'
fig_dir = f'{fig_dir}/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}/'
fig_dir = f'{fig_dir}{s_size_folder}/'
fig_dir += f'{mice_augmentation_level}/' if mice_augmentation_level > 0 else ''
fig_dir += f'{baseline_imputer}/' if baseline_imputer != 'MICE' else ''
if sparsity_metric != 'default': 
    fig_dir = f'{fig_dir}/sparsity_{sparsity_metric}'
if mgam_imputer != None: 
    fig_dir = f'{fig_dir}/imputer_{mgam_imputer}'

if train_miss != 0 or test_miss != 0: 
    fig_dir = f'{fig_dir}train_{train_miss}/test_{test_miss}/'
if num_quantiles != 8: 
     fig_dir = f'{fig_dir}q{num_quantiles}/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


'''
Plotting code: 
'''
"""
Plot with an axis break to better indicate uninterpretability.
Makes use of Stack Overflow Post 32185411
"""


TRAIN_VALS = [(sparsity_no_missing, train_auc_no_missing), 
              (sparsity_aug, train_auc_aug), 
              (sparsity_indicator, train_auc_indicator)]
TEST_VALS = [(sparsity_no_missing, test_auc_no_missing), 
             (sparsity_aug, test_auc_aug), 
             (sparsity_indicator, test_auc_indicator)]

def plot_with_break(is_train = True): 
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})

        # plot the same data on both axes
        train_or_test_str = 'Train' if is_train else 'Test'
        vals = TRAIN_VALS if is_train else TEST_VALS

        f.suptitle(f'{train_or_test_str} {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset{quantile_addition}')
        # ax.hlines(imputation_ensemble_train_auc.mean(), 0,
        #            max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
        #            label='mean performance, ensemble of 10 MICE imputations',
        #            color='grey') #TODO: add error bars? 
        # uncertainty_bands_subplot(vals[0][0], vals[0][1], 'No missingness handling', ax)
        # uncertainty_bands_subplot(vals[1][0], vals[1][1], 'Missingness with interactions', ax)
        uncertainty_bands_subplot(vals[2][0], vals[2][1], 'Missingness indicators', ax) 
        uncertainty_bands_subplot(vals[1][0], vals[1][1], 'Missingness with interactions', ax)
        ax.set_xlabel('# Nonzero Coefficients')
        ax.set_ylabel(f'{train_or_test_str} {METRIC_NAME[metric]}')
        # ax.legend()

        # ax2.set_title(f'Train {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset')
        ax2.hlines(np.nanmean(imputation_ensemble_train_auc) if is_train else np.nanmean(imputation_ensemble_test_auc), 0,
                1000, linestyles='dashed',
                label= 'MICE',#'mean performance, ensemble of 10 MICE imputations',
                color='grey') #TODO: add error bars? 
        # uncertainty_bands_subplot(sparsity_no_missing, train_auc_no_missing, 'No Missingness \n Handling', ax2)
        uncertainty_bands_subplot(sparsity_indicator, train_auc_indicator, 'Indicators', ax2) 
        uncertainty_bands_subplot(sparsity_aug, train_auc_aug, 'Interactions', ax2)
        uncertainty_bands_subplot_mice(imputation_ensemble_test_auc, None, ax2)
        ax2.set_xlabel('Non-interpretable \n models')
        ax2.set_xticks([]) # need to replace xticks with infinity


        plt.legend()

        ax.set_xlim(0, s_size_cutoff)
        ax2.set_xlim(200, 201)
        # ax.set_ylim(0.5, 2.2)
        # ax2.set_ylim(0.5, 2.2)

        # hide the spines between ax and ax2
        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(labelright=False)
        ax2.yaxis.tick_right()
        # hide the spines between ax and ax2


        # In axes coordinates, which are always
        # between 0-1, spine endpoints are at these locations (0, 0), (0, 1),
        # (1, 0), and (1, 1).  Thus, we just need to put the diagonals in the
        # appropriate corners of each of our axes, and so long as we use the
        # right transform and disable clipping.

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-(1/3)*d, 1+(1/3)*d), (-d, d), **kwargs)
        ax.plot((1-(1/3)*d, 1+1/3*d), (1-d, 1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)

        f.tight_layout()
        train_or_test_str = 'train' if is_train else 'test'
        plt.savefig(f'{fig_dir}mice_slurm_{train_or_test_str}_{metric}_with_break.{filetype}')
# plot_with_break(True)
plot_with_break(False)

plt.clf()

plt.title(f'Train {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset')
plt.hlines(imputation_ensemble_train_auc.mean(), 0,
           max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
           label='mean performance, ensemble of 10 MICE imputations',
           color='grey') #TODO: add error bars? 
# uncertainty_bands(sparsity_no_missing, train_auc_no_missing, 'No missingness handling')
# uncertainty_bands(sparsity_aug, train_auc_aug, 'Missingness with interactions')
uncertainty_bands(sparsity_indicator, train_auc_indicator, 'Missingness indicators') 
uncertainty_bands(sparsity_aug, train_auc_aug, 'Missingness with interactions')
plt.xlabel('# Nonzero Coefficients')
plt.ylabel(f'Train {METRIC_NAME[metric]}')
plt.legend()
plt.xlim(0,s_size_cutoff)
# plt.ylim(0.7, 0.86)
plt.savefig(f'{fig_dir}mice_slurm_train_{metric}.{filetype}')

plt.clf()

plt.title(f'Test {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {DATASET_NAME[dataset]} Dataset')
plt.hlines(np.nanmean(imputation_ensemble_test_auc), 0,
           max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
           label='mean performance, ensemble of 10 MICE imputations', 
           color='grey') #TODO: add error bars? 
# uncertainty_bands(sparsity_no_missing, test_auc_no_missing, 'No missingness handling')
# uncertainty_bands(sparsity_aug, test_auc_aug, 'Missingness with interactions')
uncertainty_bands(sparsity_indicator, test_auc_indicator, 'Missingness indicators')
uncertainty_bands(sparsity_aug, test_auc_aug, 'Missingness with interactions')
plt.xlabel('# Nonzero Coefficients')
plt.ylabel(f'Test {METRIC_NAME[metric]}')
plt.legend()
plt.xlim(0,52.5)
# plt.ylim(0.9, 2.2)
plt.savefig(f'{fig_dir}mice_slurm_test_{metric}.{filetype}')

print('successfully finished execution')
