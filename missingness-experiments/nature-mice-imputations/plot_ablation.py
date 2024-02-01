#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=ablationplot # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=ablationplot_%j.out
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

from cycler import cycler
# plt.style.use('seaborn-v0_8-notebook')
#['#1b9e77','#7570b3', '#8c564b']
#'#a6611a',['#018571','#dfc27d', '#80cdc1', '#a6611a']
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1b9e77','#7570b3', '#8c564b'] )#['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.rcParams.update({'font.size': 16})

dataset = 'BREAST_CANCER'
metric = 'auc' if 'BREAST_CANCER' in dataset else 'acc'
miss_handlings = ['aug', 'indicator', 'no_missing']
ylim = (0.69, 0.77) if metric == 'auc' else (0.7, 0.735)

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


ablations = ['mean/', 'distinct/' if 'FICO' in dataset else '']#, 'MICE/']
ABLATION_NAME = {
    'distinct/': 'No Imputation', # (missingness imputed as false for all thresholds)
    '': 'No Imputation',
    'median/': 'Median Imputation', 
    'MICE/': 'MICE',
    'mean/': 'Mean Imputation'
}

fig_dir = f'./figs/ablation/{dataset}/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def plot_with_break(ax, ax2, vals, ensemble_perf, dataset, metric, is_train=False, y_label = True, legend = False, y_lim = (0.5, 1.0)): 

        # plot the same data on both axes
        train_or_test_str = 'Train' if is_train else 'Test'
        uncertainty_bands_subplot(vals[0][0], vals[0][1], None, ax) 
        uncertainty_bands_subplot(vals[1][0], vals[1][1], None, ax)
        ax.set_xlabel('# Nonzero Coefficients')
        if y_label:
                ax.set_ylabel(f'{train_or_test_str} {METRIC_NAME[metric]}')

        ax2.hlines(ensemble_perf, 0,
                1000, linestyles='dashed',
                label= 'MICE' if legend else None,#'mean performance, ensemble of 10 MICE imputations',
                color='grey') #TODO: add error bars? 
        uncertainty_bands_subplot(vals[0][0], vals[0][1], ABLATION_NAME[ablations[0]] if legend else None, ax2) 
        uncertainty_bands_subplot(vals[1][0], vals[1][1], ABLATION_NAME[ablations[1]] if legend else None, ax2)
        ax2.set_xlabel('*')#'Non-interpretable \n models')
        ax2.set_xticks([])

        ax.set_xlim(0, 52.5)
        ax2.set_xlim(200, 201)
        ax.set_ylim(*y_lim)
        ax2.set_ylim(*y_lim)

        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(labelright=False)
        ax2.yaxis.tick_right()

        # ax2.legend()

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-(1/3)*d, 1+(1/3)*d), (-d, d), **kwargs)
        ax.plot((1-(1/3)*d, 1+1/3*d), (1-d, 1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)

fig = plt.figure(layout='constrained', figsize=(7, 14))
figs = fig.subfigures(len(miss_handlings), wspace=.07, hspace=.1)

for plot_idx, miss_handling in enumerate(miss_handlings): 
    #load data files from csv:
    vals = [] #of form sparsity, metric
    for ablation in ablations: 
        train_metric = np.loadtxt(f'experiment_data/{dataset}/{ablation}train_{metric}_{miss_handling}.csv')
        test_metric = np.loadtxt(f'experiment_data/{dataset}/{ablation}test_{metric}_{miss_handling}.csv')
        sparsity = np.loadtxt(f'experiment_data/{dataset}/{ablation}sparsity_{miss_handling}.csv')
        
        no_timeouts = (train_metric> 0).all(axis=0)
        sparsity = sparsity[:, no_timeouts]
        train_metric = train_metric[:, no_timeouts]
        test_metric = test_metric[:, no_timeouts]
        
        vals.append((sparsity, test_metric))

    # # placeholder: needs to be custom to the miss handling param
    if dataset == 'BREAST_CANCER': 
        imputation_ensemble_test_auc = np.loadtxt(f'experiment_data/ablation/{dataset}/MICE/imputation_ensemble_test_{metric}_{miss_handling}.csv')
        imputation_ensemble_test_auc = imputation_ensemble_test_auc[imputation_ensemble_test_auc > 0]
        ensemble_perf = imputation_ensemble_test_auc.mean()
        print(ensemble_perf)
    else: 
        imputation_ensemble_test_auc = 0
        SKIPS = {
            'FICO': [],
            'BREAST_CANCER': [(6, 3), (9, 2)]
        }
        for holdout in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: 
            for val in [0, 1, 2, 3, 4]: 
                if (holdout, val) in SKIPS[dataset]: 
                    continue
                imputation_ensemble_test_auc += np.loadtxt(f'experiment_data/ablation/{dataset}/MICE_{holdout}_{val}/imputation_ensemble_test_{metric}_{miss_handling}.csv')
        ensemble_perf = (imputation_ensemble_test_auc/(50 - len(SKIPS[dataset])))
        print(ensemble_perf)

    axs = figs[plot_idx].subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    figs[plot_idx].suptitle(f'{MISS_HANDLING_NAME[miss_handling]}')
    plot_with_break(axs[0], axs[1], vals, ensemble_perf = ensemble_perf, dataset=dataset, metric=metric, legend=(plot_idx==2), y_lim=ylim)
fig.suptitle(f'Test {METRIC_NAME[metric]} vs # Nonzero Coefficients \n for {dataset} dataset...')
# fig.legend(loc='lower right')
fig.legend(loc='lower center', fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.05))

fig.savefig(f'./figs/{dataset}/all_ablations_{metric}.pdf', bbox_inches='tight')

print('Successful execution')
