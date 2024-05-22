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
import argparse
# plt.rcParams["axes.prop_cycle"] = cycler('color', ['#92c5de','#0571b0', '#ca0020','#f4a582'])#['#1f77b4', '#ff7f0e', '#808080'])
# plt.rcParams["axes.prop_cycle"] = cycler('color', ['#762a83','#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b','#1b7837'])
# plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.size': 24,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsmath}'})

parser = argparse.ArgumentParser(description='Plotting playground')
parser.add_argument('-d', '--datasets', type=str, nargs='+', default=['FICO'], help='List of dataset names')
parser.add_argument('--distinct', action='store_true', help='Use distinctness measures')
args = parser.parse_args()

datasets = args.datasets
distinct = 'true_' if args.distinct else ''
print(distinct)
filetype = 'pdf'
is_train = False

overall_mi_intercept = False
overall_mi_ixn = False
specific_mi_intercept = True
specific_mi_ixn = True

mgam_imputer = None
mice_augmentation_level = 0 # 0 for no missingness features, 1 for indicators, 2 for interactions

sparsity_metric = 'default'# 'default', 'num_variables'
baseline_imputers = ['MICE', 'MIWAE', 'MissForest']#['MICE', 'Mean', 'MIWAE', 'MissForest']

DATASET_NAME = {
    'FICO': 'FICO',
    'BREAST_CANCER': 'Breast Cancer', 
    'FICO_MAR': 'Synthetic',
    'FICO_MAR_25': 'FICO + Miss Rate 0.25',#'Synthetic Missingness @ rate=0.25, FICO',
    'FICO_MAR_50': 'Miss Rate 0.5, FICO',#'Synthetic Missingness @ rate=0.5, FICO',
    'BREAST_CANCER_MAR': 'BREAST_CANCER + Synthetic Missingness',
    'BREAST_CANCER_MAR_25': 'Miss Rate 0.25, BREAST_CANCER',#'Synthetic Missingness 0.25, BREAST_CANCER',
    'BREAST_CANCER_MAR_50': 'Miss Rate 0.5, BREAST_CANCER',#'Synthetic Missingness 0.5, BREAST_CANCER', 
    'SYNTHETIC_MAR': 'Synthetic MAR',
    'SYNTHETIC_MAR_50': 'Synthetic data, MAR missingness 0.5',
    'SYNTHETIC_MAR_25': 'Synthetic data, MAR missingness 0.5',
    'SYNTHETIC_CATEGORICAL_MAR': 'Synthetic Categorical MAR',
    'SYNTHETIC_CATEGORICAL_MAR_50': 'Synthetic Categorical data, MAR missingness 0.5',
    'SYNTHETIC_CATEGORICAL_MAR_25': 'Synthetic Categorical data, MAR missingness 0.5',
    'HEART_DISEASE': 'Heart Disease',
    'CKD': 'CKD', 
    'PHARYNGITIS': 'Pharyngitis', 
    'MIMIC': 'MIMIC'
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
num_quantiles = 8
quantile_addition = ''

def plot_subfig(dataset, f, ax, ax1, ax2, legend=False, include_xlabel=True, add_to_legend='', include_ylabel=True):
    y_lim=(0.7, 0.87)
    y_lim=(0.6, 0.77) if 'PHARYNGITIS' in dataset else y_lim
    y_lim=(0.89, 0.94) if 'MIMIC' in dataset else y_lim
    y_lim=(0.94, 1) if 'CKD' in dataset else y_lim
    y_lim=(0.7, 0.9) if 'HEART_DISEASE' in dataset else y_lim

    metric = 'acc' if 'BREAST_CANCER' not in dataset else 'auc' 
    if sparsity_metric == 'default': 
        s_size_cutoff = 52.5 if 'CKD' not in dataset else 10
    else:
        s_size_cutoff = 25
    if dataset == 'HEART_DISEASE': 
        s_size_cutoff = 100
    if dataset == 'PHARYNGITIS': 
        s_size_cutoff = 40
    if dataset == 'MIMIC':
        s_size_cutoff = 60

    window_cutoff = 160 if 'BREAST_CANCER' not in dataset else 90
    window_cutoff = 75 if 'PHARYNGITIS' in dataset else window_cutoff
    window_cutoff = 375 if 'MIMIC' in dataset else window_cutoff
    window_cutoff = 125 if 'CKD' in dataset else window_cutoff
    # window_cutoff = 60 if 'HEART_DISEASE' in dataset else window_cutoff


    # Plot titles, etc
    x_title = ('\# Nonzero Coefficients' if sparsity_metric=='default' else '\# Nonzero Shape Functions')

    #load data files from csv: 

    res_dir = f'experiment_data/{dataset}'
    res_dir = f'{res_dir}/{distinct}distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}'

    if train_miss != 0 or test_miss != 0: 
        res_dir = f'{res_dir}/train_{train_miss}/test_{test_miss}'
    if num_quantiles != 8: 
        res_dir = f'{res_dir}/q{num_quantiles}'

    imputation_res_dirs = []
    for imputer in baseline_imputers: 
        imp_res_dir = f'experiment_data/{dataset}/noreg/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}/impute_{imputer}/100'+(
            f'/{mice_augmentation_level}' if mice_augmentation_level > 0 else '')+(
                f'/{imputer}' if imputer != 'MICE' else ''
            )
        imputation_res_dirs.append(imp_res_dir)

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
    nllambda = np.loadtxt(f'{res_dir}/nllambda.csv')

    sparsity_aug = np.loadtxt(f'{res_dir}/sparsity_aug.csv')
    sparsity_indicator = np.loadtxt(f'{res_dir}/sparsity_indicator.csv')
    sparsity_no_missing = np.loadtxt(f'{res_dir}/sparsity_no_missing.csv')

    # approach for baselines that *do* have a sparsity measure (mean, extreme value)
    def load_train_test_sparsity(method='SMIM', aug_level = 0, use_mean=True): 
        m_res_dir = f'experiment_data/{dataset}/{method}'
        m_res_dir = f'{m_res_dir}/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}'
        if not use_mean: 
            m_res_dir += f'/impute_None'
        if aug_level != 0:
            m_res_dir += f'/aug_{aug_level}'
        return (
            np.loadtxt(f'{m_res_dir}/train_{metric}_indicator.csv'),
            np.loadtxt(f'{m_res_dir}/test_{metric}_indicator.csv'),
            np.loadtxt(f'{m_res_dir}/sparsity_no_missing.csv') #use of no_missing here for consistency with a typo in baseline
        )

    smim_train_perf, smim_test_perf, smim_sparsity = load_train_test_sparsity()

    noreg_train_perf, noreg_test_perf, noreg_sparsity = load_train_test_sparsity(method='noreg')
    # noreg_ind_train_perf, noreg_ind_test_perf, noreg_ind_sparsity = load_train_test_sparsity(method='noreg', aug_level=1)
    # noreg_aug_train_perf, noreg_aug_test_perf, noreg_aug_sparsity = load_train_test_sparsity(method='noreg', aug_level=2)

    noreg_nomean_perf, noreg_nomean_test_perf, noreg_nomean_sparsity = load_train_test_sparsity(method='noreg', use_mean=False)
    # noreg_nomean_ind_train_perf, noreg_nomean_ind_test_perf, noreg_nomean_ind_sparsity = load_train_test_sparsity(method='noreg', aug_level=1, use_mean=False)
    # noreg_nomean_aug_train_perf, noreg_nomean_aug_test_perf, noreg_nomean_aug_sparsity = load_train_test_sparsity(method='noreg', aug_level=2, use_mean=False)

    imputation_results_test = []
    imputation_results_train = []
    for imp_res_dir in imputation_res_dirs: 
        imputation_ensemble_train_auc = np.loadtxt(f'{imp_res_dir}/imputation_ensemble_train_{metric}.csv')
        imputation_ensemble_test_auc = np.loadtxt(f'{imp_res_dir}/imputation_ensemble_test_{metric}.csv')
        #filter out any skipped (0 valued) runs of imputation (TODO: verify this has an effect for breca)
        # imputation_ensemble_test_auc = imputation_ensemble_test_auc[imputation_ensemble_test_auc > 0]
        imputation_ensemble_train_auc = imputation_ensemble_train_auc[imputation_ensemble_train_auc > 0]
        imputation_ensemble_test_auc[imputation_ensemble_test_auc == 0] = np.nan
        imputation_results_test.append(imputation_ensemble_test_auc)
        imputation_results_train.append(imputation_ensemble_train_auc)

    no_timeouts_aug = (train_auc_aug > 0).all(axis=0)
    no_timeouts_and_sparse_aug = np.logical_and(no_timeouts_aug, (sparsity_aug < s_size_cutoff).all(axis=0))
    # verify that no_timeouts_and_sparse_aug is monotonic (specifically, that it never goes from true to false and back)
    assert((no_timeouts_and_sparse_aug[1:] <= no_timeouts_and_sparse_aug[:-1]).all())

    sparsity_aug = sparsity_aug[:, no_timeouts_and_sparse_aug]
    train_auc_aug = train_auc_aug[:, no_timeouts_and_sparse_aug]
    test_auc_aug = test_auc_aug[:, no_timeouts_and_sparse_aug]

    no_timeouts_indicator = (train_auc_indicator > 0).all(axis=0)
    no_timeouts_and_sparse_indicator = np.logical_and(no_timeouts_indicator, (sparsity_indicator < s_size_cutoff).all(axis=0))
    assert((no_timeouts_and_sparse_indicator[1:] <= no_timeouts_and_sparse_indicator[:-1]).all())

    sparsity_indicator = sparsity_indicator[:, no_timeouts_and_sparse_indicator]
    train_auc_indicator = train_auc_indicator[:, no_timeouts_and_sparse_indicator]
    test_auc_indicator = test_auc_indicator[:, no_timeouts_and_sparse_indicator]
    # import pdb; pdb.set_trace()

    # no_timeouts_no_missing = (train_auc_no_missing > 0).all(axis=0)
    # sparsity_no_missing = sparsity_no_missing[:, no_timeouts_no_missing]
    # train_auc_no_missing = train_auc_no_missing[:, no_timeouts_no_missing]
    # test_auc_no_missing = test_auc_no_missing[:, no_timeouts_no_missing]

    fig_dir = f'./figs/{dataset}/'
    fig_dir = f'{fig_dir}distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}/'
    fig_dir = f'{fig_dir}{s_size_folder}/'
    fig_dir += f'{mice_augmentation_level}/' if mice_augmentation_level > 0 else ''
    fig_dir += f'{baseline_imputers}/' if baseline_imputers != ['MICE'] else ''
    if sparsity_metric != 'default': 
        fig_dir = f'{fig_dir}sparsity_{sparsity_metric}/'
    if mgam_imputer != None: 
        fig_dir = f'{fig_dir}imputer_{mgam_imputer}/'

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
 
    # plot the same data on both axes
    train_or_test_str = 'Train' if is_train else 'Test'
    vals = TRAIN_VALS if is_train else TEST_VALS
    ax.set_prop_cycle(None)

    # f.suptitle(f'{train_or_test_str} {METRIC_NAME[metric]} vs {x_title} \n for {dataset} dataset{quantile_addition}')
    # ax.hlines(imputation_ensemble_train_auc.mean(), 0,
    #            max([sparsity_aug.max(), sparsity_indicator.max()]), linestyles='dashed',
    #            label='mean performance, ensemble of 10 MICE imputations',
    #            color='grey') #TODO: add error bars? 
    # uncertainty_bands_subplot(vals[0][0], vals[0][1], 'No missingness handling', ax)
    # uncertainty_bands_subplot(vals[1][0], vals[1][1], 'Missingness with interactions', ax)
    uncertainty_bands_subplot(vals[2][0], vals[2][1], None, ax) 
    uncertainty_bands_subplot(vals[1][0], vals[1][1], None, ax)
    # SMIM plot (TODO: make this on a third axis if needed)
    uncertainty_bands_subplot(smim_sparsity, smim_train_perf if is_train else smim_test_perf, None, ax)
    uncertainty_bands_subplot(noreg_sparsity, noreg_train_perf if is_train else noreg_test_perf, None, ax)# w/ MVI
    # uncertainty_bands_subplot(noreg_ind_sparsity, noreg_ind_train_perf if is_train else noreg_ind_test_perf, 'GAM w/ Indicators', ax)
    # uncertainty_bands_subplot(noreg_aug_sparsity, noreg_aug_train_perf if is_train else noreg_aug_test_perf, 'GAM w/ Interactions', ax)
    uncertainty_bands_subplot(noreg_nomean_sparsity, noreg_nomean_perf if is_train else noreg_nomean_test_perf, None, ax)

    # ax.set_xlabel(x_title)
    if include_ylabel:
        ax.set_ylabel(add_to_legend + f'{train_or_test_str} {METRIC_NAME[metric]}')
    # ax.set_ylabel(f'{train_or_test_str} {METRIC_NAME[metric]}')
    if 'PHARYNGITIS' not in dataset and 'HEART_DISEASE' not in dataset:
        ax1.set_prop_cycle(None)

        uncertainty_bands_subplot(vals[2][0], vals[2][1], None, ax1) 
        uncertainty_bands_subplot(vals[1][0], vals[1][1], None, ax1)
        uncertainty_bands_subplot(smim_sparsity, smim_train_perf if is_train else smim_test_perf, None, ax1)
        uncertainty_bands_subplot(noreg_sparsity, noreg_train_perf if is_train else noreg_test_perf, None, ax1)# w/ MVI
        # uncertainty_bands_subplot(noreg_ind_sparsity, noreg_ind_train_perf if is_train else noreg_ind_test_perf, 'GAM w/ Indicators & MVI', ax1)
        # uncertainty_bands_subplot(noreg_aug_sparsity, noreg_aug_train_perf if is_train else noreg_aug_test_perf, 'GAM w/ Interactions & MVI', ax1)
        # eb = ax1.errorbar(np.nanmean(noreg_aug_sparsity), np.nanmean(noreg_aug_train_perf if is_train else noreg_aug_test_perf), yerr=np.nanstd(noreg_aug_train_perf if is_train else noreg_aug_test_perf)/np.sqrt(10), xerr=np.nanstd(noreg_aug_sparsity)/np.sqrt(10), label='GAM w/ Interactions & MVI')

        # uncertainty_bands_subplot(noreg_nomean_ind_sparsity, noreg_nomean_ind_train_perf if is_train else noreg_nomean_ind_test_perf, 'GAM w/ Indicators', ax1)
        # uncertainty_bands_subplot(noreg_nomean_aug_sparsity, noreg_nomean_aug_train_perf if is_train else noreg_nomean_aug_test_perf, 'GAM w/ Interactions', ax1)
        uncertainty_bands_subplot(noreg_nomean_sparsity, noreg_nomean_perf if is_train else noreg_nomean_test_perf, None, ax1)

    # ax1.set_xlabel(x_title)

    # uncertainty_bands_subplot(vals[2][0], vals[2][1], 'Missingness indicators', ax1b) 
    # uncertainty_bands_subplot(vals[1][0], vals[1][1], 'Missingness with interactions', ax1b)
    # uncertainty_bands_subplot(smim_sparsity, smim_train_perf if is_train else smim_test_perf, 'SMIM', ax1b)
    # uncertainty_bands_subplot(noreg_sparsity, noreg_train_perf if is_train else noreg_test_perf, 'GAM', ax1b)# w/ MVI
    # uncertainty_bands_subplot(noreg_ind_sparsity, noreg_ind_train_perf if is_train else noreg_ind_test_perf, 'GAM w/ Indicators & MVI', ax1b)
    # uncertainty_bands_subplot(noreg_aug_sparsity, noreg_aug_train_perf if is_train else noreg_aug_test_perf, 'GAM w/ Interactions & MVI', ax1b)

    # # uncertainty_bands_subplot(noreg_nomean_ind_sparsity, noreg_nomean_ind_train_perf if is_train else noreg_nomean_ind_test_perf, 'GAM w/ Indicators', ax1b)
    # # uncertainty_bands_subplot(noreg_nomean_aug_sparsity, noreg_nomean_aug_train_perf if is_train else noreg_nomean_aug_test_perf, 'GAM w/ Interactions', ax1b)

    # ax1b.set_xlabel(x_title)
    ax2.set_prop_cycle(None)
    uncertainty_bands_subplot(sparsity_indicator, train_auc_indicator, 'M-GAM (Indicators Only)' if legend else None, ax2) 
    uncertainty_bands_subplot(sparsity_aug, train_auc_aug, 'MGAM (w/ Interactions)' if legend else None, ax2)
    uncertainty_bands_subplot(smim_sparsity, smim_train_perf, 'SMIM' if legend else None, ax2)
    uncertainty_bands_subplot(noreg_sparsity, noreg_train_perf if is_train else noreg_test_perf, 'GAM w/ MVI' if legend else None, ax2)# w/ MVI
    # uncertainty_bands_subplot(noreg_ind_sparsity, noreg_ind_train_perf if is_train else noreg_ind_test_perf, 'GAM w/ MVI, Indicators', ax2)
    # uncertainty_bands_subplot(noreg_aug_sparsity, noreg_aug_train_perf if is_train else noreg_aug_test_perf, 'GAM w/ MVI, Interactions', ax2)
    uncertainty_bands_subplot(noreg_nomean_sparsity, noreg_nomean_perf if is_train else noreg_nomean_test_perf, 'GAM' if legend else None, ax2)
    # uncertainty_bands_subplot(noreg_nomean_ind_sparsity, noreg_nomean_ind_train_perf if is_train else noreg_nomean_ind_test_perf, 'GAM w/ Indicators', ax2)
    # uncertainty_bands_subplot(noreg_nomean_aug_sparsity, noreg_nomean_aug_train_perf if is_train else noreg_nomean_aug_test_perf, 'GAM w/ Interactions', ax2)
    for idx, imp_result in enumerate(imputation_results_train if is_train else imputation_results_test): 
        #plot a point with an error bar at x=201: 
        plot_position = 200 + (idx+1)/(len(baseline_imputers)+1)
        marker=['o', '^', '>', 'v', '<'][idx % 5]
        # colour = ['grey', 'purple', 'green', 'blue', 'red'][idx % 5]
        # ax2.plot(plot_position, np.nanmean(imp_result), marker=marker, color = 'black')
        eb = ax2.errorbar(plot_position, np.nanmean(imp_result), yerr=np.nanstd(imp_result)/np.sqrt(10), xerr = 1, marker=marker, label=baseline_imputers[idx] if legend else None)
        # eb[-1][0].set_linestyle('-') 
        eb[-1][1].set_linestyle('--') 
        # ax2.hlines(np.nanmean(imp_result), 0,
        #         1000, linestyles='dashed',
        #         #label= baseline_imputers[idx],#'mean performance, ensemble of 10 MICE imputations',
        #         color='grey')
        # if idx == 0: 
        #     uncertainty_bands_subplot_mice(imp_result, None, ax2)
        # uncertainty_bands_subplot(sparsity_no_missing, train_auc_no_missing, 'No Missingness \n Handling', ax2)
    ax2.set_xlabel('*')
    ax2.set_xticks([]) # need to replace xticks with infinity


    # f.legend(loc='lower center', fancybox=True, shadow=True, ncol=4, bbox_to_anchor=(0.5, -0.15))

    ax.set_xlim(0, s_size_cutoff)
    if 'PHARYNGITIS' in dataset or 'HEART_DISEASE' in dataset: 
        ax.set_xlim(0, 60)
    # if 'HEART_DISEASE' in dataset:
    #     ax.set_xlim(0, 100)
    # ax1b.set_xlim(window_cutoff, noreg_aug_sparsity.max()+100)
    ax2.set_xlim(200, 201)
    ax.set_ylim(*y_lim)
    if 'PHARYNGITIS' not in dataset and 'HEART_DISEASE' not in dataset: 
        ax1.set_ylim(*y_lim)
        if 'MIMIC' in dataset: 
            ax1.set_xlim(300, 350)
        elif 'BREAST_CANCER' not in dataset: 
            ax1.set_xlim(window_cutoff-40, window_cutoff)
        else: 
            ax1.set_xlim(window_cutoff-20, window_cutoff)
    ax2.set_ylim(*y_lim)


    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # ax1b.spines['right'].set_visible(False)
    # ax1b.spines['left'].set_visible(False)
    # ax1b.yaxis.set_ticks_position('none')
    ax.yaxis.tick_left()
    ax.tick_params(labelright=False)
    ax2.yaxis.tick_right()
    # hide the spines between ax and ax2
    if 'PHARYNGITIS' not in dataset and 'HEART_DISEASE' not in dataset: 
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.yaxis.set_ticks_position('none')


    # In axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0, 0), (0, 1),
    # (1, 0), and (1, 1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-(1/2)*d, 1+(1/2)*d), (-d, d), **kwargs)
    ax.plot((1-(1/2)*d, 1+1/2*d), (1-d, 1+d), **kwargs)

    if 'PHARYNGITIS' not in dataset and 'HEART_DISEASE' not in dataset: 
        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax1.plot((-d, +d), (-d, +d), **kwargs)

        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1-(1)*d, 1+(1)*d), (-d, d), **kwargs)
        ax1.plot((1-(1)*d, 1+1*d), (1-d, 1+d), **kwargs)

    # kwargs.update(transform=ax1b.transAxes)  # switch to the bottom axes
    # ax1b.plot((-d, +d), (1-d, 1+d), **kwargs)
    # ax1b.plot((-d, +d), (-d, +d), **kwargs)

    # kwargs = dict(transform=ax1b.transAxes, color='k', clip_on=False)
    # ax1b.plot((1-(1)*d, 1+(1)*d), (-d, d), **kwargs)
    # ax1b.plot((1-(1)*d, 1+1*d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    #update whole plot's x label: 
    if include_xlabel: 
        f.text(1.57, -0.07, x_title, ha='center')
        f.text(0.57, -0.07, x_title, ha='center')


f_overall = plt.figure(layout='constrained', figsize=(20, 25))
subfigs_overall = f_overall.subfigures(len(datasets), 1)

# mar_added = []
# for dataset in datasets: 

for i, dataset in enumerate(datasets): 
    # import pdb; pdb.set_trace()
    subfigs = subfigs_overall[i].subfigures(1, 2)
    if 'PHARYNGITIS' in dataset or 'HEART_DISEASE' in dataset:  
        axs25 = subfigs[0].subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2, 1]})
        axs50 = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2, 1]})
        plot_subfig(dataset + '_MAR_25', subfigs[0], axs25[0], None, axs25[1], add_to_legend=DATASET_NAME[dataset]+'\n\n', include_xlabel=(i==(len(datasets)-1)))
        plot_subfig(dataset + '_MAR_50', subfigs[0], axs50[0], None, axs50[1], legend=(i==(len(datasets)-1)), include_xlabel=(i==(len(datasets)-1)), include_ylabel=False)
    else: 
        axs25 = subfigs[0].subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [2, 1, 1]})
        axs50 = subfigs[1].subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [2, 1, 1]})
        plot_subfig(dataset + '_MAR_25', subfigs[0], axs25[0], axs25[1], axs25[2], add_to_legend=DATASET_NAME[dataset]+'\n\n', include_xlabel=(i==(len(datasets)-1)))
        plot_subfig(dataset + '_MAR_50', subfigs[1], axs50[0], axs50[1], axs50[2], legend=(i==(len(datasets)-1)), include_xlabel=(i==(len(datasets)-1)), include_ylabel=False)
    if i == 0: 
        subfigs[0].suptitle('Rate $\leq$ 0.25', fontsize='x-large', x=0.57)
        subfigs[1].suptitle('Rate $\leq$ 0.5', fontsize='x-large', x=0.57)
    # subfigs_overall[i].suptitle(DATASET_NAME[dataset], fontsize='x-large', x=0.1, y=0.7)
    # mar_added.append(dataset + '_MAR_25')
    # mar_added.append(dataset + '_MAR_50')

f_overall.suptitle('Performance under Added Synthetic MAR Missingness', fontsize= 'xx-large')
legend = f_overall.legend(loc='lower center', fancybox=True, shadow=True, ncol=4, bbox_to_anchor=(0.5, -0.07))

for i in [0, 1]: 
    legend.get_texts()[i].set_weight('bold')

plt.savefig(f'figs/test{datasets}.{filetype}', bbox_inches='tight')

print('successfully finished execution')
print(f'figs/test{datasets}.{filetype}')
