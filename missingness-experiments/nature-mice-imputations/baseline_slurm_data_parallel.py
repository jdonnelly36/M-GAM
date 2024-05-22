#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
import os
import sys
import re
sys.path.append(os.getcwd())
sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from mice_utils import eval_model, return_imputation, eval_model_by_clusters
from binarizer import Binarizer
from mice_utils import get_smim_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Baseline Slurm Data')

# Add the command line arguments
parser.add_argument('--run_impute_experiments', action='store_true', help='Run impute experiments')
parser.add_argument('--run_indicator_experiments', action='store_true', help='Run indicator experiments')
parser.add_argument('--method', type=str, default='noreg', help='Method (SMIM or noreg)')
parser.add_argument('--dataset', type=str, default='FICO', help='Dataset')
parser.add_argument('--metric', type=str, default='acc', help='Metric')
parser.add_argument('--mice_augmentation_level', type=int, default=0, help='Imputation augmentation level. 0, 1 (indicator), or 2(interaction) are options')
parser.add_argument('--sparsity_metric', type=str, default='default', help='Sparsity metric')
parser.add_argument('--baseline_imputer', type=str, default='Mean', help='Baseline imputer')
parser.add_argument('--baseline_aug_level', type=int, default=0, help='Baseline augmentation level, for baselines where we need to measure sparsity. 0, 1 (indicator), or 2(interaction) are options')
parser.add_argument('--rerun', action='store_true', help='flag to force a rerun if files already exist. Without this flag, experiments are not repeated')
parser.add_argument('--holdouts', type=int, nargs='+', help='which holdout sets to use')

# Parse the command line arguments
args = parser.parse_args()

# Assign the command line arguments to variables
run_impute_experiments = args.run_impute_experiments
run_indicator_experiments = args.run_indicator_experiments
rerun = args.rerun
method = args.method
dataset = args.dataset
metric = args.metric
mice_augmentation_level = args.mice_augmentation_level
sparsity_metric = args.sparsity_metric
baseline_imputer = args.baseline_imputer
baseline_aug_level = args.baseline_aug_level
holdouts = args.holdouts

if baseline_imputer == 'None':
    baseline_imputer = None

### Immutable ###
overall_mi_intercept = False
overall_mi_ixn = False
specific_mi_intercept = True
specific_mi_ixn = True
train_miss = 0
test_miss = 0
num_quantiles = 8
use_distinct = True
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]
# holdouts = np.arange(10)
validations = np.arange(5)
imputations = np.arange(10)
mice_validation_metric = metric
s_size=100
np.random.seed(0)
smim_grid = {'C':[0.01, 0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [5_000], 'tol': [1e-2]}
###################

# Print the hyperparameter settings
print('--- Hyperparameter Settings ---')
print(f'Dataset: {dataset}')
print(f'Quantiles: {num_quantiles}')
print(f'Distinctness pattern: (overall intercept, overall ixn, specific intercept, specific interaction): ({overall_mi_intercept}, {overall_mi_ixn}, {specific_mi_intercept}, {specific_mi_ixn})')
print(f'Missingness augmentation level: {mice_augmentation_level}')
print(f'Baseline imputer: {baseline_imputer}')
print(f'Sparsity metric: {sparsity_metric}')
print(f'run_indicator_experiments: {run_indicator_experiments}')
print(f'run_impute_experiments: {run_impute_experiments}')
print(f'baseline_aug_level: {baseline_aug_level}')
#print control flow variables: 
print('--- Control Flow Variables ---')
print(f'run_impute_experiments: {run_impute_experiments}')
print(f'run_indicator_experiments: {run_indicator_experiments}')

print('-----------------------------------------------')

### basic hyperparameter processing ###

miss_str = ''
if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
    miss_str = f'_train_missing_{train_miss}_test_missing_{test_miss}'

q_str = ''
if num_quantiles != 8:
    q_str = f'/q{num_quantiles}'

def calc_auc(y, score): 
    fpr, tpr, _ = metrics.roc_curve(y, score)
    return metrics.auc(fpr, tpr)
METRIC_FN = {
    'acc': lambda y, score: np.mean(
        (score > 0.5) == y),
    'auprc': lambda y, score: metrics.average_precision_score(y, score),
    'auc': calc_auc, 
    'loss': lambda y, probs: np.exp(np.log(probs/(1 - probs))*-1/2*(2*y-1)).mean()
}

def path_to_imputed(dataset, holdout_set, val_set, imputation):
    if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
        return  f'/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/JON_IMPUTED_DATA/{dataset}/{baseline_imputer}/train_per_{train_miss}/test_per_{test_miss}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'
    return f'/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/JON_IMPUTED_DATA/{dataset}/{baseline_imputer}/train_per_0/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'

def path_to_mean(dataset, holdout_set, val_set, imputation):
    if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
        return  f'/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/JON_IMPUTED_DATA/{dataset}/Mean/train_per_{train_miss}/test_per_{test_miss}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'
    return f'/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/JON_IMPUTED_DATA/{dataset}/Mean/train_per_0/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/'


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
    prefix = f'{standard_prefix}/{dataset}/'
    if use_distinct and 'FICO' in dataset or 'SYNTHETIC' in dataset or 'MAR' in dataset: 
        return f'{prefix}distinct-missingness/'
    else: 
        return prefix

###############################################

# check if experiment has already been run: 

results_path = f'experiment_data/{dataset}/{method}'
## possibly-antiquated hyperparameters still needed to preserve file structure of experiment: 
results_path = f'{results_path}/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}'
if train_miss != 0 or test_miss != 0: 
    results_path = f'{results_path}/train_{train_miss}/test_{test_miss}'

if baseline_imputer != 'Mean': #let default be mean imputation for now
    results_path += f'/impute_{baseline_imputer}'
if baseline_aug_level != 0:
    results_path += f'/aug_{baseline_aug_level}'

MICE_results_path = f'{results_path}/100'#100 lets us keep things consistent with prior folder structures
if mice_augmentation_level > 0: 
    MICE_results_path += f'/{mice_augmentation_level}'
if baseline_imputer != 'MICE':
    MICE_results_path += f'/{baseline_imputer}'

if sparsity_metric != 'default': 
    results_path = f'{results_path}/sparsity_{sparsity_metric}'

if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(MICE_results_path):
    os.makedirs(MICE_results_path)

if run_impute_experiments:
    save_file = f'{MICE_results_path}/imputation_ensemble_train_{metric}{holdouts}.csv'
    #if save file already exists: 
    if os.path.exists(save_file):
        print(f"Save file '{save_file}' already exists.")
        if not rerun: 
            run_impute_experiments = False
    

# overkill on saved files
if run_indicator_experiments: 
    save_file = f'{results_path}/train_{metric}_aug.csv'
    if os.path.exists(save_file):
        print(f"Save file '{save_file}' already exists.")
        if not rerun: 
            run_indicator_experiments = False
#############################################
### measures across trials, for plotting: ###
#############################################

#performance of best imputation ensemble (using validation set to select lambda_0) for each holdout/val combo
#(can consider larger numbers of imputations than 10, going forwards. Starting with that for now. )
imputation_ensemble_train_auc = np.zeros((len(validations), len(holdouts)))
imputation_ensemble_test_auc = np.zeros((len(validations), len(holdouts)))

#variables for performance of indicator methods, across each_holdout, for each lambda
train_auc_aug = np.zeros((len(holdouts), len(lambda_grid[0])))
train_auc_indicator = np.zeros((len(holdouts), len(lambda_grid[0])))
train_auc_no_missing = np.zeros((len(holdouts), len(lambda_grid[0])))
test_auc_aug = np.zeros((len(holdouts), len(lambda_grid[0])))
test_auc_indicator = np.zeros((len(holdouts), len(lambda_grid[0])))
test_auc_no_missing = np.zeros((len(holdouts), len(lambda_grid[0])))

#sparsity for each lambda, holdout, and method
sparsity_aug = np.zeros((len(holdouts), len(lambda_grid[0])))
sparsity_indicator = np.zeros((len(holdouts), len(lambda_grid[0])))
sparsity_no_missing = np.zeros((len(holdouts), len(lambda_grid[0])))

for holdout_index, holdout_set in enumerate(holdouts): 
    print('holdout_set: ', holdout_set)
    for val_set in validations: 
        if val_set != 0: 
            continue
        print('val_set: ', val_set)
        train = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_train_{val_set}{miss_str}.csv')
        val = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_val_{val_set}{miss_str}.csv')
        test = pd.read_csv(prefix_pre_imputed(dataset) + f'holdout_{holdout_set}{miss_str}.csv')
        label = train.columns[-1]
        predictors = train.columns[:-1]
        categorical_cols = [] #TODO: enter categorical cols
        numerical_cols = [c for c in predictors if c not in categorical_cols]

        encoder = Binarizer(quantiles = np.linspace(0, 1, num_quantiles + 2)[1:-1], label=label, 
                            miss_vals=[-7, -8, -9] if dataset=='FICO' else [np.nan, -7, -8, -9, -10], 
                            overall_mi_intercept = overall_mi_intercept, overall_mi_ixn = overall_mi_ixn, 
                            specific_mi_intercept = specific_mi_intercept, specific_mi_ixn = specific_mi_ixn, 
                            categorical_cols = categorical_cols, numerical_cols = numerical_cols) 
        ###########################
        ### Imputation approach ###
        ###########################
        if run_impute_experiments: 
            imputation_train_probs = np.zeros((len(imputations), train.shape[0] + val.shape[0]))
            imputation_test_probs = np.zeros((len(imputations), test.shape[0]))

            best_models_per_imputation = []

            for imputation_idx, imputation in enumerate(imputations):
                print('imputation: ', imputation)
                train_i, val_i, test_i = return_imputation(
                    path_to_imputed(dataset, holdout_set, val_set, imputation), 
                    label, predictors, train, test, val)
                
                imputed_and_binned_data = encoder.binarize_and_augment(pd.concat([train, val]), test, imputed_train_df = pd.concat([train_i, val_i]), imputed_test_df = test_i)

                # grab the level of augmentation from the encoder tuple
                X_train = imputed_and_binned_data[mice_augmentation_level].copy()
                X_test = imputed_and_binned_data[mice_augmentation_level + 3].copy()
                y_train = imputed_and_binned_data[6]
                y_test = imputed_and_binned_data[9]
                
                # tune & fit logistic regression on data
                clf = LogisticRegression(random_state=0)
                grid_search = GridSearchCV(clf, smim_grid, cv=5, scoring=('accuracy' if metric != 'auc' else 'roc_auc'))
                grid_search.fit(X_train, y_train)
                
                imputation_train_probs[imputation_idx] = grid_search.predict_proba(X_train)[:, 1]
                imputation_test_probs[imputation_idx] = grid_search.predict_proba(X_test)[:, 1]

            # calculate ensembled metric across all imputations: 
            ensembled_train_probs = imputation_train_probs.mean(axis=0)
            ensembled_test_probs = imputation_test_probs.mean(axis=0)

            imputation_ensemble_train_auc[val_set, holdout_index] = METRIC_FN[metric](y_train, ensembled_train_probs)

            imputation_ensemble_test_auc[val_set, holdout_index] = METRIC_FN[metric](y_test, ensembled_test_probs)
        ######################################
        ### Missingness Indicator Approach ###
        ######################################
        if run_indicator_experiments:
            if method == 'SMIM': #TODO: add version for noreg here AND for imputation
                #load non-binarized version of data, no imputation; replace all missing values with np.nan
                train_nan = train.replace(encoder.miss_vals, np.nan)
                val_nan = val.replace(encoder.miss_vals, np.nan)
                test_nan = test.replace(encoder.miss_vals, np.nan)

                # get train/test mask feats from get_smim_dataset
                train_val = pd.concat([train_nan, val_nan])
                train_mask, val_mask, _ = get_smim_dataset(train_val[predictors].to_numpy(), train_val[label].to_numpy(), test_nan[predictors].to_numpy())

                # binarize the non-binarized version of the data, filling in missing values with the mean or loading the mean imputations we made already and binarizing those
                if baseline_imputer == None: 
                    train_i, val_i, test_i = train_nan, val_nan, test_nan
                else: 
                    train_i, val_i, test_i = return_imputation(
                            path_to_imputed(dataset, holdout_set, val_set, 0), 
                            label, predictors, train, test, val)

                (train_no, train_ind, train_aug, 
                test_no, test_ind, test_aug, 
                y_train_no, y_train_ind, y_train_aug, 
                y_test_no, y_test_ind, y_test_aug, 
                cluster_no, cluster_ind, cluster_aug) = encoder.binarize_and_augment(
                    pd.concat([train_i, val_i]), test_i)

                # add the mask feats to these binarized datasets, tune & fit logistic regression on this data
                train_smim = np.concatenate([train_no, train_mask], axis=1)
                test_smim = np.concatenate([test_no, val_mask], axis=1)

                clf = LogisticRegression(random_state=0)
                grid_search = GridSearchCV(clf, smim_grid, cv=5, scoring=('accuracy' if metric != 'auc' else 'roc_auc'))
                grid_search.fit(train_smim, y_train_no)
                print("best params: ", grid_search.best_params_)

                sparsity_no_missing[holdout_index] = (grid_search.best_estimator_.coef_ !=0).sum() if sparsity_metric == 'default' else -1 #not implemented for non-default

                train_auc_indicator[holdout_index] = grid_search.score(train_smim, y_train_no) if (metric == 'acc' or metric == 'auc') else -1 #not implemented yet
                test_auc_indicator[holdout_index] = grid_search.score(test_smim, y_test_no)if (metric == 'acc' or metric == 'auc') else -1 #not implemented yet

            else:
                # binarize the non-binarized version of the data, filling in missing values with the mean or loading the mean imputations we made already and binarizing those
                # TODO: try a version with 0-imputation
                # TODO: try versions with any underlying imputations
                if baseline_imputer == None: 
                    train_i, val_i, test_i = train, val, test
                else: 
                    train_i, val_i, test_i = return_imputation(
                        path_to_imputed(dataset, holdout_set, val_set, 0), 
                        label, predictors, train, test, val)

                binarized_augmented_data = encoder.binarize_and_augment(
                    pd.concat([train, val]), test, imputed_train_df = pd.concat([train_i, val_i]), imputed_test_df = test_i)

                train_use = binarized_augmented_data[baseline_aug_level].copy()
                test_use = binarized_augmented_data[baseline_aug_level + 3].copy()
                y_train = binarized_augmented_data[6]
                y_test = binarized_augmented_data[9]

                # tune & fit logistic regression on data w/o missingness indicators
                clf = LogisticRegression(random_state=0)
                grid_search = GridSearchCV(clf, smim_grid, cv=5, scoring=('accuracy' if metric != 'auc' else 'roc_auc'))
                grid_search.fit(train_use, y_train)
                print("best params: ", grid_search.best_params_)

                sparsity_no_missing[holdout_index] = (grid_search.best_estimator_.coef_ !=0).sum() if sparsity_metric == 'default' else -1 #not implemented for non-default

                train_auc_indicator[holdout_index] = grid_search.score(train_use, y_train)if (metric == 'acc' or metric == 'auc') else -1 #not implemented yet
                test_auc_indicator[holdout_index] = grid_search.score(test_use, y_test)if (metric == 'acc' or metric == 'auc') else -1 #not implemented yet

#save data to csv: 

if run_impute_experiments:
    np.savetxt(f'{MICE_results_path}/imputation_ensemble_train_{metric}_{holdouts}.csv', imputation_ensemble_train_auc)
    np.savetxt(f'{MICE_results_path}/imputation_ensemble_test_{metric}_{holdouts}.csv', imputation_ensemble_test_auc)

# overkill on saved files
if run_indicator_experiments: 
    np.savetxt(f'{results_path}/train_{metric}_aug.csv', train_auc_aug)
    np.savetxt(f'{results_path}/train_{metric}_indicator.csv', train_auc_indicator)
    np.savetxt(f'{results_path}/train_{metric}_no_missing.csv', train_auc_no_missing)
    np.savetxt(f'{results_path}/test_{metric}_aug.csv', test_auc_aug)
    np.savetxt(f'{results_path}/test_{metric}_indicator.csv', test_auc_indicator)
    np.savetxt(f'{results_path}/test_{metric}_no_missing.csv', test_auc_no_missing)
    np.savetxt(f'{results_path}/nllambda.csv', -np.log(lambda_grid[0]))
    np.savetxt(f'{results_path}/sparsity_aug.csv', sparsity_aug)
    np.savetxt(f'{results_path}/sparsity_indicator.csv', sparsity_indicator)
    np.savetxt(f'{results_path}/sparsity_no_missing.csv',sparsity_no_missing)
print('successfully finished execution')
