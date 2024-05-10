#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=baselines # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=logs/baselines_%j.out
#SBATCH --ntasks=1                 # Run on a single Node
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=100gb                     # Job memory request
#not SBATCH  -x linux[41-60]
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci

import os
import sys
import re
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn import metrics
import fastsparsegams
import matplotlib.pyplot as plt
from mice_utils import eval_model, return_imputation, eval_model_by_clusters
from binarizer import Binarizer
from mice_utils import get_smim_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

### Constants
# Specifies approaches used for imputer underlying the MGAM approach
# (usually None, but we need alternative options for one figure)
M_GAM_IMPUTERS = {
    None: None,#lambda x : 0, # corresponds to encoding missing values as False
    'mean': np.mean,
    'median': np.median
}

#control flow variables
run_impute_experiments = True
run_indicator_experiments = False

#hyperparameters (TODO: set up with argparse)
method = 'noreg' # 'SMIM', 'noreg'
num_quantiles = 8
dataset = 'FICO'#['FICO', 'CKD', 'BREAST_CANCER', 'MIMIC', 'HEART_DISEASE', 'PHARYNGITIS', 'ADULT']
train_miss = 0
test_miss = train_miss

metric = 'acc'

overall_mi_intercept = False
overall_mi_ixn = False
specific_mi_intercept = True
specific_mi_ixn = True

#we can impute in addition to using indicators. 
mgam_imputer = None
mice_augmentation_level = 0 # 0 for no missingness features, 1 for indicators, 2 for interactions

# multiple sparsity metrics
sparsity_metric = 'default'#'default'

#imputation baseline
baseline_imputer = "MICE" # None GAIN/  Mean/  MICE/  MissForest/  MIWAE/

print('--- Hyperparameter Settings ---')
print(f'Dataset: {dataset}')
print(f'Quantiles: {num_quantiles}')
print(f'Distinctness pattern: (overall intercept, overall ixn, specific intercept, specific interaction): ({overall_mi_intercept}, {overall_mi_ixn}, {specific_mi_intercept}, {specific_mi_ixn})')
print(f'Missingness augmentation level: {mice_augmentation_level}')
print(f'Baseline imputer: {baseline_imputer}')
print(f'MGAM imputer: {mgam_imputer}')
print(f'Sparsity metric: {sparsity_metric}')
print(f'run_indicator_experiments: {run_indicator_experiments}')
print(f'run_impute_experiments: {run_impute_experiments}')

### Immutable ###
use_distinct = True
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]
holdouts = np.arange(10)
validations = np.arange(5)
imputations = np.arange(10)
mice_validation_metric = metric
s_size=100
np.random.seed(0)
smim_grid = {'C':[0.01, 0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [5_000], 'tol': [1e-2]}
###################

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

for holdout_set in holdouts: 
    for val_set in validations: 
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
                            categorical_cols = categorical_cols, numerical_cols = numerical_cols, 
                            imputer = M_GAM_IMPUTERS[mgam_imputer]) 
        # TODO: encode miss_vals to the specific set of missingness values for this dataset, 
        # or add functionality to prune out non-occurring missingness values in binarizer
        # TODO: move outside of this loop; all we need from this loop is the label and predictors, which 
        # are constant for the dataset anyways
        ###########################
        ### Imputation approach ###
        ###########################
        if run_impute_experiments: 
            if 'BREAST_CANCER' in dataset and val_set == 3 and holdout_set == 6: #missed run at this stage; will have to adjust for plotting as well
                continue
            if 'BREAST_CANCER_MAR_25' in dataset and val_set == 2 and holdout_set == 9: #missed run at this stage; will have to adjust for plotting as well
                continue
            imputation_train_probs = np.zeros((len(imputations), train.shape[0] + val.shape[0]))
            imputation_test_probs = np.zeros((len(imputations), test.shape[0]))

            best_models_per_imputation = []

            #find best lambda for each imputation, using validation set. 
                #currently, selects best auc using only one validation fold. 
                # one possible TODO is to select using 5-fold cross validation
                # (this may require verifying whether all 5 train/val splits use the same imputation)
                # using all folds for each imputation may increase the runtime of the full train/val/test pipeline non-trivially. 
            for imputation_idx, imputation in enumerate(imputations):
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
                grid_search = GridSearchCV(clf, smim_grid, cv=5)
                grid_search.fit(X_train, y_train)
                
                imputation_train_probs[imputation_idx] = grid_search.predict_proba(X_train)[:, 1]
                imputation_test_probs[imputation_idx] = grid_search.predict_proba(X_test)[:, 1]

            # calculate ensembled metric across all imputations: 
            ensembled_train_probs = imputation_train_probs.mean(axis=0)
            ensembled_test_probs = imputation_test_probs.mean(axis=0)

            imputation_ensemble_train_auc[val_set, holdout_set] = METRIC_FN[metric](y_train, ensembled_train_probs)

            imputation_ensemble_test_auc[val_set, holdout_set] = METRIC_FN[metric](y_test, ensembled_test_probs)
        ######################################
        ### Missingness Indicator Approach ###
        ######################################
        if run_indicator_experiments:
            if val_set != validations[0]: #we can just use the first val_set; no need to rerun this for each validation
                continue

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
                grid_search = GridSearchCV(clf, smim_grid, cv=5)
                grid_search.fit(train_smim, y_train_no)
                print("best params: ", grid_search.best_params_)

                sparsity_no_missing[holdout_set] = (grid_search.best_estimator_.coef_ !=0).sum() if sparsity_metric == 'default' else -1 #not implemented for non-default

                train_auc_indicator[holdout_set] = grid_search.score(train_smim, y_train_no)if metric == 'acc' else -1 #not implemented yet
                test_auc_indicator[holdout_set] = grid_search.score(test_smim, y_test_no)if metric == 'acc' else -1 #not implemented yet

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

                (train_no, train_ind, train_aug, 
                test_no, test_ind, test_aug, 
                y_train_no, y_train_ind, y_train_aug, 
                y_test_no, y_test_ind, y_test_aug, 
                cluster_no, cluster_ind, cluster_aug) = encoder.binarize_and_augment(
                    pd.concat([train_i, val_i]), test_i)

                # tune & fit logistic regression on data w/o missingness indicators
                clf = LogisticRegression(random_state=0)
                grid_search = GridSearchCV(clf, smim_grid, cv=5)
                grid_search.fit(train_no, y_train_no)
                print("best params: ", grid_search.best_params_)

                sparsity_no_missing[holdout_set] = (grid_search.best_estimator_.coef_ !=0).sum() if sparsity_metric == 'default' else -1 #not implemented for non-default

                train_auc_indicator[holdout_set] = grid_search.score(train_no, y_train_no)if metric == 'acc' else -1 #not implemented yet
                test_auc_indicator[holdout_set] = grid_search.score(test_no, y_test_no)if metric == 'acc' else -1 #not implemented yet

#save data to csv: 


results_path = f'experiment_data/{dataset}/{method}'
## possibly-antiquated hyperparameters still needed to preserve file structure of experiment: 
results_path = f'{results_path}/distinctness_{overall_mi_intercept}_{overall_mi_ixn}_{specific_mi_intercept}_{specific_mi_ixn}'
if train_miss != 0 or test_miss != 0: 
    results_path = f'{results_path}/train_{train_miss}/test_{test_miss}'

if baseline_imputer != 'Mean': #let default be mean imputation for now
    results_path += f'/impute_{baseline_imputer}'

MICE_results_path = f'{results_path}/100'#100 lets us keep things consistent with prior folder structures
if mice_augmentation_level > 0: 
    MICE_results_path += f'/{mice_augmentation_level}'
if baseline_imputer != 'MICE':
    MICE_results_path += f'/{baseline_imputer}'
if not os.path.exists(MICE_results_path):
    os.makedirs(MICE_results_path)

if sparsity_metric != 'default': 
    results_path = f'{results_path}/sparsity_{sparsity_metric}'
if mgam_imputer != None: 
    results_path = f'{results_path}/imputer_{mgam_imputer}'
if not os.path.exists(results_path):
    os.makedirs(results_path)

if run_impute_experiments:
    np.savetxt(f'{MICE_results_path}/imputation_ensemble_train_{metric}.csv', imputation_ensemble_train_auc)
    np.savetxt(f'{MICE_results_path}/imputation_ensemble_test_{metric}.csv', imputation_ensemble_test_auc)

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
