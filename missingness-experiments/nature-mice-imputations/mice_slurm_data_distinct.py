#!/home/users/ham51/.venvs/fastsparsebuild/bin/python
#SBATCH --job-name=distinct # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ham51@duke.edu     # Where to send mail
#SBATCH --output=distinct_%j.out
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
from mice_utils import return_imputation, binarize_according_to_train, eval_model, get_train_test_binarized, binarize_and_augment, binarize_and_augment_distinct, errors

#hyperparameters (TODO: set up with argparse)
num_quantiles = 32
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]

holdouts = np.arange(10)#[0, 1, 2]
validations = np.arange(5)#
imputations = np.arange(10)

dataset = 'FICO'
train_miss = 0
test_miss = train_miss

metric = 'acc'
mice_validation_metric = metric
s_size=100

distinct_missingness = True

q_str = ''
if num_quantiles != 8:
    q_str = f'/q{num_quantiles}'

miss_str = ''
if train_miss != 0 or test_miss != 0 or 'SYNTHETIC' in dataset:
    miss_str = f'_train_missing_{train_miss}_test_missing_{test_miss}'

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
    print(holdout_set)
    for val_set in validations: 
        train = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_train_{val_set}{miss_str}.csv')
        val = pd.read_csv(prefix_pre_imputed(dataset)+f'devel_{holdout_set}_val_{val_set}{miss_str}.csv')
        test = pd.read_csv(prefix_pre_imputed(dataset) + f'holdout_{holdout_set}{miss_str}.csv')
        label = train.columns[-1]
        predictors = train.columns[:-1]

        ###########################
        ### Imputation approach ###
        ###########################
        if 'BREAST_CANCER' in dataset and val_set == 3 and holdout_set == 6: #missed run at this stage; will have to adjust for plotting as well
            continue
        if 'BREAST_CANCER_MAR_25' in dataset and val_set == 2 and holdout_set == 9: #missed run at this stage; will have to adjust for plotting as well
            continue
        imputation_train_probs = np.zeros((len(imputations), train.shape[0] + val.shape[0]))
        imputation_test_probs = np.zeros((len(imputations), test.shape[0]))

        best_lambdas = np.zeros(len(imputations))

        #find best lambda for each imputation, using validation set. 
            #currently, selects best auc using only one validation fold. 
            # one possible TODO is to select using 5-fold cross validation
            # (this may require verifying whether all 5 train/val splits use the same imputation)
            # using all folds for each imputation may increase the runtime of the full train/val/test pipeline non-trivially. 
        for imputation_idx, imputation in enumerate(imputations):
            X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_binarized(
                label, predictors, train, test, val, num_quantiles, 
                path_to_imputed(dataset, holdout_set, val_set, imputation), 
                validation=True) #decides quantiles using train and val, not test
            
            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, 
                                       max_support_size=s_size)
            
            (_, _, val_probs, _, _) = eval_model(
                model, X_train, X_val, y_train, y_val, lambda_grid[0],
                METRIC_FN[metric]
                )
            ensembled_val_aucs = np.zeros(len(lambda_grid[0]))
            
            for i in range(len(lambda_grid[0])):
                ensembled_val_aucs[i] = METRIC_FN[mice_validation_metric](y_val, val_probs[i])
            best_lambda = np.argmax(ensembled_val_aucs) #not optimal runtime, but should not be an issue
            best_lambdas[imputation_idx] = best_lambda
            
            #now that we know the best lambda for each imputation, we can go through 
            # and find the probabilities for each imputation, while training on the full train set, 
            # using these validation-optimal lambdas.
            X_train, X_test, y_train, y_test = get_train_test_binarized(
                label, predictors, train, test, val, num_quantiles, 
                path_to_imputed(dataset, holdout_set, val_set, imputation))
            
            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, 
                                       num_lambda=None, num_gamma=None, max_support_size=s_size)#inefficient to search whole grid(TODO)
            
            (train_probs, _, test_probs, _, _) = eval_model(
                model, X_train, X_test, y_train, y_test, lambda_grid[0], 
                METRIC_FN[metric]
                )
            imputation_train_probs[imputation_idx] = train_probs[best_lambda]
            imputation_test_probs[imputation_idx] = test_probs[best_lambda]

        # calculate ensembled metric across all imputations: 
        ensembled_train_probs = imputation_train_probs.mean(axis=0)
        ensembled_test_probs = imputation_test_probs.mean(axis=0)

        imputation_ensemble_train_auc[val_set, holdout_set] = METRIC_FN[metric](y_train, ensembled_train_probs)

        imputation_ensemble_test_auc[val_set, holdout_set] = METRIC_FN[metric](y_test, ensembled_test_probs)

        ######################################
        ### Missingness Indicator Approach ###
        ######################################
        if val_set != validations[0]: #we can just use the first val_set; no need to rerun this for each validation
            continue

        if distinct_missingness: 
            train_d = pd.read_csv(prefix_pre_imputed(dataset)+f'distinct-missingness/devel_{holdout_set}_train_{val_set}{miss_str}.csv')
            val_d = pd.read_csv(prefix_pre_imputed(dataset)+f'distinct-missingness/devel_{holdout_set}_val_{val_set}{miss_str}.csv')
            test_d = pd.read_csv(prefix_pre_imputed(dataset) + f'distinct-missingness/holdout_{holdout_set}{miss_str}.csv')
            (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
            y_train_no, y_train_ind, y_train_aug, 
            y_test_no, y_test_ind, y_test_aug) = binarize_and_augment_distinct(
                pd.concat([train_d, val_d]), test_d, np.linspace(0, 1, num_quantiles + 2)[1:-1], label, 
                miss_vals=[-7, -8, -9, -10] if not ('BREAST_CANCER' in dataset) else [-7, -8]
                )
            
        else: 
            (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
            y_train_no, y_train_ind, y_train_aug, 
            y_test_no, y_test_ind, y_test_aug) = binarize_and_augment(
                pd.concat([train, val]), test, np.linspace(0, 1, num_quantiles + 2)[1:-1], label
                )

        model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=s_size)
        (_, train_auc_no_missing[holdout_set], 
        _, test_auc_no_missing[holdout_set], sparsity_no_missing[holdout_set]) = eval_model(
            model_no, train_no, test_no, y_train_no, y_test_no, lambda_grid[0], METRIC_FN[metric]
            )
        
        model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=s_size)
        (_, train_auc_indicator[holdout_set], 
        _, test_auc_indicator[holdout_set], sparsity_indicator[holdout_set]) = eval_model(
            model_ind, train_ind, test_ind, y_train_ind, y_test_ind, lambda_grid[0], METRIC_FN[metric]
            )
        
        model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                                    lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=s_size)
        (_, train_auc_aug[holdout_set], 
        _, test_auc_aug[holdout_set], sparsity_aug[holdout_set]) = eval_model(
            model_aug, train_aug, test_aug, y_train_aug, y_test_aug, lambda_grid[0], 
            METRIC_FN[metric]
                )

#save data to csv: 
        
print('saving\n')

results_path = f'experiment_data/{dataset}{q_str}' if not distinct_missingness else f'experiment_data/{dataset}/distinct{q_str}'
if train_miss != 0 or test_miss != 0: 
    results_path = f'{results_path}/train_{train_miss}/test_{test_miss}'
if not os.path.exists(results_path):
    os.makedirs(results_path)

MICE_results_path = f'{results_path}/{s_size}'
if not os.path.exists(MICE_results_path):
    os.makedirs(MICE_results_path)

np.savetxt(f'{results_path}/train_{metric}_aug.csv', train_auc_aug)
np.savetxt(f'{results_path}/train_{metric}_indicator.csv', train_auc_indicator)
np.savetxt(f'{results_path}/train_{metric}_no_missing.csv', train_auc_no_missing)
np.savetxt(f'{results_path}/test_{metric}_aug.csv', test_auc_aug)
np.savetxt(f'{results_path}/test_{metric}_indicator.csv', test_auc_indicator)
np.savetxt(f'{results_path}/test_{metric}_no_missing.csv', test_auc_no_missing)
np.savetxt(f'{MICE_results_path}/imputation_ensemble_train_{metric}.csv', imputation_ensemble_train_auc)
np.savetxt(f'{MICE_results_path}/imputation_ensemble_test_{metric}.csv', imputation_ensemble_test_auc)
np.savetxt(f'{results_path}/nllambda.csv', -np.log(lambda_grid[0]))

np.savetxt(f'{results_path}/sparsity_aug.csv', sparsity_aug)
np.savetxt(f'{results_path}/sparsity_indicator.csv', sparsity_indicator)
np.savetxt(f'{results_path}/sparsity_no_missing.csv',sparsity_no_missing)


#can also try pickle file of all hyperparameters, and save to a folder with corresponding hash


print('successfully finished execution')
