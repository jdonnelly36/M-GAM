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

#hyperparameters (TODO: set up with argparse)
num_quantiles = 8
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]

holdouts = np.arange(2)#[0, 1, 2]
validations = np.arange(5)#[0, 1, 2, 3, 4]
imputations = 10

dataset = 'FICO'#'BREAST_CANCER'

metric = 'acc'

def calc_auc(y, score): 
    fpr, tpr, _ = metrics.roc_curve(y, score)
    return metrics.auc(fpr, tpr)
METRIC_FN = {
    'acc': lambda y, score: np.mean(
        (score > 0.5) == y),
    'auprc': lambda y, score: metrics.average_precision_score(y, score),
    'auc': calc_auc
}


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
        train = pd.read_csv(f'{dataset}/devel_{holdout_set}_train_{val_set}.csv')
        val = pd.read_csv(f'{dataset}/devel_{holdout_set}_val_{val_set}.csv')
        test = pd.read_csv(f'{dataset}/holdout_{holdout_set}.csv')
        label = train.columns[-1]
        predictors = train.columns[:-1]

        ###########################
        ### Imputation approach ###
        ###########################
        imputation_train_probs = np.zeros((imputations, train.shape[0] + val.shape[0]))
        imputation_test_probs = np.zeros((imputations, test.shape[0]))

        best_lambdas = np.zeros(imputations)

        #find best lambda for each imputation, using validation set. 
            #currently, selects best auc using only one validation fold. 
            # one possible TODO is to select using 5-fold cross validation
            # (this may require verifying whether all 5 train/val splits use the same imputation)
            # using all folds for each imputation may increase the runtime of the full train/val/test pipeline non-trivially. 
        for imputation in range(imputations):
            X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_binarized(
                label, predictors, train, test, val, num_quantiles, 
                f'{dataset}/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/', 
                validation=True) #decides quantiles using train and val, not test
            
            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, 
                                       max_support_size=200)
            
            (_, _, val_probs, _, _) = eval_model(
                model, X_train, X_val, y_train, y_val, lambda_grid[0],
                METRIC_FN[metric]
                )
            ensembled_val_aucs = np.zeros(len(lambda_grid[0]))
            
            for i in range(len(lambda_grid[0])):
                fpr, tpr, _ = metrics.roc_curve(y_val, val_probs[i])
                ensembled_val_aucs[i] = metrics.auc(fpr, tpr)
            best_lambda = np.argmax(ensembled_val_aucs) #not optimal runtime, but should not be an issue
            best_lambdas[imputation] = best_lambda

            #now that we know the best lambda for each imputation, we can go through 
            # and find the probabilities for each imputation, while training on the full train set, 
            # using these validation-optimal lambdas.
            X_train, X_test, y_train, y_test = get_train_test_binarized(
                label, predictors, train, test, val, num_quantiles, 
                f'{dataset}/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/')

            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, 
                                       num_lambda=None, num_gamma=None, max_support_size=200)#inefficient to search whole grid(TODO)
            
            (train_probs, _, test_probs, _, _) = eval_model(
                model, X_train, X_test, y_train, y_test, lambda_grid[0], 
                METRIC_FN[metric]
                )
            imputation_train_probs[imputation] = train_probs[best_lambda]
            imputation_test_probs[imputation] = test_probs[best_lambda]

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

        (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
        y_train_no, y_train_ind, y_train_aug, 
        y_test_no, y_test_ind, y_test_aug) = binarize_and_augment(
            pd.concat([train, val]), test, np.linspace(0, 1, num_quantiles + 2)[1:-1], label
            )

        model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=200)
        (_, train_auc_no_missing[holdout_set], 
         _, test_auc_no_missing[holdout_set], sparsity_no_missing[holdout_set]) = eval_model(
            model_no, train_no, test_no, y_train_no, y_test_no, lambda_grid[0], METRIC_FN[metric]
            )
        
        model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=200)
        (_, train_auc_indicator[holdout_set], 
         _, test_auc_indicator[holdout_set], sparsity_indicator[holdout_set]) = eval_model(
            model_ind, train_ind, test_ind, y_train_ind, y_test_ind, lambda_grid[0], METRIC_FN[metric]
            )
        
        model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                                    lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=50)
        (_, train_auc_aug[holdout_set], 
         _, test_auc_aug[holdout_set], sparsity_aug[holdout_set]) = eval_model(
            model_aug, train_aug, test_aug, y_train_aug, y_test_aug, lambda_grid[0], 
            METRIC_FN[metric]
            )

#save data to csv: 

np.savetxt(f'experiment_data/{dataset}/train_{metric}_aug.csv', train_auc_aug)
np.savetxt(f'experiment_data/{dataset}/train_{metric}_indicator.csv', train_auc_indicator)
np.savetxt(f'experiment_data/{dataset}/train_{metric}_no_missing.csv', train_auc_no_missing)
np.savetxt(f'experiment_data/{dataset}/test_{metric}_aug.csv', test_auc_aug)
np.savetxt(f'experiment_data/{dataset}/test_{metric}_indicator.csv', test_auc_indicator)
np.savetxt(f'experiment_data/{dataset}/test_{metric}_no_missing.csv', test_auc_no_missing)
np.savetxt(f'experiment_data/{dataset}/imputation_ensemble_train_{metric}.csv', imputation_ensemble_train_auc)
np.savetxt(f'experiment_data/{dataset}/imputation_ensemble_test_{metric}.csv', imputation_ensemble_test_auc)
np.savetxt(f'experiment_data/{dataset}/nllambda.csv', -np.log(lambda_grid[0]))

np.savetxt(f'experiment_data/{dataset}/sparsity_aug.csv', sparsity_aug)
np.savetxt(f'experiment_data/{dataset}/sparsity_indicator.csv', sparsity_indicator)
np.savetxt(f'experiment_data/{dataset}/sparsity_no_missing.csv',sparsity_no_missing)


#can also try pickle file of all hyperparameters, and save to a folder with corresponding hash


print('successfully finished execution')
