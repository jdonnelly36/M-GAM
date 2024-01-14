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
from mice_utils import return_imputation, binarize_according_to_train, eval_model, get_train_test_binarized, binarize_and_augment, errors, binarize_and_augment_median, binarize_and_augment_mean, binarize_and_augment_imputation

#hyperparameters (TODO: set up with argparse)
num_quantiles = 8
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]

holdouts = np.arange(5)#[0, 1, 2]
validations = np.arange(10)#[0, 1, 2, 3, 4]
imputations = 10

dataset = 'FICO'

metric = 'acc'
mice_validation_metric = 'acc'
s_size=150

ablation = 'mean'

def calc_auc(y, score): 
    fpr, tpr, _ = metrics.roc_curve(y, score)
    return metrics.auc(fpr, tpr)
METRIC_FN = {
    'acc': lambda y, score: np.mean(
        (score > 0.5) == y),
    'auprc': lambda y, score: metrics.average_precision_score(y, score),
    'auc': calc_auc, 
    'loss': lambda y, probs: np.exp(np.log(probs/(1 - probs))*-1/2*y).mean()
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

        ######################################
        ### Missingness Indicator Approach ###
        ######################################
        if ablation == 'median' or ablation == 'mean':
            if val_set != validations[0]: #we can just use the first val_set; no need to rerun this for each validation
                continue

            if ablation == 'median': 
                (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
                y_train_no, y_train_ind, y_train_aug, 
                y_test_no, y_test_ind, y_test_aug) = binarize_and_augment_median(
                    pd.concat([train, val]), test, np.linspace(0, 1, num_quantiles + 2)[1:-1], label
                    )
            else:
                (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
                y_train_no, y_train_ind, y_train_aug, 
                y_test_no, y_test_ind, y_test_aug) = binarize_and_augment_mean(
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
        ###########################
        ### Imputation approach ###
        ###########################
        if 'MICE' in ablation: 
            #variables for performance of indicator methods, across each_holdout, for each lambda
            train_prob_aug_i = np.zeros((imputations, len(lambda_grid[0]), train.shape[0] + val.shape[0]))
            train_prob_indicator_i = np.zeros((imputations, len(lambda_grid[0]), train.shape[0] + val.shape[0]))
            train_prob_no_missing_i = np.zeros((imputations, len(lambda_grid[0]), train.shape[0] + val.shape[0]))
            test_prob_aug_i = np.zeros((imputations, len(lambda_grid[0]), test.shape[0]))
            test_prob_indicator_i = np.zeros((imputations, len(lambda_grid[0]), test.shape[0]))
            test_prob_no_missing_i = np.zeros((imputations, len(lambda_grid[0]), test.shape[0]))

            #sparsity for each lambda, holdout, and method
            sparsity_aug_i = np.zeros((imputations, len(lambda_grid[0])))
            sparsity_indicator_i = np.zeros((imputations, len(lambda_grid[0])))
            sparsity_no_missing_i = np.zeros((imputations, len(lambda_grid[0])))

            #find best lambda for each imputation, using validation set. 
                #currently, selects best auc using only one validation fold. 
                # one possible TODO is to select using 5-fold cross validation
                # (this may require verifying whether all 5 train/val splits use the same imputation)
                # using all folds for each imputation may increase the runtime of the full train/val/test pipeline non-trivially. 
            for imputation in range(imputations):
                imputed_train, imputed_val, imputed_test = return_imputation(
                    f'{dataset}/test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/', 
                    label, predictors, train, test, val)
                    
                (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
                y_train_no, y_train_ind, y_train_aug, 
                y_test_no, y_test_ind, y_test_aug) = binarize_and_augment_imputation(
                    pd.concat([train, val]), test, pd.concat([imputed_train, imputed_val]), 
                    imputed_test, np.linspace(0, 1, num_quantiles + 2)[1:-1], label
                    )

                model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                            num_lambda=None, num_gamma=None, max_support_size=s_size)
                (train_prob_no_missing_i[imputation], _, 
                test_prob_no_missing_i[imputation], _, sparsity_no_missing_i[imputation]) = eval_model(
                    model_no, train_no, test_no, y_train_no, y_test_no, lambda_grid[0], METRIC_FN[metric]
                    )
                
                model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                            num_lambda=None, num_gamma=None, max_support_size=s_size)
                (train_prob_indicator_i[imputation], _,
                test_prob_indicator_i[imputation], _, sparsity_indicator_i[imputation]) = eval_model(
                    model_ind, train_ind, test_ind, y_train_ind, y_test_ind, lambda_grid[0], METRIC_FN[metric]
                    )
                
                model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                                            lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=s_size)
                (train_prob_aug_i[imputation], _, 
                test_prob_aug_i[imputation], _, sparsity_aug_i[imputation]) = eval_model(
                    model_aug, train_aug, test_aug, y_train_aug, y_test_aug, lambda_grid[0], 
                    METRIC_FN[metric]
                    )
                ############
                ###just going to do an average, not a cross validation lambda threshold selecting for each lambda in mice average. 
                ### TODO: check benefit from adding this in (allowing averaging across one model per imputation, with a different lambda for each imputation)
                ############

            # calculate ensembled metric across all imputations: 
            ensembled_train_probs_no_missing = train_prob_no_missing_i.mean(axis=0)
            ensembled_test_probs_no_missing = test_prob_no_missing_i.mean(axis=0)
            for idx, lammy in enumerate(lambda_grid[0]):
                train_auc_no_missing[holdout_set, idx] = METRIC_FN[metric](y_train_no, ensembled_train_probs_no_missing[idx]) if (sparsity_no_missing_i[:, idx] > 0).all() else 0
                test_auc_no_missing[holdout_set, idx] = METRIC_FN[metric](y_test_no, ensembled_test_probs_no_missing[idx])
            sparsity_no_missing[holdout_set] = sparsity_no_missing_i.sum(axis=0) #rough approximation of sparsity
            
            ensembled_train_probs_indicator = train_prob_indicator_i.mean(axis=0)
            ensembled_test_probs_indicator = test_prob_indicator_i.mean(axis=0)
            for idx, lammy in enumerate(lambda_grid[0]):
                train_auc_indicator[holdout_set, idx] = METRIC_FN[metric](y_train_ind, ensembled_train_probs_indicator[idx]) if (sparsity_indicator_i[:, idx] > 0).all() else 0
                test_auc_indicator[holdout_set, idx] = METRIC_FN[metric](y_test_ind, ensembled_test_probs_indicator[idx])
            sparsity_indicator[holdout_set] = sparsity_indicator_i.sum(axis=0)

            ensembled_train_probs_aug = train_prob_aug_i.mean(axis=0)
            ensembled_test_probs_aug = test_prob_aug_i.mean(axis=0)
            for idx, lammy in enumerate(lambda_grid[0]):
                train_auc_aug[holdout_set, idx] = METRIC_FN[metric](y_train_aug, ensembled_train_probs_aug[idx]) if (sparsity_aug_i[:, idx] > 0).all() else 0
                test_auc_aug[holdout_set, idx] = METRIC_FN[metric](y_test_aug, ensembled_test_probs_aug[idx])
            sparsity_aug[holdout_set] = sparsity_aug_i.sum(axis=0)
            
#save data to csv: 

np.savetxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_aug.csv', train_auc_aug)
np.savetxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_indicator.csv', train_auc_indicator)
np.savetxt(f'experiment_data/{dataset}/{ablation}/train_{metric}_no_missing.csv', train_auc_no_missing)
np.savetxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_aug.csv', test_auc_aug)
np.savetxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_indicator.csv', test_auc_indicator)
np.savetxt(f'experiment_data/{dataset}/{ablation}/test_{metric}_no_missing.csv', test_auc_no_missing)
np.savetxt(f'experiment_data/{dataset}/{ablation}/nllambda.csv', -np.log(lambda_grid[0]))

np.savetxt(f'experiment_data/{dataset}/{ablation}/sparsity_aug.csv', sparsity_aug)
np.savetxt(f'experiment_data/{dataset}/{ablation}/sparsity_indicator.csv', sparsity_indicator)
np.savetxt(f'experiment_data/{dataset}/{ablation}/sparsity_no_missing.csv',sparsity_no_missing)


#can also try pickle file of all hyperparameters, and save to a folder with corresponding hash


print('successfully finished execution')
