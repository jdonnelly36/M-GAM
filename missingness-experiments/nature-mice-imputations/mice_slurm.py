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
num_quantiles = 4#8
lambda_grid = [[10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]]

holdouts = [0, 1]#[0, 1, 2]
validations = [0, 1]#[0, 1, 2, 3, 4]
imputations = 10


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

for holdout_set in holdouts: 
    for val_set in validations: 
        train = pd.read_csv(f'BREAST_CANCER/devel_{holdout_set}_train_{val_set}.csv')
        val = pd.read_csv(f'BREAST_CANCER/devel_{holdout_set}_val_{val_set}.csv')
        test = pd.read_csv(f'BREAST_CANCER/holdout_{holdout_set}.csv')
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
                f'test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/', 
                validation=True) #decides quantiles using train and val, not test
            
            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, 
                                       max_support_size=200)
            
            (_, _, val_probs, _, _) = eval_model(
                model, X_train, X_val, y_train, y_val, lambda_grid[0]
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
                f'test_per_0/holdout_{holdout_set}/val_{val_set}/m_{imputation}/')

            model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                       lambda_grid=lambda_grid, 
                                       num_lambda=None, num_gamma=None, max_support_size=200)#inefficient to search whole grid(TODO)
            
            (train_probs, _, test_probs, _, _) = eval_model(
                model, X_train, X_test, y_train, y_test, lambda_grid[0]
                )
            imputation_train_probs[imputation] = train_probs[best_lambda]
            imputation_test_probs[imputation] = test_probs[best_lambda]

        # calculate ensembled metric across all imputations: 
        ensembled_train_probs = imputation_train_probs.mean(axis=0)
        ensembled_test_probs = imputation_test_probs.mean(axis=0)

        fpr, tpr, _ = metrics.roc_curve(y_train, ensembled_train_probs)
        imputation_ensemble_train_auc[val_set, holdout_set] = metrics.auc(fpr, tpr)

        fpr, tpr, _ = metrics.roc_curve(y_test, ensembled_test_probs)
        imputation_ensemble_test_auc[val_set, holdout_set] = metrics.auc(fpr, tpr)

        ######################################
        ### Missingness Indicator Approach ###
        ######################################
        if val_set != validations[0]: #we can just use the first val_set; no need to rerun this for each validation
            continue

        (train_no, train_ind, train_aug, test_no, test_ind, test_aug, 
        y_train_no, y_train_ind, y_train_aug, 
        y_test_no, y_test_ind, y_test_aug) = binarize_and_augment(
            pd.concat([train, val]), test, np.linspace(0, 1, num_quantiles + 2)[1:-1] 
            )

        model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=200)
        _, train_auc_no_missing[holdout_set], _, test_auc_no_missing[holdout_set], _ = eval_model(
            model_no, train_no, test_no, y_train_no, y_test_no, lambda_grid[0]
            )
        
        model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                    num_lambda=None, num_gamma=None, max_support_size=200)
        _, train_auc_indicator[holdout_set], _, test_auc_indicator[holdout_set], _ = eval_model(
            model_ind, train_ind, test_ind, y_train_ind, y_test_ind, lambda_grid[0]
            )
        
        model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                                    lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=50)
        _, train_auc_aug[holdout_set], _, test_auc_aug[holdout_set], _ = eval_model(
            model_aug, train_aug, test_aug, y_train_aug, y_test_aug, lambda_grid[0]
            )

#TODO: collect array of probs/aucs for indicator method and then use errors from utils to plot error bars from our approach. 
# also fix the MICE reference to be a flat bar calculated correctly, rather than varying with lambda. 
# And print coeffs after you're done with that. 

nllambda = -np.log(lambda_grid[0])

plt.title('Train AUC vs Negative Log Lambda_0')
plt.scatter(nllambda, imputation_ensemble_train_auc.mean()*np.ones(len(nllambda)), label='ensemble of 10 MICE imputations')
plt.scatter(nllambda, train_auc_no_missing.mean(axis=0), label='No missingness handling (single run)')
plt.scatter(nllambda, train_auc_indicator.mean(axis=0), label='Missingness indicators (single run)')
plt.scatter(nllambda, train_auc_aug.mean(axis=0), label='Missingness with interactions (single run)')
plt.xlabel('Negative Log Lambda_0')
plt.ylabel('Train AUC')
plt.legend()
# plt.ylim(0.7, 0.85)
plt.savefig('./figs/mice_slurm_train_AUC.png')

plt.clf()

plt.title('Test AUC vs Negative Log Lambda_0')
plt.plot(nllambda, imputation_ensemble_test_auc.mean()*np.ones(len(nllambda)), label='ensemble of 10 MICE imputations')
plt.plot(nllambda, test_auc_no_missing.mean(axis=0), label='No missingness handling (single run)')
plt.plot(nllambda, test_auc_indicator.mean(axis=0), label='Missingness indicators (single run)')
plt.plot(nllambda, test_auc_aug.mean(axis=0), label='Missingness with interactions (single run)')
plt.xlabel('Negative Log Lambda_0')
plt.ylabel('Test AUC')
plt.legend()
# plt.ylim(0.7, 0.8)
plt.savefig('./figs/mice_slurm_test_AUC.png')

print('successfully finished execution')
