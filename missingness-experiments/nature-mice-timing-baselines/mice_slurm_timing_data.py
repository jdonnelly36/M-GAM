#!/usr/xtmp/jcd97/environments/missing_repl_env/bin/python
#SBATCH --job-name=missing_data # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jcd97@duke.edu     # Where to send mail
#SBATCH --output=./parallelized_logs/missing_data_%j_%a.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --nodelist=linux[41-45],linux[47-50]
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=100gb                     # Job memory request
#not SBATCH  -x linux[41-60],gpu-compute[1-7]
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

import os
import sys
sys.path.append(os.getcwd())

import timeit
import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
import fastsparsegams
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV as GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm
import os
import xgboost as xgb

from datetime import date

sys.path.append('../nature-mice-imputations/')
from mice_utils import get_smim_dataset, return_imputation, eval_model, get_train_test_binarized
from binarizer import Binarizer

print("Started, importing")
#hyperparameters (TODO: set up with argparse)
num_quantiles = 8
lambda_grid = [[20, 10, 5, 2, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]]

holdouts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #np.arange(2)#
validations = [0, 1, 2, 3, 4] #np.arange(5)#
imputations = 10
max_support_size = 50

#dataset = '/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/DATA/SYNTHETIC/SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE'#'BREAST_CANCER'
#dataset_imp = '/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data/IMPUTED_DATA/SYNTHETIC/MICE/train_per_0.5'#'BREAST_CANCER'
#dataset_suffix = '_train_missing_0.5_test_missing_0.25'

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
'''imputation_ensemble_train_auc = np.zeros((len(validations), len(holdouts)))
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
timing_dict_impu = {
    'holdout_set': [],
    'val_set': [],
    'imputation': [],
    'fit_time': []
}
timing_dict = {
    'holdout_set': [],
    'val_set': [],
    'fit_time_no': [],
    'fit_time_ind': [],
    'fit_time_aug': []
}'''

def get_timing_and_accuracy_baselines(model_types, param_grids, dataset, dataset_imp, dataset_suffix,
                                    holdouts=[0,1,2,3,4,5,6,7,8,9], val_set=0, imputations=10, 
                                    dataset_name='Synthetic', imputation_method='MICE', val_metric='accuracy',
                                    use_smim=False):
    metric_dict = {
        'metric': [],
        'metric_value_train': [],
        'metric_value_test': [],
        'model_type': [],
        'missingness_handling': [],
        'holdout_set': [],
        'std_fit_time': [],
        'mean_fit_time': [],
        'num_imputations': [],
        'dataset': [],
        'use_smim': [],
        'smim_time': []
    }
    for model_index, model_initializer in enumerate(model_types):
        for holdout_set in holdouts:
            #try:
            #print(f"Holdout: {holdout_set}, model class: {model_initializer.__name__}")
            train = pd.read_csv(f'{dataset}/devel_{holdout_set}_train_{val_set}{dataset_suffix}.csv')
            val = pd.read_csv(f'{dataset}/devel_{holdout_set}_val_{val_set}{dataset_suffix}.csv')
            test = pd.read_csv(f'{dataset}/holdout_{holdout_set}{dataset_suffix}.csv')

            if 'PHARYNGITIS' in dataset:
                label = 'radt'
                predictors = list(set(train.columns) - set(['radt']))
                ordered_cols = predictors + [label]
                train = train[ordered_cols]
                val = val[ordered_cols]
                test = test[ordered_cols]
            else:
                label = train.columns[-1]
                predictors = train.columns[:-1]

            train_full = pd.concat([val, train])
            imputation_train_probs = np.zeros((imputations, train.shape[0] + val.shape[0]))
            imputation_test_probs = np.zeros((imputations, test.shape[0]))

            _, y_train = train_full[predictors], train_full[label]
            _, y_test = test[predictors], test[label]

            if use_smim:
                train_X_SMIM, test_X_SMIM, SMIM_time = get_smim_dataset(train_full[predictors], train_full[label], test[predictors])
            else:
                SMIM_time = 0

            fit_times = []
            print("Type of model init is", str(model_initializer))
            if 'xgboost' in str(model_initializer):
                try:
                    clf = GridSearchCV(estimator=model_initializer(), param_grid=param_grids[model_index], scoring=val_metric)
                    clf.fit(train_full[predictors], train_full[label])
                except Exception as e:
                    print("Acceptable error (xgboost): ", repr(e))
                fit_times.append(clf.cv_results_['mean_fit_time'][clf.best_index_])

                ensembled_train_probs = clf.predict_proba(train_full[predictors])[:, 1]
                ensembled_test_probs = clf.predict_proba(test[predictors])[:, 1]

            else:
                for imputation in range(imputations):
                    print("Label: ", label, "predictors: ", predictors)

                    train_imp, val_imp, test_imp = return_imputation(
                        f'{dataset_imp}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/', 
                        label, predictors, train, test, val)
                    train_imp = pd.concat([val_imp, train_imp])

                    X_train, y_train_imp = train_imp[predictors], train_imp[label]
                    X_test, y_test_imp = test_imp[predictors], test_imp[label]
                    if use_smim and train_X_SMIM.shape[1] > 0:
                        print(train_X_SMIM.shape)
                        print(np.isnan(train_X_SMIM).any())
                        print(X_train.values.shape)
                        X_train = pd.DataFrame(np.concatenate([X_train.values, train_X_SMIM], axis=1))
                        X_test = pd.DataFrame(np.concatenate([X_test.values, test_X_SMIM], axis=1))

                    assert np.linalg.norm(y_train_imp - y_train) < 1e-3, \
                        "Error: Label ordering is not consistent across imputations"
                    assert np.linalg.norm(y_test_imp - y_test) < 1e-3, \
                        "Error: Label ordering is not consistent across imputations"

                    try:
                        print("Starting try block")
                        clf = GridSearchCV(model_initializer(), param_grids[model_index], scoring=val_metric)
                        print("X_train.shape", X_train.shape, "y_train_imp.shape", y_train_imp.shape)
                        clf.fit(X_train, y_train_imp)
                        print("done try block")
                    except Exception as e:
                        print("Acceptable error: ", repr(e))
                    
                    print(f"clf.cv_results_: {clf.cv_results_}")
                    fit_times.append(clf.cv_results_['mean_fit_time'][clf.best_index_])

                    try:
                        imputation_train_probs[imputation] = clf.predict_proba(X_train)[:, 1]
                        imputation_test_probs[imputation] = clf.predict_proba(X_test)[:, 1]
                    except:
                        imputation_train_probs[imputation] = clf.predict(X_train)
                        imputation_test_probs[imputation] = clf.predict(X_test)

                # calculate ensembled metric across all imputations: 
                ensembled_train_probs = imputation_train_probs.mean(axis=0)
                ensembled_test_probs = imputation_test_probs.mean(axis=0)

            for metric in METRIC_FN:
                train_metric_val = METRIC_FN[metric](y_train, ensembled_train_probs)
                test_metric_val = METRIC_FN[metric](y_test, ensembled_test_probs)

                metric_dict['num_imputations'] = metric_dict['num_imputations'] + [imputations]
                metric_dict['metric'] = metric_dict['metric'] + [metric]
                metric_dict['metric_value_train'] = metric_dict['metric_value_train'] + [train_metric_val]
                metric_dict['metric_value_test'] = metric_dict['metric_value_test'] + [test_metric_val]
                metric_dict['model_type'] = metric_dict['model_type'] + [model_initializer.__name__]
                metric_dict['missingness_handling'] = metric_dict['missingness_handling'] + [imputation_method]
                metric_dict['holdout_set'] = metric_dict['holdout_set'] + [holdout_set]
                #TODO: Change the mean below to a sum, because right now we're being overly generous to the baselines
                metric_dict['mean_fit_time'] = metric_dict['mean_fit_time'] + [np.mean(fit_times)]
                metric_dict['std_fit_time'] = metric_dict['std_fit_time'] + [np.std(fit_times)]
                metric_dict['dataset'] = metric_dict['dataset'] + [dataset_name]
                metric_dict['use_smim'] = metric_dict['use_smim'] + [use_smim]
                metric_dict['smim_time'] = metric_dict['smim_time'] + [SMIM_time]


                if slurm_iter is None:
                    pd.DataFrame(metric_dict).to_csv(f'./full_baseline_results_many_clf_tmp_multi_{date.today()}_10_holdouts_{dataset_name}_{imputations}_imp_{imputation_method}_50_max_coef.csv',index=False)
                else:
                    pd.DataFrame(metric_dict).to_csv(f'./parallelized_results/tmp_multi_{date.today()}_iter_{slurm_iter}_{imputation_method}_imp_{imputations}_imp_all_50_max_coef.csv',index=False)
            #except:
            #    print(f"Except for holdout {holdout_set}")
    return pd.DataFrame(metric_dict)

def get_timing_and_accuracy_gams(dataset, dataset_imp, dataset_suffix,
                                    holdouts=[0,1,2,3,4,5,6,7,8,9], val_set=0, imputations=10, dataset_name='Synthetic',
                                    na_check=lambda x: x.isna(), imputation_method='MICE', val_metric='acc'):
    metric_dict = {
        'metric': [],
        'metric_value_train': [],
        'metric_value_test': [],
        'model_type': [],
        'missingness_handling': [],
        'holdout_set': [],
        'std_fit_time': [],
        'mean_fit_time': [],
        'num_imputations': [],
        'dataset': []
    }
    for holdout_set in holdouts: 
        #try:
        for val_set in validations: 
            print(f"Holdout: {holdout_set}, val: {val_set}")
            train = pd.read_csv(f'{dataset}/devel_{holdout_set}_train_{val_set}{dataset_suffix}.csv')
            val = pd.read_csv(f'{dataset}/devel_{holdout_set}_val_{val_set}{dataset_suffix}.csv')
            test = pd.read_csv(f'{dataset}/holdout_{holdout_set}{dataset_suffix}.csv')
            if 'PHARYNGITIS' in dataset:
                label = 'radt'
                predictors = list(set(train.columns) - set(['radt']))
                ordered_cols = predictors + [label]
                train = train[ordered_cols]
                val = val[ordered_cols]
                test = test[ordered_cols]
            else:
                label = train.columns[-1]
                predictors = train.columns[:-1]
            encoder = Binarizer(quantiles = np.linspace(0, 1, num_quantiles + 2)[1:-1], label=label, 
                                miss_vals=[-7, -8, -9] if dataset=='FICO' else [np.nan, -7, -8, -9, -10], 
                                overall_mi_intercept = False, overall_mi_ixn = False, 
                                specific_mi_intercept = True, specific_mi_ixn = True, 
                                numerical_cols = [col for col in train.columns], 
                                categorical_cols= []) 

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
            #fit_times = []
            for imputation in range(imputations):
                X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_binarized(
                    label, predictors, train, test, val, num_quantiles, 
                    f'{dataset_imp}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/', 
                    validation=True) #decides quantiles using train and val, not test
                
                
                model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                        lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, 
                                        max_support_size=max_support_size)

                
                (_, _, val_probs, _, _) = eval_model(
                    model, X_train, X_val, y_train, y_val, lambda_grid[0],
                    METRIC_FN[metric]
                    )
                ensembled_val_aucs = np.zeros(len(lambda_grid[0]))
                
                for i in range(len(lambda_grid[0])):
                    # FIXED THIS AS OF 1-29 11:37 PM
                    ensembled_val_aucs[i] = METRIC_FN[val_metric](y_val, val_probs[i])
                    #fpr, tpr, _ = metrics.roc_curve(y_val, val_probs[i])
                    #ensembled_val_aucs[i] = metrics.auc(fpr, tpr)
                print("ensembled_val_aucs: ", ensembled_val_aucs)
                best_lambda = np.argmax(ensembled_val_aucs) #not optimal runtime, but should not be an issue
                best_lambdas[imputation] = best_lambda

                #now that we know the best lambda for each imputation, we can go through 
                # and find the probabilities for each imputation, while training on the full train set, 
                # using these validation-optimal lambdas.
                X_train, X_test, y_train, y_test = get_train_test_binarized(
                    label, predictors, train, test, val, num_quantiles, 
                    f'{dataset_imp}/holdout_{holdout_set}/val_{val_set}/m_{imputation}/')

                start_time = timeit.default_timer()
                model = fastsparsegams.fit(X_train, y_train, loss="Exponential", algorithm="CDPSI", 
                                        lambda_grid=lambda_grid, 
                                        num_lambda=None, num_gamma=None, max_support_size=max_support_size)#inefficient to search whole grid(TODO)
                fit_time = timeit.default_timer() - start_time
                
                print("imputation_train_probs.shape", imputation_train_probs.shape)
                print("X_train.shape", X_train.shape)
                (train_probs, _, test_probs, _, _) = eval_model(
                    model, X_train, X_test, y_train, y_test, lambda_grid[0], 
                    METRIC_FN[metric]
                    )
                print("train_probs.shape", train_probs.shape)
                imputation_train_probs[imputation] = train_probs[best_lambda]
                imputation_test_probs[imputation] = test_probs[best_lambda]
                #fit_times.append(fit_time)
                

            # calculate ensembled metric across all imputations: 
            #ensembled_train_probs = imputation_train_probs.mean(axis=0)
            #ensembled_test_probs = imputation_test_probs.mean(axis=0)

            def log_metric_vals(cur_train_probs, cur_test_probs, y_train, 
                                y_test, fit_times, model_type):
                for metric in METRIC_FN:
                    train_metric_val = METRIC_FN[metric](y_train, cur_train_probs)
                    test_metric_val = METRIC_FN[metric](y_test, cur_test_probs)

                    metric_dict['num_imputations'] = metric_dict['num_imputations'] + [imputations]
                    metric_dict['metric'] = metric_dict['metric'] + [metric]
                    metric_dict['metric_value_train'] = metric_dict['metric_value_train'] + [train_metric_val]
                    metric_dict['metric_value_test'] = metric_dict['metric_value_test'] + [test_metric_val]
                    metric_dict['model_type'] = metric_dict['model_type'] + [model_type]
                    metric_dict['missingness_handling'] = metric_dict['missingness_handling'] + [imputation_method]
                    metric_dict['holdout_set'] = metric_dict['holdout_set'] + [holdout_set]
                    metric_dict['mean_fit_time'] = metric_dict['mean_fit_time'] + [np.mean(fit_times)]
                    metric_dict['std_fit_time'] = metric_dict['std_fit_time'] + [np.std(fit_times)]
                    metric_dict['dataset'] = metric_dict['dataset'] + [dataset_name]
            
            #log_metric_vals(ensembled_train_probs, ensembled_test_probs, y_train, y_test, fit_times, 'GAM_imputation')

            ######################################
            ### Missingness Indicator Approach ###
            ######################################
            #if val_set != validations[0]: #we can just use the first val_set; no need to rerun this for each validation
            #    continue

            (train_no_overall, train_ind_overall, train_aug_overall, test_no, test_ind, test_aug, 
            y_train_no_overall, y_train_ind_overall, y_train_aug_overall, 
            y_test_no, y_test_ind, y_test_aug, cluster_no, cluster_ind, cluster_aug)= encoder.binarize_and_augment(pd.concat([train, val]), test)
            print("train_no_overall.shape", train_no_overall.shape)
            print("train_ind_overall.shape", train_ind_overall.shape)
            print("train_aug_overall.shape", train_aug_overall.shape)

            # 1-29 11:41 Changing to stratified k-fold
            kf = KFold(n_splits=5)
            #kf = StratifiedKFold(n_splits=5)
            
            accs_no = np.zeros((5, len(lambda_grid[0])))
            accs_ind = np.zeros((5, len(lambda_grid[0])))
            accs_aug = np.zeros((5, len(lambda_grid[0])))
            for i, (train_index, test_index) in enumerate(kf.split(train_no_overall)):#, y_train_no_overall)):
                train_no, y_train_no = train_no_overall[train_index], y_train_no_overall[train_index]
                val_no, y_val_no = train_no_overall[test_index], y_train_no_overall[test_index]

                train_ind, y_train_ind = train_ind_overall[train_index], y_train_ind_overall[train_index]
                val_ind, y_val_ind = train_ind_overall[test_index], y_train_ind_overall[test_index]

                train_aug, y_train_aug = train_aug_overall[train_index], y_train_aug_overall[train_index]
                val_aug, y_val_aug = train_aug_overall[test_index], y_train_aug_overall[test_index]

                print(f"CV METRIC IS {val_metric}")
                
                model_no = fastsparsegams.fit(train_no, y_train_no, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                            num_lambda=None, num_gamma=None, max_support_size=max_support_size)
                (_, _, _, val_acc, _) = eval_model(
                    model_no, train_no, val_no, y_train_no, y_val_no, lambda_grid[0], METRIC_FN[val_metric]
                    )
                accs_no[i] = val_acc
                
                model_ind = fastsparsegams.fit(train_ind, y_train_ind, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, 
                                            num_lambda=None, num_gamma=None, max_support_size=max_support_size)
                (_, _, _, val_acc, _) = eval_model(
                    model_ind, train_ind, val_ind, y_train_ind, y_val_ind, lambda_grid[0], METRIC_FN[val_metric]
                    )
                accs_ind[i] = val_acc
                
                model_aug = fastsparsegams.fit(train_aug, y_train_aug, loss="Exponential", algorithm="CDPSI", 
                                            lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=max_support_size)
                (_, _, _, val_acc, _) = eval_model(
                    model_aug, train_aug, val_aug, y_train_aug, y_val_aug, lambda_grid[0], 
                    METRIC_FN[val_metric]
                    )
                accs_aug[i] = val_acc

            print("accs_no", accs_no)
            print("accs_ind", accs_ind)
            print("accs_aug", accs_aug)
            best_lam_no = np.argmax(accs_no.mean(axis=0))
            best_lam_ind = np.argmax(accs_ind.mean(axis=0))
            best_lam_aug = np.argmax(accs_aug.mean(axis=0))
            print("Best lambdas: ", best_lam_no, best_lam_ind, best_lam_aug)

            # TODO: Need to grab best lambda via CV, then eval that guy
            start_time = timeit.default_timer()
            model_no = fastsparsegams.fit(train_no_overall, y_train_no_overall, loss="Exponential", algorithm="CDPSI", lambda_grid=[[lambda_grid[0][best_lam_no]]], 
                                        num_lambda=None, num_gamma=None, max_support_size=max_support_size)
            fit_time_no = timeit.default_timer() - start_time
            (train_probs, _, test_probs, _, _) = eval_model(
                model_no, train_no_overall, test_no, y_train_no_overall, y_test_no, [lambda_grid[0][best_lam_no]], METRIC_FN[metric]
            )
            train_probs = train_probs[0]
            test_probs = test_probs[0]
            log_metric_vals(train_probs, test_probs, y_train_no_overall, y_test_no, fit_time_no, 'GAM_no_missing')
            
            start_time = timeit.default_timer()
            model_ind = fastsparsegams.fit(train_ind_overall, y_train_ind_overall, loss="Exponential", algorithm="CDPSI", lambda_grid=[[lambda_grid[0][best_lam_ind]]], 
                                        num_lambda=None, num_gamma=None, max_support_size=max_support_size)
            fit_time_ind = timeit.default_timer() - start_time
            (train_probs, _, test_probs, _, _) = eval_model(
                model_ind, train_ind_overall, test_ind, y_train_ind_overall, y_test_ind, [lambda_grid[0][best_lam_ind]], METRIC_FN[metric]
                )
            train_probs = train_probs[0]
            test_probs = test_probs[0]
            log_metric_vals(train_probs, test_probs, y_train_ind_overall, y_test_ind, fit_time_ind, 'GAM_ind')
            
            start_time = timeit.default_timer()
            model_aug = fastsparsegams.fit(train_aug_overall, y_train_aug_overall, loss="Exponential", algorithm="CDPSI", 
                                        lambda_grid=[[lambda_grid[0][best_lam_aug]]], num_lambda=None, num_gamma=None, max_support_size=max_support_size)
            fit_time_aug = timeit.default_timer() - start_time
            (train_probs, _, test_probs, _, _) = eval_model(
                model_aug, train_aug_overall, test_aug, y_train_aug_overall, y_test_aug, [lambda_grid[0][best_lam_aug]], 
                METRIC_FN[metric]
                )
            train_probs = train_probs[0]
            test_probs = test_probs[0]
            log_metric_vals(train_probs, test_probs, y_train_aug_overall, y_test_aug, fit_time_aug, 'GAM_aug')
            if slurm_iter is None:
                pd.DataFrame(metric_dict).to_csv(f'./full_baseline_results_many_clf_tmp_gam_{date.today()}_10_holdouts_{dataset_name}_{imputations}_imp_{imputation_method}_50_max_coef.csv',index=False)
            else:
                pd.DataFrame(metric_dict).to_csv(f'./parallelized_results/tmp_gam_{date.today()}_iter_{slurm_iter}_{imputation_method}_imp_{imputations}_imp_all_50_max_coef.csv',index=False)
        #except:
        #    print("Except for", holdout_set, val_set)
        #    continue
    return pd.DataFrame(metric_dict)


if __name__ == '__main__':
    np.random.seed(0)
    model_types = [
        LogisticRegression,
        RandomForestClassifier,
        AdaBoostClassifier,
        DecisionTreeClassifier,
        MLPClassifier,
        xgb.sklearn.XGBClassifier
        #KNeighborsClassifier
    ]

    param_grids = [
        {'C':[0.01, 0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [5_000], 'tol': [1e-2]},#, 'elasticnet')},
        {'n_estimators':[25, 50, 100, 200], 'criterion':("gini", "entropy")},
        {'n_estimators':[10, 25, 50, 100, 200], 'algorithm': ['SAMME']},
        {'max_depth':[3, 5, 7, 9, None], 'criterion':("gini", "entropy")},
        {
            'hidden_layer_sizes':[(50,), (100,), (200,), (50, 50), (50, 100), (100, 100), (100, 200), (200, 200)],
            'tol': [1e-3],
            'max_iter': [1000]
        },
        {
            'n_estimators':[100, 500, 1000],
            'gamma':[0, 0.1],
            'lambda':[.5, 1, 2],
            'alpha':[.5, 1, 2]
            #Other parameters
        }
        #{'n_neighbors':[1, 3, 5, 7, 11], 'p': [1, 2]}
    ]


    if os.environ['SLURM_ARRAY_TASK_ID'] is not None:
        print(os.environ['SLURM_ARRAY_TASK_ID'])
        print(os.environ['SLURM_ARRAY_TASK_COUNT'])
        slurm_iter = int(os.environ['SLURM_ARRAY_TASK_ID'])
        #assert int(os.environ['SLURM_ARRAY_TASK_COUNT']) == len(param_grids)*len(holdouts)
    else:
        slurm_iter = None
        

    dataset_base_path = '/home/users/jcd97/code/missing_data/fastsparse_take_3/fastsparsemissing/handling_missing_data'
    overall_df = None

    imputations = 10
    
    """
    Have completed:
    for imputation_method in ['Mean', 'MICE', 'MissForest']:
        for ds_name in ['CKD', 'HEART_DISEASE', 'FICO', 'BREAST_CANCER', 'PHARYNGITIS']:

    for imputation_method in ['Mean', 'MICE', 'MissForest',]:
        for subsample in ['_0.25', '_0.5', '_0.75']:
            for ds_name in ['CKD', 'HEART_DISEASE', 'BREAST_CANCER', 'PHARYNGITIS']:

    Have running:
    for imputation_method in ['MIWAE]:
        for subsample in ['', '_0.25', '_0.5', '_0.75']:
            for ds_name in ['CKD', 'HEART_DISEASE', 'PHARYNGITIS', 'FICO']:

    Need to run:

    for imputation_method in ['Mean', 'MICE', 'MissForest',]:
        for subsample in ['_0.25', '_0.5', '_0.75']:
            for ds_name in ['FICO']:

    for imputation_method in ['Mean', 'MICE', 'MissForest', 'MIWAE']:
        for subsample in ['', '_0.25', '_0.5', '_0.75']:
            for ds_name in ['ADULT', 'BREAST_CANCER', 'MIMIC']:
    """
    for imputation_method in ['Mean', 'MICE', 'MissForest', 'MIWAE']:
        for subsample in ['_0.5', '_0.75', '', '_0.25']:
            for ds_name in ['PHARYNGITIS', 'FICO', 'HEART_DISEASE', 'BREAST_CANCER', 'MIMIC', 'ADULT', 'CKD']:
                ds_name = ds_name + subsample
                print(f"Running for {imputation_method}, {ds_name}")
                #for ds_name in ['BREAST_CANCER', 'BREAST_CANCER_0.25', 'BREAST_CANCER_0.5', 'BREAST_CANCER_0.75']:
                if False and 'FICO' == ds_name:
                    dataset = f'{dataset_base_path}/DATA/{ds_name}/distinct-missingness/'#'BREAST_CANCER'
                else:
                    dataset = f'{dataset_base_path}/DATA_REDUCED/{ds_name}/'#'BREAST_CANCER'
                for train_rate in [0]:
                    for test_rate in [0]:
                        if os.path.exists(f'./parallelized_results/baselines_iter_{slurm_iter}_{ds_name}_{imputation_method}.csv'):
                            continue
                        dataset_imp = f'{dataset_base_path}/JON_IMPUTED_DATA/{ds_name}/{imputation_method}/train_per_{train_rate}/test_per_{test_rate}'#'BREAST_CANCER'
                        dataset_suffix = f''

                        if 'FICO' in ds_name:
                            na_check = lambda x: x < -5
                        else:
                            na_check = lambda x: x.isna()

                        if slurm_iter is None:
                            gam_res_df = get_timing_and_accuracy_gams(dataset, dataset_imp, dataset_suffix,
                                        dataset_name=f'{ds_name}', imputations=imputations, na_check=na_check, imputation_method=imputation_method,
                                        val_metric='auc' if 'BREAST' in ds_name else 'acc')
                            res_df = get_timing_and_accuracy_baselines(model_types, param_grids, dataset, dataset_imp, dataset_suffix,
                                dataset_name=f'{ds_name}', imputation_method=imputation_method, imputations=imputations,
                                val_metric='roc_auc' if 'BREAST' in ds_name else 'accuracy')
                        else:

                            
                            res_df = pd.DataFrame()
                            model_ind = (slurm_iter // len(holdouts)) % len(model_types)
                            res_df = get_timing_and_accuracy_baselines([model_types[model_ind]], 
                                        [param_grids[model_ind]], dataset, dataset_imp, dataset_suffix,
                                        dataset_name=f'{ds_name}', imputation_method=imputation_method, 
                                        imputations=imputations, holdouts=[holdouts[slurm_iter % len(holdouts)]],
                                        val_metric='roc_auc' if 'BREAST' in ds_name else 'accuracy',
                                        use_smim=(slurm_iter // (len(holdouts) * len(model_types))) < 1)

                            # We only want to run the GAM bit num_holdouts times
                            # rather than num_holdouts * num_baselines times
                            if slurm_iter // len(holdouts) == 0:
                                gam_res_df = get_timing_and_accuracy_gams(dataset, dataset_imp, dataset_suffix,
                                            dataset_name=f'{ds_name}', imputations=imputations, 
                                            na_check=na_check, imputation_method=imputation_method,
                                            holdouts=[holdouts[slurm_iter % len(holdouts)]],
                                            val_metric='auc' if 'BREAST' in ds_name else 'acc')
                            else:
                                gam_res_df = pd.DataFrame()


                        new_res = pd.concat([res_df, gam_res_df], axis=0)
                        
                        if slurm_iter is not None:
                            new_res.to_csv(f'./parallelized_results/baselines_iter_{slurm_iter}_{ds_name}_{imputation_method}.csv',index=False)

                        if overall_df is None:
                            overall_df = new_res
                        else:
                            print("res_df: ", res_df)
                            print("overall_df: ", overall_df)
                            overall_df = pd.concat([overall_df, new_res], axis=0)

                        if slurm_iter is None:
                            overall_df.to_csv(f'./full_baseline_results_many_clf_{date.today()}_10_holdouts_{imputation_method}_imp_{imputations}_imp_50_max_coef.csv',index=False)
                        else:
                            overall_df.to_csv(f'./parallelized_results/baselines_{date.today()}_iter_{slurm_iter}_{imputations}_imp_all_50_max_coef.csv',index=False)

    '''for fico_vers in ['FICO_0.25', 'FICO_0.5', 'FICO_0.75']:
        dataset = f'{dataset_base_path}/DATA_REDUCED/{fico_vers}/'#'BREAST_CANCER'
        for train_rate in [0]:
            for test_rate in [0]:
                dataset_imp = f'{dataset_base_path}/JON_IMPUTED_DATA/{fico_vers}/MICE/train_per_{train_rate}/test_per_{test_rate}'#'BREAST_CANCER'
                dataset_suffix = f''
                gam_res_df = get_timing_and_accuracy_gams(dataset, dataset_imp, dataset_suffix,
                            dataset_name=f'{fico_vers}')
                res_df = get_timing_and_accuracy_baselines(model_types, param_grids, dataset, dataset_imp, dataset_suffix,
                    dataset_name=f'{fico_vers}')
                if overall_df is None:
                    overall_df = pd.concat([res_df, gam_res_df], axis=0)
                else:
                    overall_df = pd.concat([overall_df, res_df, gam_res_df], axis=0)
                overall_df.to_csv(f'./full_baseline_results_many_clf_{date.today()}_10_holdouts_f_sub.csv',index=False)'''


            

    #save data to csv: 

    '''np.savetxt(f'experiment_data/{dataset}/train_{metric}_aug.csv', train_auc_aug)
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
    np.savetxt(f'experiment_data/{dataset}/sparsity_no_missing.csv',sparsity_no_missing)'''


    #can also try pickle file of all hyperparameters, and save to a folder with corresponding hash


    print('successfully finished execution')