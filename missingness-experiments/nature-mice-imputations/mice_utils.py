import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

sys.path.append('../smim_code/')
from imputation_models import MIM

def get_smim_dataset(X_train, y_train, X_test, alpha=0.05):
    # Performance for SMIM
    smim = MIM(features="dynamic", alpha=alpha)

    start = time.time()
    train_mask_feats = smim.fit_transform(X_train, y_train)
    test_mask_feats = smim.transform(X_test)
    end = time.time()

    return train_mask_feats, test_mask_feats, end-start

"""
Accesses files from the imputation data folder structure. 

Parameters: 
    path_prefix: 
        The folder containing the imputed data in .npy format.
    label:
        The index of the label column in the data.
    predictors:
        The names for the predictors in the data.
    train, test, val:
        The non-imputed dataframes for the train, test, and validation sets; needed for the label column.
"""
def return_imputation(
        path_prefix, label, predictors, train, test, val
        ):
    raw_imputed_train = np.load(path_prefix+'imputed_train_x.npy', allow_pickle=True)
    imputed_train = pd.DataFrame(raw_imputed_train, columns=predictors)
    imputed_train[label] = train[label]

    raw_imputed_val = np.load(path_prefix+'imputed_val_x.npy', allow_pickle=True)
    imputed_val = pd.DataFrame(raw_imputed_val, columns=predictors)
    imputed_val[label] = val[label]

    raw_imputed_test = np.load(path_prefix+'imputed_test_x.npy', allow_pickle=True)
    imputed_test = pd.DataFrame(raw_imputed_test, columns=predictors)
    imputed_test[label] = test[label]

    return imputed_train, imputed_val, imputed_test

"""
Binarizes the data according to the train set, and then binarizing the test set according to the same quantiles.

Parameters:
    train_df, test_df:
        The train and test dataframes.
    quantiles_for_binarizing:
        The quantiles to use for binarizing the data.
    label:
        The column name for the labels in the data.
"""
def binarize_according_to_train(train_df, test_df, quantiles_for_binarizing = [0.2, 0.4, 0.6, 0.8], label='Overall Survival Status'):
    n_train, d_train = train_df.shape
    n_test, d_test = test_df.shape
    train, test = {}, {}
    for c in train_df.columns:
        if c == label:
            continue
        for v in list(train_df[c].quantile(quantiles_for_binarizing).unique()):
            new_col_name = f'{c} <= {v}'

            new_row_train = np.zeros(n_train)
            new_row_train[train_df[c] <= v] = 1
            train[new_col_name] = new_row_train
            
            new_row_test = np.zeros(n_test)
            new_row_test[test_df[c] <= v] = 1
            test[new_col_name] = new_row_test
   
    train[label] = train_df[label]
    test[label] = test_df[label]
    train, test = pd.DataFrame(train), pd.DataFrame(test)
    X_train, y_train = train[([c for c in train.columns if c != label])], train[label]
    X_test, y_test = test[([c for c in train.columns if c != label])], test[label]
    return X_train.values, X_test.values, y_train.values, y_test.values

"""
Uses the above helpers to access binarized versions of the imputed data. 

Parameters:
    label:
        The index of the label column in the data.
    predictors:
        The names for the predictors in the data.
    train, test, val:
        The non-imputed dataframes for the train, test, and validation sets; needed for the label column.
    num_quantiles:
        The number of quantiles to use for binarizing the data.
    path_prefix:
        The folder containing the imputed data in .npy format.
    validation:
        Whether to return a validation set; if False, the validation and train sets are combined in the return values.
"""
def get_train_test_binarized(label, predictors, train, test, val,
        num_quantiles=8, path_prefix='test_per_0/holdout_0/val_0/m_0/', validation=False): 
    quantiles = np.linspace(0, 1, num_quantiles + 2)[1:-1]

    train, val, test = return_imputation(path_prefix, label, predictors, train, test, val)
    train = pd.concat([train, val])#can change to remove validation set entirely, if desired
    
    X_train, X_test, y_train, y_test = binarize_according_to_train(train, test, quantiles, label)

    if validation: 
        X_val = X_train[-val.shape[0]:].copy()
        X_train = X_train[:-val.shape[0]].copy()
        y_val = y_train[-val.shape[0]:].copy()
        y_train = y_train[:-val.shape[0]].copy()
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test

"""
Evaluates a fastsparsegams model. For each lambda value provided to fastsparsegams, reports 
the train/test performance under the provided metric, as well as the sparsity of the solution.  

Parameters:
    model:
        The model to evaluate.
    X_train, X_test, y_train, y_test:
        The train and test data.
    provided_lambdas:
        The lambda values provided to fastsparsegams for regularization.
    metric_fn:
        The metric to use for evaluation.
"""
def eval_model(model, X_train, X_test, y_train, y_test, provided_lambdas, metric_fn): 
    num_coeffs = np.zeros((len(provided_lambdas)))
    train_auc = np.zeros((len(provided_lambdas)))
    test_auc = np.zeros((len(provided_lambdas)))
    train_probs = np.zeros((len(provided_lambdas), X_train.shape[0]))
    test_probs = np.zeros((len(provided_lambdas), X_test.shape[0]))

    for lamby in model.lambda_0[0]:

        i = provided_lambdas.index(lamby)

        train_probs[i] = model.predict(X_train.astype(float),lambda_0=lamby).reshape(-1)
        train_auc[i] = metric_fn(y_train, train_probs[i])

        test_probs[i] = model.predict(X_test.astype(float),lambda_0=lamby).reshape(-1)
        test_auc[i] = metric_fn(y_test, test_probs[i])

        coeffs = (model.coeff(lambda_0=lamby).toarray().flatten())[1:] #first entry is intercept
        num_coeffs[i] = (coeffs != 0).sum()


    return train_probs, train_auc, test_probs, test_auc, num_coeffs

"""
Evaluates a fastsparsegams model. For each lambda value provided to fastsparsegams, reports 
the train/test performance under the provided metric, as well as the sparsity of the solution.  

Parameters:
    model:
        The model to evaluate.
    X_train, X_test, y_train, y_test:
        The train and test data.
    provided_lambdas:
        The lambda values provided to fastsparsegams for regularization.
    metric_fn:
        The metric to use for evaluation.
    cluster: 
        Object that maps from a column name to a set of feature indices that should be penalized once per occurrence, rather than using the default sparsity measure. 
"""
def eval_model_by_clusters(model, X_train, X_test, y_train, y_test, provided_lambdas, metric_fn, cluster): 
    num_coeffs = np.zeros((len(provided_lambdas)))
    train_auc = np.zeros((len(provided_lambdas)))
    test_auc = np.zeros((len(provided_lambdas)))
    train_probs = np.zeros((len(provided_lambdas), X_train.shape[0]))
    test_probs = np.zeros((len(provided_lambdas), X_test.shape[0]))
    num_features = np.zeros((len(provided_lambdas)))

    for lamby in model.lambda_0[0]:

        i = provided_lambdas.index(lamby)

        train_probs[i] = model.predict(X_train.astype(float),lambda_0=lamby).reshape(-1)
        train_auc[i] = metric_fn(y_train, train_probs[i])

        test_probs[i] = model.predict(X_test.astype(float),lambda_0=lamby).reshape(-1)
        test_auc[i] = metric_fn(y_test, test_probs[i])

        coeffs = (model.coeff(lambda_0=lamby).toarray().flatten())[1:] #first entry is intercept
        num_coeffs[i] = (coeffs != 0).sum()

        #TODO cluster sparsity add
        for c in cluster.keys(): 
            if (coeffs[cluster[c]] != 0).any(): 
                num_features[i] += 1

    return train_probs, train_auc, test_probs, test_auc, num_coeffs, num_features

"""
Plotting helper to compute error bars
"""
def errors(accs, axis=0, error_bar_type='standard error'):
    if error_bar_type == 'standard error': 
        standard_error = accs.std(axis=axis)/np.sqrt(accs.shape[axis]) #currently has no small sample size correction
        return standard_error #multiply by 2 for 95% CI if sample size large enough
    else: 
        print(f'Unsupported error bar type: {error_bar_type}. Must be range or confidence interval')
        return 'Error!'
    
"""
Plotting helper to plot uncertainty bands
TODO: tidy, move plotting code to another utils file
"""
def uncertainty_bands(x_mat, y_mat, label, facecolor = '#F0F8FF', color='C0', axis=0): 
    x = x_mat.mean(axis=axis)
    y = y_mat.mean(axis=axis)
    yerr = errors(y_mat, axis=0)
    xerr = errors(x_mat, axis=0)

    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', lw=1, label=label)
    # plt.fill_between(x, y-yerr, y+yerr, alpha = 0.4)

    # conservative bands: consider max xerr and yerr simultaneously
    # plt.fill_between(np.column_stack([x - xerr, x, x + xerr]).flatten(), 
    #                  np.column_stack([y-yerr, y-yerr, y-yerr]).flatten(), 
    #                  np.column_stack([y+yerr, y+yerr, y+yerr]).flatten(),
    #                  alpha=0.4)
    
    ordered_x = np.column_stack([x - xerr, x, x + xerr]).flatten()
    increasing_x_indices = np.argsort(ordered_x)
    ordered_y_lower = np.column_stack([y-yerr, y-yerr, y-yerr]).flatten()
    ordered_y_upper = np.column_stack([y+yerr, y+yerr, y+yerr]).flatten()
    for i in range(ordered_x.shape[0]):
        if (ordered_x[i] < ordered_x[:i]).any(): # if this item has smaller x value than for some preceding item
            #for every preceding x value in the list that was larger, 
            # we want to consider the minimum of its lower bound and ours
            for j in range(i): 
                if ordered_x[j] > ordered_x[i]: 
                    ordered_y_lower[i] = min(ordered_y_lower[i], ordered_y_lower[j])
        elif (ordered_x[i] > ordered_x[i:]).any(): #if this item has larger x val than for some following item
            # for every proceding item in the list that has a smaller x value, we want to consider the max of its upper bound and ours
            for j in range(i+1, ordered_x.shape[0]): 
                if ordered_x[j] < ordered_x[i]: 
                    ordered_y_upper[i] = max(ordered_y_upper[i], ordered_y_upper[j])
    plt.fill_between(ordered_x[increasing_x_indices], 
                     ordered_y_lower[increasing_x_indices], 
                     ordered_y_upper[increasing_x_indices],
                     alpha=0.4)
    
def uncertainty_bands_subplot(x_mat, y_mat, label, ax, facecolor = '#F0F8FF', color='C0', axis=0): 
    x = x_mat.mean(axis=axis)
    y = y_mat.mean(axis=axis)
    yerr = errors(y_mat, axis=0)
    xerr = errors(x_mat, axis=0)

    if label is not None: 
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', lw=1, label=label)
    else: 
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', lw=1)
    # plt.fill_between(x, y-yerr, y+yerr, alpha = 0.4)

    # conservative bands: consider max xerr and yerr simultaneously
    # plt.fill_between(np.column_stack([x - xerr, x, x + xerr]).flatten(), 
    #                  np.column_stack([y-yerr, y-yerr, y-yerr]).flatten(), 
    #                  np.column_stack([y+yerr, y+yerr, y+yerr]).flatten(),
    #                  alpha=0.4)
    
    ordered_x = np.column_stack([x - xerr, x, x + xerr]).flatten()
    increasing_x_indices = np.argsort(ordered_x)
    ordered_y_lower = np.column_stack([y-yerr, y-yerr, y-yerr]).flatten()
    ordered_y_upper = np.column_stack([y+yerr, y+yerr, y+yerr]).flatten()
    for i in range(ordered_x.shape[0]):
        if (ordered_x[i] < ordered_x[:i]).any(): # if this item has smaller x value than for some preceding item
            #for every preceding x value in the list that was larger, 
            # we want to consider the minimum of its lower bound and ours
            for j in range(i): 
                if ordered_x[j] > ordered_x[i]: 
                    ordered_y_lower[i] = min(ordered_y_lower[i], ordered_y_lower[j])
        elif (ordered_x[i] > ordered_x[i:]).any(): #if this item has larger x val than for some following item
            # for every proceding item in the list that has a smaller x value, we want to consider the max of its upper bound and ours
            for j in range(i+1, ordered_x.shape[0]): 
                if ordered_x[j] < ordered_x[i]: 
                    ordered_y_upper[i] = max(ordered_y_upper[i], ordered_y_upper[j])
    ax.fill_between(ordered_x[increasing_x_indices], 
                     ordered_y_lower[increasing_x_indices], 
                     ordered_y_upper[increasing_x_indices],
                     alpha=0.4)
    
"""
Plotting helper to plot uncertainty bands, for mice imputations()
TODO: refactor to be more general
"""
def uncertainty_bands_subplot_mice(ensemble_data, label, ax, facecolor = '#F0F8FF', color='C0', axis=0): 
    x = np.array([199, 202])
    y = np.array([np.nanmean(ensemble_data), np.nanmean(ensemble_data)])
    # by_val = ensemble_data.reshape([10, 5]).mean(axis=1)
    by_val = np.nanmean(ensemble_data.reshape([10, 5]), axis=1)
    yerr = by_val.std()/np.sqrt(10)

    if label is not None: 
        ax.errorbar(x, y, yerr=yerr, fmt='.', lw=1, label=label)
    else: 
        ax.errorbar(x, y, yerr=yerr, fmt='.', lw=1)
    ax.fill_between(x, y-yerr, y+yerr, alpha = 0.4)
