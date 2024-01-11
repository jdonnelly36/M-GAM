import pandas as pd
import numpy as np
from sklearn import metrics

def return_imputation(
        path_prefix, label, predictors, train, test, val
        ):
    raw_imputed_train = np.load(path_prefix+'imputed_train_x.npy')
    imputed_train = pd.DataFrame(raw_imputed_train, columns=predictors)
    imputed_train[label] = train[label]

    raw_imputed_val = np.load(path_prefix+'imputed_val_x.npy')
    imputed_val = pd.DataFrame(raw_imputed_val, columns=predictors)
    imputed_val[label] = val[label]

    raw_imputed_test = np.load(path_prefix+'imputed_test_x.npy')
    imputed_test = pd.DataFrame(raw_imputed_test, columns=predictors)
    imputed_test[label] = test[label]

    return imputed_train, imputed_val, imputed_test

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

def binarize_and_augment(train_df, test_df, quantiles_for_binarizing = [0.2, 0.4, 0.6, 0.8], label='Overall Survival Status'):
    n_train, d_train = train_df.shape
    n_test, d_test = test_df.shape
    train_binned, train_augmented_binned, test_binned, test_augmented_binned = {}, {}, {}, {}
    train_no_missing, test_no_missing = {}, {}
    for c in train_df.columns:
        if c == label:
            continue
        missing_col_name = f'{c} missing'
        missing_row_train = np.zeros(n_train)
        missing_row_test = np.zeros(n_test)
        for v in list(train_df[c].quantile(quantiles_for_binarizing).unique()):
            new_col_name = f'{c} <= {v}'

            new_row_train = np.zeros(n_train)
            new_row_train[train_df[c] <= v] = 1
            train_no_missing[new_col_name] = new_row_train
            train_binned[new_col_name] = new_row_train
            train_augmented_binned[new_col_name] = new_row_train
            
            new_row_test = np.zeros(n_test)
            new_row_test[test_df[c] <= v] = 1
            test_no_missing[new_col_name] = new_row_test
            test_binned[new_col_name] = new_row_test
            test_augmented_binned[new_col_name] = new_row_test

        missing_row_train[train_df[c].isna()] = 1
        missing_row_test[test_df[c].isna()] = 1

        train_binned[missing_col_name] = missing_row_train
        train_augmented_binned[missing_col_name] = missing_row_train
    
        test_binned[missing_col_name] = missing_row_test
        test_augmented_binned[missing_col_name] = missing_row_test
    
    for c_outer in train_df.columns:
        if c_outer == label:
            continue
        for c_inner in train_df.columns:
            for v in train_df[c_inner].quantile(quantiles_for_binarizing).unique():
                if c_inner == label:
                    continue
                else:
                    missing_ixn_name = f'{c_outer} missing & {c_inner} <= {v}'
                    missing_ixn_row_train = np.zeros(n_train)
                    missing_ixn_row_test = np.zeros(n_test)

                    missing_ixn_row_train[(train_df[c_outer].isna()) & (train_df[c_inner] <= v)] = 1
                    missing_ixn_row_test[(test_df[c_outer].isna()) & (test_df[c_inner] <= v)] = 1

                    train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                    test_augmented_binned[missing_ixn_name] = missing_ixn_row_test
                        
    train_binned[label] = train_df[label]
    test_binned[label] = test_df[label]
    train_no_missing[label] = train_df[label]
    test_no_missing[label] = test_df[label]
    train_augmented_binned[label] = train_df[label]
    test_augmented_binned[label] = test_df[label]

    
    return (pd.DataFrame(train_no_missing)[[c for c in train_no_missing.keys() if c != label]].values, 
            pd.DataFrame(train_binned)[[c for c in train_binned.keys() if c != label]].values, 
            pd.DataFrame(train_augmented_binned)[[c for c in train_augmented_binned.keys() if c != label]].values,
            pd.DataFrame(test_no_missing)[[c for c in test_no_missing.keys() if c != label]].values, 
            pd.DataFrame(test_binned)[[c for c in test_binned.keys() if c != label]].values, 
            pd.DataFrame(test_augmented_binned)[[c for c in test_augmented_binned.keys() if c != label]].values, 
            pd.DataFrame(train_no_missing)[label].values, 
            pd.DataFrame(train_binned)[label].values, 
            pd.DataFrame(train_augmented_binned)[label].values,
            pd.DataFrame(test_no_missing)[label].values, 
            pd.DataFrame(test_binned)[label].values, 
            pd.DataFrame(test_augmented_binned)[label].values, 
    )

def errors(accs, axis=0, error_bar_type='standard error'):
    if error_bar_type == 'standard error': 
        standard_error = accs.std(axis=axis)/np.sqrt(accs.shape[axis]) #currently has no small sample size correction
        return standard_error #multiply by 2 for 95% CI if sample size large enough
    else: 
        print(f'Unsupported error bar type: {error_bar_type}. Must be range or confidence interval')
        return 'Error!'