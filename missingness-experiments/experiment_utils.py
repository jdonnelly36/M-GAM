import fastsparsegams
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.utils import resample

def generate_data(num_samples=10, num_features=5, 
                    dgp_function=None, noise_scale=1):
    '''
    This function produces synthetic binary data and the corresponding
    labels with the specified size and optionally following a specified
    DGP.

    Parameters
    ----------
    num_samples : int, optional
        The number of samples to include in the dataset
    num_features : int, optional
        The number of binary features to include in the dataset
    dgp_function : function, optional
        A function that takes a matrix of binary features and produces
        a vector of binary labels
    noise_scale : float, optional
        The standard deviation to use when adding gaussian noise


    Returns
    -------
    X : array (num_samples, num_features)
        A binary matrix of features for the generated data
    y : array (num_samples)
        A binary vector of labels for the generated data

    '''
    ## Generate synthetic, binary data. 
    # first 10 features are used, none others. Normal noise, features all random integers
    # (documentation of this setup can be found at https://tnonet.github.io/L0Learn/tutorial.html
    #  and this setup is also used in the tutorial example for fastsparsegams)
    X = np.random.normal(size=(num_samples, num_features))

    if dgp_function is None:
        B = np.random.normal(size=(num_features,))
        #B[: (k // 2)] = 1
        #B[(k // 2) : k] = -1
        # Add gaussian noise 
        e = np.random.normal(size=(num_samples,), scale=noise_scale)
        y = np.sign(X @ B + 1e-10 + e)
    else:
        # Add gaussian noise 
        e = np.random.normal(size=(num_samples,), scale=noise_scale)
        y = np.sign(dgp_function(X) + 1e-10 + e)
    y[y == 0] = 1

    return X, y.astype(int)

def bin_data(X, bin_type='overlapping_upper'):
    '''
    This function produces synthetic binary data and the corresponding
    labels with the specified size and optionally following a specified
    DGP.

    Parameters
    ----------
    X : array (num_samples, num_features)
        The original, unbinarized features to bin
    bin_type : str, optional
        The binning strategy to use; should be in {'overlapping_upper'}


    Returns
    -------
    X_binned : array (num_samples, num_binned_features)
        A binary matrix of features for the generated data
    column_labels : list (num_binned_features)
        The name of each column in the returned data
    '''
    supported_strategies = ['overlapping_upper']
    assert bin_type in supported_strategies, f"Error: Binning strategy {bin_type} is not supported"

    X_binned_list_of_lists = []
    col_names = []
    for column in range(X.shape[-1]):
        unique_vals = np.unique(X[:, column])

        # Create a bin for being less than halfway
        # between each pair of sorted values
        for ind in range(len(unique_vals) - 1):
            middle_val = (unique_vals[ind] +  unique_vals[ind+1])/2
            new_col = np.zeros(X.shape[0])
            new_col[X[:, column] <= middle_val] = 1

            X_binned_list_of_lists.append(new_col)
            col_names.append(f"col_{column}<={middle_val}")

    X_binned = np.array(X_binned_list_of_lists)
    return X_binned, col_names

def obfuscate_data(X, obfuscation_rate=0.2, missingness_model='MCAR'):
    '''
    This function randomly removes entries from a feature matrix
    with the given probabilty and following the specified missingness
    model

    Parameters
    ----------
    X : matrix (num_samples, num_features)
        The original unperturbed input features
    obfuscation_rate : float, optional
        The probability to remove each entry in X
    missingness_model : function: matrix (num_samples, num_features) ->  
                        binary matrix (num_samples, num_features), optional
        The missingness model to follow; if not None, should
        be a function that takes the data matrix as input and 
        returns a binary mask indicating which entries to inclue


    Returns
    -------
    X : array (num_samples, num_features)
        A matrix of features with some removed

    '''
    # Generate a matrix of missingness indicators with the given
    # probability of missingness
    
    if missingness_model is None:
        mask = np.random.choice([0, 1], p=[obfuscation_rate, 1 - obfuscation_rate], 
                                size=X.shape)
    else:
        mask = missingness_model(X)

    return X * mask

def add_missingness_terms(X, use_inclusion_bound=False):
    '''
    This function adds features for each potentially missing
    feature, currently not considering interaction terms
    
    Parameters
    ----------
    X : matrix (num_samples, num_features)
        The original unperturbed input features
    use_inclusion_bound: binary
        Whether to exclude features that are never missing

    Returns
    -------
    X_augmented : array (num_samples, num_features*num_features)
        An augmented version of X with a new set of columns
        added for each potentially missing feature

    '''
    num_samples = X.shape[0]
    num_features = X.shape[1]

    X_augmented = X.copy()
    for f in range(num_features):
        missing_term = np.zeros((num_samples, num_features))
        missing_term[:, f] = 1
        missingness_mask = np.expand_dims(1 - np.abs(X[:, f]), 1)
        if np.sum(missingness_mask) == 0 and use_inclusion_bound:
            continue
        
        new_cols = missingness_mask * (X + missing_term)
        new_cols[new_cols == -1] = 0
        X_augmented = np.concatenate([X_augmented, 
                                    np.expand_dims(1 - np.abs(X[:, f]), 1) * (X + missing_term)], 
                                    axis=1)
    return X_augmented
    
def my_linear_dgp(X):
    weights = np.zeros(X.shape[1])
    weights[:X.shape[1]//2] = 1
    X_copy = (X.copy() + 1) // 2
    return X_copy @ weights - np.mean(X_copy @ weights)

def choose_best_gam(X_train, X_val, X_test, y_train, y_val, y_test, num_bootstraps=0):
    
    # Fit fastsparse on the data with missing data
    fit_model = fastsparsegams.fit(
        X_train, y_train, loss="Exponential", max_support_size=20, algorithm="CDPSI"
    )

    val_accs = []

    for l in range(len(fit_model.lambda_0[0])):
        lambda0 = fit_model.lambda_0[0][l]
        yhat_val = np.where(fit_model.predict(X_val, lambda_0=lambda0).flatten() > 0.5, 1, -1)
        val_accs.append(np.mean(yhat_val == y_val))
        
    best_ind = np.argmax(val_accs)

    lambda0 = fit_model.lambda_0[0][best_ind]
    test_accs = []
    if num_bootstraps > 0:
        for b in range(num_bootstraps):
            bootstrap_X, bootstrap_y = resample(X_test, y_test, replace=True, random_state=b)
            yhat_test = np.where(fit_model.predict(bootstrap_X, lambda_0=lambda0).flatten() > 0.5, 1, -1)
            test_acc = np.mean(yhat_test == bootstrap_y)
            test_accs.append(test_acc)
    else:
        yhat_test = np.where(fit_model.predict(X_test, lambda_0=lambda0).flatten() > 0.5, 1, -1)
        test_accs.append(np.mean(yhat_test == y_test))
    
    return fit_model, best_ind, test_accs
        