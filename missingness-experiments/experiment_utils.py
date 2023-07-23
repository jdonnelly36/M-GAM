import fastsparsegams
import numpy as np
import copy
from matplotlib import pyplot as plt

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
    X = np.random.randint(2, size=(num_samples, num_features)).astype(float)
    # X is a matrix of 0 and 1's; we want 0 to indicate false, so adjust
    # them to -1
    X = 2 * X - 1

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

def add_missingness_terms(X):
    '''
    This function adds features for each potentially missing
    feature, currently not considering interaction terms
    
    Parameters
    ----------
    X : matrix (num_samples, num_features)
        The original unperturbed input features

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
        new_cols = np.expand_dims(1 - np.abs(X[:, f]), 1) * (X + missing_term)
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

def choose_best_gam(X_train, X_val, X_test, y_train, y_val, y_test):
    
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
    yhat_test = np.where(fit_model.predict(X_test, lambda_0=lambda0).flatten() > 0.5, 1, -1)
    test_acc = np.mean(yhat_test == y_test)
    
    return fit_model, best_ind, test_acc
        