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

# In[1]:


import pandas as pd
import numpy as np
import fastsparsegams
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import time


# In[2]:


### input parameters for plots, info about MICE


# In[3]:


num_trials = 10
split_seed = 10


# In[4]:


subset = False
# subset_seed=1
# subset = f'Random_seed={subset_seed}'
# subset_size=5


# In[5]:


num_quantiles = 4
quantiles = np.linspace(0, 1, num_quantiles + 2)[1:-1] #don't take first quantile in the space, because it is 0 and will give a vacuous threshold
                                                       #The last is always true for the training set, so we do not use that either. 


# In[6]:


no_external=True


# ### Dataset Loading

# In[7]:


df = pd.read_csv('./fico_full.csv')


# In[8]:


if subset: 
    if 'Random' in subset: 
        np.random.seed(subset_seed)
        cols = df.columns[list(np.random.choice(df.shape[1]-1, 5, replace=False)) + [-1]]
        df = df[cols]
    else: 
        df = df[df.columns[[-9, -5, -4, -3, -2, -1]]] #highest missingness prop


# In[9]:


if no_external and 'ExternalRiskEstimate' in list(df): 
    df.drop('ExternalRiskEstimate', axis=1, inplace=True)


# In[10]:


# for c in df.columns:
#     print(f"Missing rate for {c}", df[df[c] <= -7].shape[0] / df[c].shape[0])


# ### Utility functions for running experiments

# In[11]:


def binarize_according_to_train(train_df, test_df, quantiles_for_binarizing = [0.2, 0.4, 0.6, 0.8, 1], overall_mi_intercept = True, overall_mi_ixn = True, specific_mi_intercept = True, specific_mi_ixn = True):
    n_train, d_train = train_df.shape
    n_test, d_test = test_df.shape
    train_binned, train_augmented_binned, test_binned, test_augmented_binned = {}, {}, {}, {}
    train_no_missing, test_no_missing = {}, {}
    for c in train_df.columns:
        if c == 'PoorRiskPerformance':
            continue
        missing_col_name = f'{c} missing'
        missing_row_train = np.zeros(n_train)
        missing_row_test = np.zeros(n_test)
        for v in list(train_df[c].quantile(quantiles_for_binarizing).unique()) + [-7, -8, -9]:
            if v in [-7, -8, -9]:

                if specific_mi_intercept:
                    new_col_name = f'{c} == {v}'
    
                    new_row_train = np.zeros(n_train)
                    new_row_train[train_df[c] == v] = 1
                    train_binned[new_col_name] = new_row_train
                    train_augmented_binned[new_col_name] = new_row_train
                    
                    new_row_test = np.zeros(n_test)
                    new_row_test[test_df[c] == v] = 1
                    test_binned[new_col_name] = new_row_test
                    test_augmented_binned[new_col_name] = new_row_test

                missing_row_train[train_df[c] == v] = 1
                missing_row_test[test_df[c] == v] = 1
            else:
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

        if overall_mi_intercept: 
            train_binned[missing_col_name] = missing_row_train
            train_augmented_binned[missing_col_name] = missing_row_train
        
            test_binned[missing_col_name] = missing_row_test
            test_augmented_binned[missing_col_name] = missing_row_test
    
    for c_outer in train_df.columns:
        if c_outer == 'PoorRiskPerformance':
            continue
        for c_inner in train_df.columns:
            for v in train_df[c_inner].quantile(quantiles_for_binarizing).unique():
                if (v in [-7, -8, -9]) or c_inner == 'PoorRiskPerformance':
                    continue
                else:
                    missing_ixn_name = f'{c_outer} missing & {c_inner} <= {v}'
                    missing_ixn_row_train = np.zeros(n_train)
                    missing_ixn_row_test = np.zeros(n_test)
                    for m_val in [-7, -8, -9]:
                        if specific_mi_ixn: 
                            new_col_name = f'{c_outer}_missing_{m_val} & {c_inner} <= {v}'
    
                            new_row_train = np.zeros(n_train)
                            new_row_train[(train_df[c_outer] == m_val) & (train_df[c_inner] <= v)] = 1
                            train_augmented_binned[new_col_name] = new_row_train
    
                            new_row_test = np.zeros(n_test)
                            new_row_test[(test_df[c_outer] == m_val) & (test_df[c_inner] <= v)] = 1
                            test_augmented_binned[new_col_name] = new_row_test

                        missing_ixn_row_train[(train_df[c_outer] == m_val) & (train_df[c_inner] <= v)] = 1
                        missing_ixn_row_test[(test_df[c_outer] == m_val) & (test_df[c_inner] <= v)] = 1

                    if overall_mi_ixn: 
                        train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                        test_augmented_binned[missing_ixn_name] = missing_ixn_row_test
                        
    train_binned['PoorRiskPerformance'] = train_df['PoorRiskPerformance']
    test_binned['PoorRiskPerformance'] = test_df['PoorRiskPerformance']
    train_no_missing['PoorRiskPerformance'] = train_df['PoorRiskPerformance']
    test_no_missing['PoorRiskPerformance'] = test_df['PoorRiskPerformance']
    train_augmented_binned['PoorRiskPerformance'] = train_df['PoorRiskPerformance']
    test_augmented_binned['PoorRiskPerformance'] = test_df['PoorRiskPerformance']
    return pd.DataFrame(train_no_missing), pd.DataFrame(train_binned), pd.DataFrame(train_augmented_binned), \
         pd.DataFrame(test_no_missing), pd.DataFrame(test_binned), pd.DataFrame(test_augmented_binned)


# In[12]:


#recover coefficients and train/test probabilities
#updated to have a list of provided lambdas, & leave values as 0 for any provided
# lambdas to the model, that did not run
def eval_model(model, X_train, X_test, col_names, provided_lambdas): 
    coeffs = np.zeros((len(provided_lambdas), X_train.shape[1]))
    missing_coeffs = np.zeros((len(provided_lambdas)))
    inter_coeffs = np.zeros((len(provided_lambdas)))
    train_probs = np.zeros((len(provided_lambdas), X_train.shape[0]))
    test_probs = np.zeros((len(provided_lambdas), X_test.shape[0]))

    for idx, lamby in enumerate(model.lambda_0[0]):

        i = provided_lambdas.index(lamby)

        train_probs[i] = model.predict(X_train.astype(float),lambda_0=lamby).reshape(-1)
        test_probs[i] = model.predict(X_test.astype(float),lambda_0=lamby).reshape(-1)

        cur_col_names = col_names[(model.coeff(lambda_0=lamby).toarray().flatten()[1:] != 0)]
        missing_coeffs[i] = sum(['-' in c for c in cur_col_names])
        inter_coeffs[i] = sum(['&' in c for c in cur_col_names])
        coeffs[i] = (model.coeff(lambda_0=lamby).toarray().flatten())[1:] #first entry is intercept
    return train_probs, test_probs, coeffs, missing_coeffs, inter_coeffs


# In[13]:


# recover mean exponential loss from probs
def loss(probs, y): 
    return np.exp(np.log(probs/(1 - probs))*-1/2*y).mean(axis=1)


# ### Running Trials

# In[14]:


folds = KFold(n_splits=num_trials, shuffle=True, random_state=split_seed)


# In[15]:


lambda_grid = [[100, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]]#, 0.01, 0.001]]#[100, 10, 1, 0.5, 0.2, 0.05, 0.02]


# In[16]:


trainacc_aug = np.zeros((len(lambda_grid[0]), num_trials))
testacc_aug = np.zeros((len(lambda_grid[0]), num_trials))
num_terms_aug = np.zeros((len(lambda_grid[0]), num_trials))

trainacc_indicator = np.zeros((len(lambda_grid[0]), num_trials))
testacc_indicator = np.zeros((len(lambda_grid[0]), num_trials))
num_terms_indicator = np.zeros((len(lambda_grid[0]), num_trials))

trainacc_no_missing = np.zeros((len(lambda_grid[0]), num_trials))
testacc_no_missing = np.zeros((len(lambda_grid[0]), num_trials))
num_terms_no_missing = np.zeros((len(lambda_grid[0]), num_trials))


# In[17]:


trainobj_aug = np.zeros((len(lambda_grid[0]), num_trials))
trainobj_indicator = np.zeros((len(lambda_grid[0]), num_trials))
trainobj_no_missing = np.zeros((len(lambda_grid[0]), num_trials))

testobj_aug = np.zeros((len(lambda_grid[0]), num_trials))
testobj_indicator = np.zeros((len(lambda_grid[0]), num_trials))
testobj_no_missing = np.zeros((len(lambda_grid[0]), num_trials))


# In[ ]:


for trial_idx, (train_index, test_index) in enumerate(folds.split(df)): 
    print(trial_idx)
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_no_missing, train_binned, train_binned_augmented, test_no_missing, test_binned, test_binned_augmented = binarize_according_to_train(train_df, test_df, quantiles_for_binarizing = quantiles)
    X_indicator_train = train_binned[train_binned.columns[:-1]].values
    y_train = train_binned['PoorRiskPerformance'].values
    X_indicator_test = test_binned[test_binned.columns[:-1]].values
    y_test = test_binned['PoorRiskPerformance'].values
    X_no_missing_train = train_no_missing[train_no_missing.columns[:-1]].values
    X_no_missing_test = test_no_missing[test_no_missing.columns[:-1]].values
    X_aug_train = train_binned_augmented[train_binned_augmented.columns[:-1]].values
    X_aug_test = test_binned_augmented[test_binned_augmented.columns[:-1]].values


    # run fastsparse on these 3 datasets
    model_aug = fastsparsegams.fit(X_aug_train.astype(float), y_train.astype(int)*2 - 1, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=500)
    model_indicator = fastsparsegams.fit(X_indicator_train.astype(float), y_train.astype(int)*2 - 1, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=500)
    model_no_missing = fastsparsegams.fit(X_no_missing_train.astype(float), y_train.astype(int)*2 - 1, loss="Exponential", algorithm="CDPSI", lambda_grid=lambda_grid, num_lambda=None, num_gamma=None, max_support_size=500)
    
    # evaluate models
    train_probs_aug, test_probs_aug, coeff_aug, missing_coeff_aug, inter_coeffs = eval_model(model_aug, X_aug_train, 
                                                                            X_aug_test, train_binned_augmented.columns[:-1], lambda_grid[0])
    trainacc_aug[:, trial_idx] = ((train_probs_aug > 0.5) == y_train).mean(axis = 1)
    testacc_aug[:, trial_idx] = ((test_probs_aug > 0.5) == y_test).mean(axis = 1)
    num_terms_aug[:, trial_idx] = (coeff_aug != 0).sum(axis=1)

    trainobj_aug[:, trial_idx] = loss(train_probs_aug, y_train)
    testobj_aug[:, trial_idx] = loss(test_probs_aug, y_test)
    
    train_probs_indicator, test_probs_indicator, coeff_indicator, missing_coeff_indicator, _ = eval_model(model_indicator, 
                                                                                                    X_indicator_train, 
                                                                                                    X_indicator_test,
                                                                                                    train_binned.columns[:-1], 
                                                                                                    lambda_grid[0])
    trainacc_indicator[:, trial_idx] = ((train_probs_indicator > 0.5) == y_train).mean(axis=1)
    testacc_indicator[:, trial_idx] = ((test_probs_indicator > 0.5) == y_test).mean(axis=1)
    num_terms_indicator[:, trial_idx] = (coeff_indicator != 0).sum(axis=1)

    trainobj_indicator[:, trial_idx] = loss(train_probs_indicator, y_train)
    testobj_indicator[:, trial_idx] = loss(test_probs_indicator, y_test)
    
    train_probs_no_missing, test_probs_no_missing, coeff_no_missing, _, __ = eval_model(model_no_missing, X_no_missing_train, X_no_missing_test, train_no_missing.columns[:-1], lambda_grid[0])
    trainacc_no_missing[:, trial_idx] = ((train_probs_no_missing > 0.5) == y_train).mean(axis=1)
    testacc_no_missing[:, trial_idx] = ((test_probs_no_missing > 0.5) == y_test).mean(axis=1)
    num_terms_no_missing[:, trial_idx] = (coeff_no_missing != 0).sum(axis=1)

    trainobj_no_missing[:, trial_idx] = loss(train_probs_no_missing, y_train)
    testobj_no_missing[:, trial_idx] = loss(test_probs_no_missing, y_test)


#currently gives min/max errors with a center mean; other options like standard error should also work. 
def errors(accs, error_bar_type='standard error'):
    if error_bar_type == 'range': 
        lower_error = (accs.mean(axis=1)[:, np.newaxis] - accs).max(axis=1)[np.newaxis, :]
        upper_error = (accs - accs.mean(axis=1)[:, np.newaxis]).max(axis=1)[np.newaxis, :]
        return np.concatenate([lower_error, upper_error], axis=0)
    elif error_bar_type == 'standard error': 
        standard_error = accs.std(axis=1)/np.sqrt(accs.shape[1]) #currently has no small sample size correction
        return standard_error #multiply by 2 for 95% CI if sample size large enough
    else: 
        print(f'Unsupported error bar type: {error_bar_type}. Must be range or confidence interval')
        return 'Error!'


# In[ ]:


## Plot results

# Assume all 0-values are due to timeouts/not running some specified lambdas
no_timeouts_no_missing = (num_terms_no_missing > 0).all(axis=1)#todo: abstract code like this into eval_model
no_timeouts_aug = (num_terms_aug > 0).all(axis=1)
no_timeouts_indicator = (num_terms_indicator > 0).all(axis=1)

nllambda = np.array([-np.log(x) for x in lambda_grid[0]])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

ax = axs[0, 0]
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_title('Train Accuracy vs Negative Log Sparsity Penalty')
ax.errorbar(nllambda[no_timeouts_no_missing], trainacc_no_missing[no_timeouts_no_missing].mean(axis=1), yerr = errors(trainacc_no_missing[no_timeouts_no_missing]), capsize=3 , label='no missingness variables')
ax.errorbar(nllambda[no_timeouts_aug], trainacc_aug[no_timeouts_aug].mean(axis=1), yerr=errors(trainacc_aug[no_timeouts_aug]), capsize=3, label='missingness interactions')
ax.errorbar(nllambda[no_timeouts_indicator], trainacc_indicator[no_timeouts_indicator].mean(axis=1), yerr=errors(trainacc_indicator[no_timeouts_indicator]), capsize=3, label='only missingness indicator')
ax.set_ylabel('Classification Accuracy')
ax.set_xlabel('-Log(Lambda_0)')
ax.legend()
# ax.set_ylim([0.69, 0.75])

ax = axs[1, 0]
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_title('Train Exponential Loss vs Negative Log Lambda_0')#\n (Other than intercept)
ax.errorbar(nllambda[no_timeouts_no_missing], trainobj_no_missing[no_timeouts_no_missing].mean(axis=1), yerr=errors(trainobj_no_missing[no_timeouts_no_missing]), capsize=3 ,label='no missingness variables')
ax.errorbar(nllambda[no_timeouts_aug], trainobj_aug[no_timeouts_aug].mean(axis=1), yerr=errors(trainobj_aug[no_timeouts_aug]), capsize=3 ,label='missingness interaction')
ax.errorbar(nllambda[no_timeouts_indicator], trainobj_indicator[no_timeouts_indicator].mean(axis=1), yerr=errors(trainobj_indicator[no_timeouts_indicator]), capsize=3 ,label='only missingness indicator')
ax.set_ylabel('Exponential Loss')
ax.set_xlabel('-Log(Lambda_0)')
ax.legend()

ax = axs[0, 1]

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title('Test Accuracy vs Negative Log Sparsity Penalty')
ax.errorbar(nllambda[no_timeouts_no_missing], testacc_no_missing[no_timeouts_no_missing].mean(axis=1), yerr = errors(testacc_no_missing[no_timeouts_no_missing]), capsize=3, label='no missingness variables')
ax.errorbar(nllambda[no_timeouts_aug], testacc_aug[no_timeouts_aug].mean(axis=1), yerr = errors(testacc_aug[no_timeouts_aug]), capsize=3, label='missingness interaction')
ax.errorbar(nllambda[no_timeouts_indicator], testacc_indicator[no_timeouts_indicator].mean(axis=1), yerr = errors(testacc_indicator[no_timeouts_indicator]), capsize=3, label='only missingness indicator')
ax.set_ylabel('Classification Accuracy')
ax.set_xlabel('-Log(Lambda_0)')
ax.legend()
# ax.set_ylim([0.69, 0.75])

ax = axs[1, 1]
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_title('Test Exponential Loss vs Negative Log Lambda_0')#\n (Other than intercept)
ax.errorbar(nllambda[no_timeouts_no_missing], testobj_no_missing[no_timeouts_no_missing].mean(axis=1), yerr=errors(testobj_no_missing[no_timeouts_no_missing]), capsize=3 ,label='no missingness variables')
ax.errorbar(nllambda[no_timeouts_aug], trainobj_aug[no_timeouts_aug].mean(axis=1), yerr=errors(testobj_aug[no_timeouts_aug]), capsize=3 ,label='missingness interaction')
ax.errorbar(nllambda[no_timeouts_indicator], trainobj_indicator[no_timeouts_indicator].mean(axis=1), yerr=errors(testobj_indicator[no_timeouts_indicator]), capsize=3 ,label='only missingness indicator')
ax.set_ylabel('Exponential Loss')
ax.set_xlabel('-Log(Lambda_0)')
ax.legend()

fig.tight_layout()

plt.savefig(f'figs/Dec/fico_comparisons_subset={subset}_quantiles={num_quantiles}_seed={split_seed}_noexternal={no_external}.png')


# In[ ]:

plt.clf()
plt.title('Sparsity vs Lambda')
plt.errorbar(nllambda[no_timeouts_no_missing], num_terms_no_missing[no_timeouts_no_missing].mean(axis=1), yerr = errors(num_terms_no_missing[no_timeouts_no_missing]), capsize=3, label='No missingness handling')
plt.errorbar(nllambda[no_timeouts_aug], num_terms_aug[no_timeouts_aug].mean(axis=1), yerr = errors(num_terms_aug[no_timeouts_aug]), capsize=3, label='Interaction')
plt.errorbar(nllambda[no_timeouts_indicator], num_terms_indicator[no_timeouts_indicator].mean(axis=1), yerr = errors(num_terms_indicator[no_timeouts_indicator]), capsize=3, label='Only indicator')
plt.ylabel('Number of Terms in model')
plt.xlabel('-Log(Lambda_0)')
plt.legend()

plt.savefig(f'figs/Dec/sparsity_lambda_subset={subset}_quantiles={num_quantiles}_seed={split_seed}_noexternal={no_external}.png')


# In[ ]:


# with open(f'figs/Dec/fico_comparisons_subset={subset}_quantiles={num_quantiles}_seed={split_seed}_noexternal={no_external}.txt', 'w') as outfile: 
#     for c in df.columns:
#         outfile.write(f"Missing rate for {c}" + str(df[df[c] <= -7].shape[0] / df[c].shape[0]) + '\n')


# In[ ]:




