#!/usr/bin/env python
# coding: utf-8

# # Generating and splitting a synthetic dataset

# In[1]:


from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold


# ## Generate a synthetic dataset

# In[2]:
for added_missingness_rate in [0.25, 0.5]: 

    N_SAMPLES = 1000
    N_FEATURES = 25
    N_INFORMATIVE = 25
    outdir = Path(f'SYNTHETIC_{N_SAMPLES}_SAMPLES_{N_FEATURES}_FEATURES_{N_INFORMATIVE}_INFORMATIVE/{added_missingness_rate}')
    outdir.mkdir(parents=True, exist_ok=True)


    # In[3]:


    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES,
        n_redundant=0,
        random_state=1234
    )


    # In[4]:


    ### INJECT MAR MISSINGNESS

    added_missingness_num_cols = 1

    np.random.seed(0)
    target_cols = np.array([0])
    inter_cols = np.array([1])
    targets = np.random.choice([0, 1], size=(y.shape[0], target_cols.shape[0]), p=[1-added_missingness_rate, added_missingness_rate])

    for i, col in enumerate(target_cols):
        print(f"Adding missingness to column: {col}")
        thresh_col = inter_cols[i]
        thresh_mask = x[:, thresh_col] >= np.quantile(x[:, thresh_col], .6)
        tartget_labels = np.zeros_like(thresh_mask)
        tartget_labels[thresh_mask] = 1
        mask = (targets[:, i] == 1) & (y == tartget_labels)
        x[mask, col] = np.nan


    # In[5]:


    features_df = pd.DataFrame(x)
    output_df = pd.DataFrame(y, columns=['output'])
    data_df = pd.concat([features_df, output_df], axis=1)


    # In[6]:


    data_df


    # Save a copy of the complete dataset for future reference.

    # In[7]:


    data_df.to_csv(outdir / 'synthetic_complete.csv', index=False)


    # ## Create training and test sets with completely at random missingness

    # In[8]:


    def binary_sampler(p, rows, cols):
        np.random.seed(6289278)
        unif_random_matrix = np.random.uniform(0., 1., size = (rows, cols))
        binary_random_matrix = 1 * (unif_random_matrix < p)
        return binary_random_matrix


    def make_missing_mcar(data_df, miss_rate=0.25, outcome_column='output'):
        data_features = data_df.drop(columns=[outcome_column])
        data_features_arr = np.array(data_features)

        n_rows, n_cols = data_features_arr.shape

        data_features_mask = binary_sampler(1 - miss_rate, n_rows, n_cols)
        miss_data_features_arr = data_features_arr.copy()
        miss_data_features_arr[data_features_mask == 0] = np.nan

        miss_data_features = pd.DataFrame(miss_data_features_arr)
        outcome = pd.DataFrame(data_df[outcome_column].reset_index(drop=True))
        
        miss_data = pd.concat([miss_data_features, outcome], axis=1)

        return miss_data


    # In[9]:


    n_splits = 10
    n_folds = 5
    idx = np.arange(len(data_df))

    kf_splits = KFold(n_splits=n_splits, random_state=1896, shuffle=True)

    for holdout_num, out_split in enumerate(kf_splits.split(idx)):
        idx_train = idx[out_split[0]]
        idx_test = idx[out_split[1]]
        devel_fold = data_df.iloc[idx_train, ]
        test_fold = data_df.iloc[idx_test, ]

        for train_percentage in [0,0.25,0.50]:
            for test_percentage in [0,0.25,0.50]:
                percent_str = f'train_missing_{train_percentage}_test_missing_{test_percentage}'
                train_data = make_missing_mcar(devel_fold, train_percentage)
                test_data  = make_missing_mcar(test_fold, test_percentage)

                test_data.to_csv(outdir / f'holdout_{holdout_num}_{percent_str}.csv', index=False)

                kf_folds = KFold(n_splits=n_folds, random_state=165782 * holdout_num, shuffle=True)
                idx_folds = np.arange(len(train_data))
                for fold_num, idx_fold_split in enumerate(kf_folds.split(idx_folds)):
                    train_fold = train_data.iloc[idx_fold_split[0]]
                    val_fold = train_data.iloc[idx_fold_split[1]]
                    train_fold.to_csv(outdir / f'devel_{holdout_num}_train_{fold_num}_{percent_str}.csv', index=False)
                    val_fold.to_csv(outdir / f'devel_{holdout_num}_val_{fold_num}_{percent_str}.csv', index=False)

