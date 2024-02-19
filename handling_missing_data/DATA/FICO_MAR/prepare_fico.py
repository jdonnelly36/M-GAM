#!/usr/bin/env python
# coding: utf-8

# # Preprocessing and splitting the breast cancer dataset

# ## Load the data

# In[1]:


import json
from pathlib import Path
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold


# In[2]:

added_missingness_num_cols = 1
for added_missingness_rate in [0.25, 0.5]: 
    


    data = pd.read_csv('fico_full.csv')


    # In[3]:


    data.shape


    # ### Add additional synthetic missingness, Non-ignorable

    # In[4]:

    np.random.seed(0)

    target_cols = np.array([0])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)

    inter_cols = np.array([1])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)
    targets = np.random.choice([0, 1], size=(data.shape[0], target_cols.shape[0]), p=[1-added_missingness_rate, added_missingness_rate])

    for i, col in enumerate(target_cols):
        print(f"Adding missingness to: {data.columns[col]}")
        thresh_col = data.columns[inter_cols[i]]
        thresh_mask = data[thresh_col] >= data[thresh_col].quantile(0.6)
        tartget_labels = np.zeros_like(thresh_mask)
        tartget_labels[thresh_mask] = 1
        mask = (targets[:, i] == 1) & (data['PoorRiskPerformance'] == tartget_labels)
        data.loc[mask, data.columns[col]] = -10


    # ### Continue processing data

    # In[5]:


    missing_val = [-7, -8, -9, -10]
    data = data.replace(missing_val, np.nan)


    # In[6]:


    data.isnull().sum()


    # Features we could encode in a special way: 
    # 
    # - 'NumTrades60Ever2DerogPubRec' and other integer features: some have few enough distinct categories that we could treat them as ordinal, perhaps. 
    # 
    # - Percent and Fraction features do need to be constrained between 0 and 100. 
    # 

    # In[7]:


    data.hist(figsize=(14, 18), layout=(5, 5));


    # ### Create training, validation and holdout sets

    # In[8]:


    outdir = Path('.')
    outdir.mkdir(exist_ok=True)

    missing_folder = f'{added_missingness_rate}'
    if not os.path.exists(outdir/ missing_folder):
        os.makedirs(outdir/ missing_folder)

    n_splits = 10
    n_folds = 5
    idx = np.arange(len(data))

    kf_splits = KFold(n_splits=n_splits, random_state=1896, shuffle=True)

    for holdout_num, out_split in enumerate(kf_splits.split(idx)):
        idx_train = idx[out_split[0]]
        idx_test = idx[out_split[1]]
        devel_fold = data.iloc[idx_train, ]
        test_fold = data.iloc[idx_test, ]

        test_fold.to_csv(outdir / f'{missing_folder}/holdout_{holdout_num}.csv', index=False)

        kf_folds = KFold(n_splits=n_folds, random_state=165782 * holdout_num, shuffle=True)
        idx_folds = np.arange(len(devel_fold))
        for fold_num, idx_fold_split in enumerate(kf_folds.split(idx_folds)):
            train_fold = devel_fold.iloc[idx_fold_split[0]]
            val_fold = devel_fold.iloc[idx_fold_split[1]]
            train_fold.to_csv(outdir / f'{missing_folder}/devel_{holdout_num}_train_{fold_num}.csv', index=False)
            val_fold.to_csv(outdir / f'{missing_folder}/devel_{holdout_num}_val_{fold_num}.csv', index=False)


    # In[ ]:




