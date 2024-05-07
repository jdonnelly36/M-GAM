#!/usr/bin/env python


from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

added_missingness_num_cols = 1
for added_missingness_rate in [0.25, 0.5]: 
    
    # fetch dataset 
    chronic_kidney_disease = fetch_ucirepo(id=336) 
    
    # data (as pandas dataframes) 
    X = chronic_kidney_disease.data.features 
    y = chronic_kidney_disease.data.targets 
    
    # metadata 
    df = pd.concat([X, y], axis=1)



    for c in df.columns:
        if df[c].dtype != 'float64':
            print(f"{c}:", df[c].unique())



    def binarize(col):
        col[col == 'normal'] = 1.0
        col[col == 'abnormal'] = 0.0

        col[col == 'yes'] = 1.0
        col[col == 'no'] = 0.0
        col[col == '\tno'] = 0.0

        col[col == 'present'] = 1.0
        col[col == 'notpresent'] = 0.0

        col[col == 'good'] = 1.0
        col[col == 'poor'] = 0.0

        # Binarize labels
        col[col == 'ckd'] = 1
        col[col == 'ckd\t'] = 1
        col[col == 'notckd'] = 0
        return col

    df = df.apply(binarize, axis=1)


    np.random.seed(0)

    target_cols = np.array([0])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)

    inter_cols = np.array([1])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)
    targets = np.random.choice([0, 1], size=(df.shape[0], target_cols.shape[0]), p=[1-added_missingness_rate, added_missingness_rate])

    for i, col in enumerate(target_cols):
        print(f"Adding missingness to: {df.columns[col]}")
        thresh_col = df.columns[inter_cols[i]]
        thresh_mask = df[thresh_col] >= df[thresh_col].quantile(0.6)
        tartget_labels = np.zeros_like(thresh_mask)
        tartget_labels[thresh_mask] = 1
        mask = (targets[:, i] == 1) & (df['class'] == tartget_labels)
        df.loc[mask, df.columns[col]] = -10

    import json
    import os
    from pathlib import Path

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from sklearn.model_selection import KFold

    missing_folder = f'{added_missingness_rate}/distinct-missingness'
    if not os.path.exists(missing_folder):
        os.makedirs(missing_folder)

    n_splits = 10
    n_folds = 5
    idx = np.arange(len(df))

    kf_splits = KFold(n_splits=n_splits, random_state=1896, shuffle=True)

    for holdout_num, out_split in enumerate(kf_splits.split(idx)):
        idx_train = idx[out_split[0]]
        idx_test = idx[out_split[1]]
        devel_fold = df.iloc[idx_train, ]
        test_fold = df.iloc[idx_test, ]

        # Check that we haven't got any duplicates
        temp = pd.concat([devel_fold, test_fold])
        assert temp.duplicated().sum() == 0

        test_fold.to_csv(f'{missing_folder}/holdout_{holdout_num}.csv', index=False)

        kf_folds = KFold(n_splits=n_folds, random_state=165782 * holdout_num, shuffle=True)
        idx_folds = np.arange(len(devel_fold))
        for fold_num, idx_fold_split in enumerate(kf_folds.split(idx_folds)):
            train_fold = devel_fold.iloc[idx_fold_split[0]]
            val_fold = devel_fold.iloc[idx_fold_split[1]]
            train_fold.to_csv(f'{missing_folder}/devel_{holdout_num}_train_{fold_num}.csv', index=False)
            val_fold.to_csv(f'{missing_folder}/devel_{holdout_num}_val_{fold_num}.csv', index=False)




