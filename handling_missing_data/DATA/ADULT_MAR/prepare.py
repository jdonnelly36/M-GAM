#!/usr/bin/env python


import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold



added_missingness_num_cols = 1
for added_missingness_rate in [0.25, 0.5]:
    data = pd.read_csv('adult.csv')

    data.replace('?', np.nan, inplace=True)


    # ## Encode categorical and ordinal columns

    # Currently, we encode education as categorical; there's not an obvious way to encode an ordinality for some college vs the two associate's degrees. 


    cols_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'race', 'sex']

    features_cat = data[cols_cat]
    features_numerical = data.drop(columns = cols_cat + ['income']).convert_dtypes()

    outcome = data['income']

    # Add synthetic missingness
    np.random.seed(0)

    target_cols = np.array([0])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)

    inter_cols = np.array([1])#np.random.choice(data.shape[1]-1, added_missingness_num_cols, replace=False)
    targets = np.random.choice([0, 1], size=(features_numerical.shape[0], target_cols.shape[0]), p=[1-added_missingness_rate, added_missingness_rate])

    for i, col in enumerate(target_cols):
        print(f"Adding missingness to: {features_numerical.columns[col]}")
        thresh_col = features_numerical.columns[inter_cols[i]]
        thresh_mask = features_numerical[thresh_col] >= features_numerical[thresh_col].quantile(0.6)
        tartget_labels = np.zeros_like(thresh_mask)
        tartget_labels[thresh_mask.to_numpy(dtype='bool')] = 1
        mask = (targets[:, i] == 1) & (outcome == tartget_labels)
        features_numerical.loc[mask, features_numerical.columns[col]] = np.nan




    ohe = OneHotEncoder(sparse=False, dtype=int)



    cat_array = ohe.fit_transform(features_cat)


    feature_labels = ohe.get_feature_names_out()
    features_cat_onehot = pd.DataFrame(cat_array, columns=feature_labels)


    nan_indicators = features_cat_onehot.columns[features_cat_onehot.columns.str.contains('nan')]


    for indicator in nan_indicators: 
        feature = indicator.split('_nan')[0]
        other_indicators = features_cat_onehot.columns[features_cat_onehot.columns.str.contains(feature)]
        missing_mask = features_cat_onehot[indicator] == 1
        features_cat_onehot.loc[missing_mask,other_indicators] = np.nan


    #TODO: verify - would imputation drop these indicators, or keep them in addition? 
    for indicator in nan_indicators: 
        features_cat_onehot.drop(columns=indicator, inplace=True)


    # ### Collect into a single DataFrame


    df_onehot = pd.concat([features_cat_onehot, outcome], axis=1)

    features = features_numerical.join([features_cat])
    df = pd.concat([features, outcome], axis=1)


    # Some of the code requires encoding the categorical columns (factors) with numerical levels. To ensure consistency, we save a list of all the levels of these columns.


    levels = [(col, sorted(df[col][df[col].notnull()].unique())) for col in sorted(cols_cat)]
    with open("factor_levels.json", "w", encoding="UTF-8") as levelsfile:
        json.dump(levels, levelsfile)


    # Some of the imputation code requires knowing which columns are categorical and ordinal, so we store this information.  We now include yes/no (or similar) columns in the list of categorical columns.
    # 
    # One of the imputation methods (MissForest) required encoding the one-hot columns as a single ordinal column; we also determine the column numbers of the categorical and ordinal columns for this encoded version.  For this purpose, we use a standalone variant of the `onehot_to_ord_multicat` method from `data_loader.py` that just returns the columns in the encoded version.  It does more than strictly needed, but that is to ensure it behaves as the `data_loader.py` method does.  Furthermore, the imputation methods only see the non-outcome columns, so we remove the outcome column before performing the encoding.


    def get_encoders(factor_levels):
        # factor_levels should be the `levels` variable created above
        factors = [fl[0] for fl in factor_levels]
        levels = [fl[1] for fl in factor_levels]

        # sklearn requires us to fit a non-empty DataFrame even if we specify all
        # the levels
        dummy_df = pd.DataFrame({fl[0]: [fl[1][0]] for fl in factor_levels})
        cat_colnames = factors
        # building the model for transformations
        ohe = OneHotEncoder(categories=levels, sparse=False)
        onehot_encoder = ohe.fit(dummy_df)
        encoded_colnames = ohe.get_feature_names_out(factors)
        # building LabelEncoder dictionary model
        orde = OrdinalEncoder(categories=levels)
        ordinal_encoder = orde.fit(dummy_df)

        return {
            "cat_colnames": cat_colnames,
            "onehot_encoder": onehot_encoder,
            "encoded_colnames": encoded_colnames,
            "ordinal_encoder": ordinal_encoder,
        }


    def onehot_to_ord_columns(df, factor_levels):
        encoders = get_encoders(factor_levels)
        onehot_encoder = encoders["onehot_encoder"]
        ordinal_encoder = encoders["ordinal_encoder"]
        encoded_colnames = encoders["encoded_colnames"]
        cat_colnames = encoders["cat_colnames"]

        onehot_df = df[encoded_colnames]
        oh_decoded = onehot_encoder.inverse_transform(onehot_df)
        # silence warning in ordinal_encoder.transform
        oh_decoded_df = pd.DataFrame(oh_decoded, columns=cat_colnames, index=df.index)
        ord_df = ordinal_encoder.transform(oh_decoded_df)
        ord_df = pd.DataFrame(ord_df, columns=cat_colnames, index=df.index)
        rest_df = df.drop(encoded_colnames, axis=1)
        converted_df = pd.concat([rest_df, ord_df], axis=1)
        return list(converted_df.columns)


    cols_ord = []

    idxs = {}

    columns = list(df_onehot.columns)
    idx_cat = []
    for idx, col in enumerate(columns):
        for cat in cols_cat:
            if col.startswith(cat):
                idx_cat.append(idx)
    idx_ord = [columns.index(col) for col in cols_ord]
    idxs["onehot"] = [idx_cat, idx_ord]

    encoded_cols = onehot_to_ord_columns(df_onehot.dropna(), levels)#todo: more nuanced nan handling
    idx_cat = [encoded_cols.index(col) for col in cols_cat]
    idx_ord = [encoded_cols.index(col) for col in cols_ord]
    idxs["encoded"] = [idx_cat, idx_ord]

    idxs["colnames"] = {"onehot": columns, "encoded": encoded_cols}


    with open("adult_cols.json", "w", encoding="UTF-8") as colsfile:
        json.dump(idxs, colsfile)


    # ### Create training, validation and holdout sets

    # We use the one-hot encoded data to create the standard datasets.

    import os

    missing_folder = f'{added_missingness_rate}'
    if not os.path.exists(missing_folder):
        os.makedirs(missing_folder)
    if not os.path.exists(f'{missing_folder}/distinct-missingness'): 
        os.makedirs(f'{missing_folder}/distinct-missingness')

    n_splits = 10
    n_folds = 5
    idx = np.arange(len(df))

    kf_splits = KFold(n_splits=n_splits, random_state=1896, shuffle=True)

    for holdout_num, out_split in enumerate(kf_splits.split(idx)):
        idx_train = idx[out_split[0]]
        idx_test = idx[out_split[1]]
        devel_fold = df_onehot.iloc[idx_train, ]
        test_fold = df_onehot.iloc[idx_test, ]

        test_fold.to_csv(f'{missing_folder}/holdout_{holdout_num}.csv', index=False)
        test_fold.to_csv(f'{missing_folder}/distinct-missingness/holdout_{holdout_num}.csv', index=False)

        kf_folds = KFold(n_splits=n_folds, random_state=165782 * holdout_num, shuffle=True)
        idx_folds = np.arange(len(devel_fold))
        for fold_num, idx_fold_split in enumerate(kf_folds.split(idx_folds)):
            train_fold = devel_fold.iloc[idx_fold_split[0]]
            val_fold = devel_fold.iloc[idx_fold_split[1]]
            train_fold.to_csv(f'{missing_folder}/devel_{holdout_num}_train_{fold_num}.csv', index=False)
            val_fold.to_csv(f'{missing_folder}/devel_{holdout_num}_val_{fold_num}.csv', index=False)
            train_fold.to_csv(f'{missing_folder}/distinct-missingness/devel_{holdout_num}_train_{fold_num}.csv', index=False)
            val_fold.to_csv(f'{missing_folder}/distinct-missingness/devel_{holdout_num}_val_{fold_num}.csv', index=False)






