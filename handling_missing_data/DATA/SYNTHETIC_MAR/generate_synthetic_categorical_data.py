#!/usr/bin/env python
# coding: utf-8

# # Generating and splitting a synthetic dataset with artificial missingness, with categorial, ordinal and uninformative features

# This generates synthetic data with the following properties; the defaults in this notebook are noted in parentheses:
# * There are `N_SAMPLES` samples (1000)
# * There are `N_FEATURES` features in total (25)
# * There are `N_INFORMATIVE` features (20)
# * Of the `N_INFORMATIVE` features, `N_CATEGORICAL` (5) are categorical and `N_ORDINAL` (5) are ordinal (that is, categorical with a meaningful order, stored as 0, 1, 2, ...).  Of the categorical features, the first one has four categories, called A, B, C, D (in no meaningful order), while the other four are binary (stored as 0, 1).
# * Of the `N_FEATURES - N_INFORMATIVE` "useless" (non-informative) features, `N_USELESS_CATEGORICAL` (1) are categorical and `N_USELESS_ORDINAL` (1) are ordinal.  ("Useless" is the scikit-learn terminology.)
# 
# These data columns are named and arranged in the following order:
# * `cts1 .. cts13` continuous features (in a random order)
# * `cat1 .. cat6` categorical features (the first is the four category one, the others are in a random order)
# * `ord1 .. ord6` ordinal features (in a random order)
# 
# The categorical features are produced by thresholding a continuous variable; different thresholds are used for different features.  Likewise, the ordinal features are produced by binning a continuous variable, with different bin collections for different features.

# In[1]:


import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# Setting up the various constants.

# In[2]:


N_SAMPLES = 1000
N_FEATURES = 25
N_INFORMATIVE = 20
N_CATEGORICAL = 5
N_ORDINAL = 5
N_USELESS_CATEGORICAL = 1
N_USELESS_ORDINAL = 1

N_CONTINUOUS = N_INFORMATIVE - N_CATEGORICAL - N_ORDINAL
N_USELESS = N_FEATURES - N_INFORMATIVE
N_USELESS_CONTINUOUS = N_USELESS - N_USELESS_CATEGORICAL - N_USELESS_ORDINAL

USELESS_SCALING = 2.5

RANDOM_SEED = 6289278
RANDOM_STATE = 1896

for added_missingness_rate in [0.25, 0.5]: 

    rng = np.random.default_rng(RANDOM_SEED)

    outdir = Path(f'SYNTHETIC_CATEGORICAL/{added_missingness_rate}')
    outdir.mkdir(parents=True, exist_ok=True)


    # We specify the breakpoints for the categorical and ordinal features. The dataset generated by `make_classification` contains four cluster, two for the positive cases and two for the negative cases.  For the informative features, each cluster has mean -1 or 1 in each variable.  However, the standard deviation in each variable varies somewhat, as each cluster is generated as a standard multivariate normal transformed by a random matrix (entries uniform in [-1, 1]), giving a multivariate normal distribution.  Our breakpoints take account of this.
    # 
    # On the other hand, the useless features are standard normal distributions.  This potentially gives the algorithms an easy way of distinguishing between informative and useless features.  To mitigate this, we scale the useless features by a factor of 2.5 to make them somewhat more similar to the informative features.
    # 
    # For more details on this, see https://github.com/scikit-learn/scikit-learn/issues/25908

    # In[3]:


    # This is for the binary categorical variables
    categorical_thresholds = [-2, 0, 2, 4]
    categorical_useless_thresholds = [0]
    ordinal_thresholds = [
        [-3, 4],
        [-2, -1, 2],
        [-3, 2, 5],
        [-4, -1, 1.5, 3.5],
        [-5, -3, -1, 1, 4]
    ]
    ordinal_useless_thresholds = [
        [-4.5, 0.5, 3]
    ]

    # for the 4-category variable, probabilities of being in category A-D for the two classes
    categories4 = [
        [0.3, 0.4, 0.2, 0.1],
        [0.2, 0.2, 0.4, 0.2],
    ]

    # Some validity checks
    assert len(categorical_thresholds) + 1 == N_CATEGORICAL
    assert len(categorical_useless_thresholds) == N_USELESS_CATEGORICAL
    assert len(ordinal_thresholds) == N_ORDINAL
    assert len(ordinal_useless_thresholds) == N_USELESS_ORDINAL
    assert math.isclose(sum(categories4[0]), 1)
    assert math.isclose(sum(categories4[1]), 1)


    # ## Generating the synthetic dataset

    # We start with a standard `make_classification`.

    # In[4]:


    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=0,
        random_state=RANDOM_STATE,
        shuffle=False,
    )


    # ### Inject MAR missingness

    # In[5]:


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


    # ### Back to Generating Synthetic Dataset

    # In[6]:


    orig_features_df = pd.DataFrame(x)
    output_df = pd.DataFrame(y, columns=["output"])


    # We scale the useless features as described above.

    # In[7]:


    orig_features_df.iloc[:, N_INFORMATIVE:] *= USELESS_SCALING


    # ### Generate the continuous features of our final dataset

    # In[8]:


    cts_informative_features_df = orig_features_df.iloc[:, :N_CONTINUOUS]
    cts_useless_features_df = orig_features_df.iloc[:, N_INFORMATIVE:(N_INFORMATIVE + N_USELESS_CONTINUOUS)]
    cts_features_df = pd.concat(
        [cts_informative_features_df, cts_useless_features_df],
        axis=1
    )
    cts_indices = np.arange(N_CONTINUOUS + N_USELESS_CONTINUOUS)
    rng.shuffle(cts_indices)
    cts_features_df = cts_features_df.iloc[:, cts_indices]
    cts_features_df = cts_features_df.set_axis(
        [f"cts{i+1}" for i in range(N_CONTINUOUS + N_USELESS_CONTINUOUS)], axis=1
    )


    # In[9]:


    cts_features_df


    # ### Generate the categorical features of our final dataset

    # In[10]:


    def cts_to_cat(series, threshold):
        return (series >= threshold).astype(int)


    # The 4-category variable is trickier.  We do not want it to behave like an ordinal variable (with the categories having a natural order), but we want the class distribution to differ between classes.  We therefore look at the fractional part of the variable value and use that to decide which class to put the sample in.
    # 
    # We use the stated class, even though in a small fraction of cases this is wrong (see the `flip_y` parameter of the `make_classification` function); this should have little impact.

    # In[11]:


    def cts_to_multicat(series, outcomes, probs):
        assert type(outcomes) == pd.Series

        sfrac = series.mod(1)
        classes = []
        for i in [0, 1]:
            # The bin limits need to start with a lower limit of 0
            bins = np.insert(np.cumsum(probs[i]), 0, 0)
            classes.append(pd.cut(sfrac, bins, labels=False))
        
        cats = classes[0].where(outcomes == 0, classes[1])
        cats = cats.map(lambda c: chr(ord("A") + c))
        return cats


    # Check that these do what we expect.

    # In[12]:


    s = pd.Series([2.71, 5.13, -3.48, 0.35, -1.46, 2.58])
    cts_to_cat(s, 0.2)


    # In[13]:


    s = pd.Series([2.71, 5.13, -3.48, 0.25, -1.46, 2.98])
    outcomes = pd.Series([0, 0, 1, 1, 0, 0])
    test_multicategories = [
        [0.3, 0.4, 0.2, 0.1],
        [0.2, 0.3, 0.3, 0.2],
    ]
    # Should give categories:
    # C, A, C, B, B, D
    cts_to_multicat(s, outcomes, test_multicategories)


    # In[14]:


    cat_informative_features_df = orig_features_df.iloc[:, N_CONTINUOUS:(N_CONTINUOUS + N_CATEGORICAL)].copy()
    cat_useless_features_df = orig_features_df.iloc[
        :, (N_INFORMATIVE + N_USELESS_CONTINUOUS):(N_INFORMATIVE + N_USELESS_CONTINUOUS + N_USELESS_CATEGORICAL)
    ].copy()

    colname = cat_informative_features_df.columns[0]
    cat_informative_features_df[colname] = cts_to_multicat(cat_informative_features_df.iloc[:, 0], output_df["output"],
                                                        categories4)

    for i, threshold in enumerate(categorical_thresholds, start=1):
        colname = cat_informative_features_df.columns[i]
        cat_informative_features_df[colname] = cts_to_cat(cat_informative_features_df.iloc[:, i], threshold)

    for i, threshold in enumerate(categorical_useless_thresholds):
        colname = cat_useless_features_df.columns[i]
        cat_useless_features_df[colname] = cts_to_cat(cat_useless_features_df.iloc[:, i], threshold)

    cat_features_df = pd.concat(
        [cat_informative_features_df, cat_useless_features_df],
        axis=1
    )

    # Shuffle the binary features
    cat_indices = np.arange(N_ORDINAL + N_USELESS_ORDINAL - 1) + 1
    rng.shuffle(cat_indices)
    cat_indices = np.insert(cat_indices, 0, 0)
    cat_features_df = cat_features_df.iloc[:, cat_indices]

    cat_features_df = cat_features_df.set_axis(
        [f"cat{i+1}" for i in range(N_CATEGORICAL + N_USELESS_CATEGORICAL)], axis=1
    )


    # In[15]:


    cat_features_df


    # ### Generate the ordinal features of our final dataset

    # In[16]:


    def cts_to_ord(series, cuts):
        bins = [-np.inf] + cuts + [np.inf]
        ord_np = pd.cut(series, bins, labels=False)
        return ord_np


    # Check that this does what we expect.

    # In[17]:


    s = pd.Series([0.7, 0.1, -0.4, 0.3, 1.2, -0.8])
    cutpoints = [-0.5, 0, 0.5]
    cts_to_ord(s, cutpoints)


    # In[18]:


    ord_informative_features_df = orig_features_df.iloc[
        :, (N_CONTINUOUS + N_CATEGORICAL):(N_CONTINUOUS + N_CATEGORICAL + N_ORDINAL)].copy()
    ord_useless_features_df = orig_features_df.iloc[
        :, (N_INFORMATIVE + N_USELESS_CONTINUOUS + N_USELESS_CATEGORICAL):
        (N_INFORMATIVE + N_USELESS_CONTINUOUS + N_USELESS_CATEGORICAL + N_USELESS_ORDINAL)
    ].copy()

    for i, thresholds in enumerate(ordinal_thresholds):
        colname = ord_informative_features_df.columns[i]
        ord_informative_features_df[colname] = cts_to_ord(ord_informative_features_df.iloc[:, i], thresholds)

    for i, thresholds in enumerate(ordinal_useless_thresholds):
        colname = ord_useless_features_df.columns[i]
        ord_useless_features_df[colname] = cts_to_ord(ord_useless_features_df.iloc[:, i], thresholds)

    ord_features_df = pd.concat(
        [ord_informative_features_df, ord_useless_features_df],
        axis=1
    )
    ord_indices = np.arange(N_ORDINAL + N_USELESS_ORDINAL)
    rng.shuffle(cat_indices)
    ord_features_df = ord_features_df.iloc[:, cat_indices]
    ord_features_df = ord_features_df.set_axis(
        [f"ord{i+1}" for i in range(N_ORDINAL + N_USELESS_ORDINAL)], axis=1
    )


    # In[19]:


    ord_features_df


    # ### Generating the final dataset and one-hot encoded version

    # Finally combine all of the features together and shuffle the samples.

    # In[20]:


    data_df = pd.concat([cts_features_df, cat_features_df, ord_features_df, output_df], axis=1)


    # The following function generates a one-hot encoded version of the data; only the `cat1` column needs encoding.

    # In[21]:


    def onehot_encode(df, full_df, encode="cat1"):
        """
        One-hot encode column `encode` in `df`.
        """
        enc_col = df.columns.get_loc(encode)
        df_preenc = df.iloc[:, :enc_col]
        df_postenc = df.iloc[:, (enc_col + 1):]
        df_enccol = df[[encode]]

        ohe = OneHotEncoder(sparse=False, dtype=int)
        encoded_array = ohe.fit_transform(df_enccol)
        feature_labels = ohe.get_feature_names_out([encode])
        encoded = pd.DataFrame(encoded_array, columns=feature_labels, index=df.index)

        df_onehot = pd.concat([df_preenc, encoded, df_postenc], axis=1)
        
        return df_onehot


    # In[22]:


    data_df_onehot = onehot_encode(data_df, data_df, "cat1")


    # In[23]:


    data_df, data_df_onehot = shuffle(data_df, data_df_onehot, random_state=RANDOM_STATE)


    # In[24]:


    data_df


    # In[25]:


    data_df_onehot


    # Save a copy of the complete dataset for future reference.

    # In[26]:


    data_df.to_csv(outdir / "synthetic_categorical_complete.csv", index=False)
    data_df_onehot.to_csv(outdir / "synthetic_categorical_complete_onehot.csv", index=False)


    # Some of the code requires encoding the categorical columns (non-binary factors) with numerical levels. To ensure consistency, we save a list of all the levels of these columns.  This code is taken from the Breast cancer preparation notebook.

    # In[27]:


    levels = [(col, sorted(data_df[col].unique())) for col in ["cat1"]]
    with open(outdir / "synthetic_categorical_factor_levels.json", "w", encoding="UTF-8") as levelsfile:
        json.dump(levels, levelsfile)


    # Some of the imputation code requires knowing which columns are categorical and ordinal, so we store this information.  We now include yes/no (or similar) columns in the list of categorical columns.
    # 
    # One of the imputation methods (MissForest) required encoding the one-hot columns as a single ordinal column; we also determine the column numbers of the categorical and ordinal columns for this encoded version.  For this purpose, we use a standalone variant of the `onehot_to_ord_multicat` method from `data_loader.py` that just returns the columns in the encoded version.  It does more than strictly needed, but that is to ensure it behaves as the `data_loader.py` method does.  Furthermore, the imputation methods only see the non-outcome columns, so we remove the outcome column before performing the encoding.

    # In[28]:


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


    # In[29]:


    cols_cat = [col for col in data_df.columns if col.startswith("cat")]
    cols_ord = [col for col in data_df.columns if col.startswith("ord")]

    data_df_onehot_no_outcome = data_df_onehot.drop(columns=["output"])

    idxs = {}

    columns = list(data_df_onehot_no_outcome.columns)
    idx_cat = []
    for idx, col in enumerate(columns):
        for cat in cols_cat:
            if col.startswith(cat):
                idx_cat.append(idx)
    idx_ord = [columns.index(col) for col in cols_ord]
    idxs["onehot"] = [idx_cat, idx_ord]

    encoded_cols = onehot_to_ord_columns(data_df_onehot_no_outcome, levels)
    idx_cat = [encoded_cols.index(col) for col in cols_cat]
    idx_ord = [encoded_cols.index(col) for col in cols_ord]
    idxs["encoded"] = [idx_cat, idx_ord]

    idxs["colnames"] = {"onehot": columns, "encoded": encoded_cols}


    # In[30]:


    with open(outdir / "synthetic_categorical_cols.json", "w", encoding="UTF-8") as colsfile:
        json.dump(idxs, colsfile)


    # ## Create training and test sets with completely at random missingness

    # In[31]:


    def binary_sampler(p, rows, cols):
        sampler_rng = np.random.default_rng(RANDOM_SEED)
        unif_random_matrix = sampler_rng.uniform(size = (rows, cols))
        binary_random_matrix = (unif_random_matrix < p).astype(int)
        return binary_random_matrix


    def make_missing_mcar(data_df, miss_rate=0.25, outcome_column="output", ignored_columns=None):
        """
        Introduce missingness completely at random into a DataFrame
        
        Parameters
        ----------
        data_df : pd.DataFrame
            The input data.
        miss_rate : float
            The missingness rate, between 0 (no missingness) and 1 (completely missing).
            This specifies the probability that any entry is missing.  All entries are
            considered independently.
        outcome_column : str
            The column containing the outcome of the sample.  No missingness may be
            introduced here.
        ignored_columns : list of str
            Columns names to be ignored when introducing missingness, i.e., no
            missingness is to be introduced to these columns.
        
        """
        data_features = data_df.drop(columns=[outcome_column])
        data_features_arr = np.array(data_features)
        if ignored_columns:
            ignored_columns_locs = [data_features.columns.get_loc(col) for col in ignored_columns]
        else:
            ignored_columns_locs = []
        
        n_rows, n_cols = data_features_arr.shape

        data_features_mask = binary_sampler(1 - miss_rate, n_rows, n_cols)
        data_features_mask[:, ignored_columns_locs] = 1
        miss_data_features_arr = data_features_arr.copy()
        miss_data_features_arr[data_features_mask == 0] = np.nan

        miss_data_features = pd.DataFrame(
            miss_data_features_arr,
            index=data_features.index,
            columns=data_features.columns
        )
        outcome = pd.DataFrame(data_df[outcome_column])
        
        miss_data = pd.concat([miss_data_features, outcome], axis=1)
        miss_data = miss_data.convert_dtypes()

        return miss_data


    # A subtlety in this data generation, compared to the plain synthetic generation or the Breast Cancer set, is that we must induce the missingness without touching the one-hot encoded columns; most of the imputation methods we are using cannot handle this scenario.

    # In[32]:


    multicat_columns = [col for col in data_df_onehot.columns if col.startswith("cat1_")]


    # In[33]:


    n_splits = 10
    n_folds = 5
    idx = np.arange(len(data_df))

    kf_splits = KFold(n_splits=n_splits, random_state=RANDOM_STATE, shuffle=True)

    for holdout_num, out_split in enumerate(kf_splits.split(idx)):
        idx_train = idx[out_split[0]]
        idx_test = idx[out_split[1]]
        devel_fold = data_df_onehot.iloc[idx_train, ]
        test_fold = data_df_onehot.iloc[idx_test, ]

        for train_percentage in [0,0.25,0.50]:
            for test_percentage in [0,0.25,0.50]:
                percent_str = f'train_missing_{train_percentage}_test_missing_{test_percentage}'
                train_data = make_missing_mcar(devel_fold, train_percentage, ignored_columns=multicat_columns)
                test_data = make_missing_mcar(test_fold, test_percentage, ignored_columns=multicat_columns)

                test_data.to_csv(outdir / f'holdout_{holdout_num}_{percent_str}.csv', index=False)

                kf_folds = KFold(n_splits=n_folds, random_state=RANDOM_STATE * holdout_num, shuffle=True)
                idx_folds = np.arange(len(train_data))
                for fold_num, idx_fold_split in enumerate(kf_folds.split(idx_folds)):
                    train_fold = train_data.iloc[idx_fold_split[0]]
                    val_fold = train_data.iloc[idx_fold_split[1]]
                    train_fold.to_csv(outdir / f'devel_{holdout_num}_train_{fold_num}_{percent_str}.csv', index=False)
                    val_fold.to_csv(outdir / f'devel_{holdout_num}_val_{fold_num}_{percent_str}.csv', index=False)

