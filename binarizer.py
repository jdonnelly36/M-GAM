import pandas as pd
import numpy as np

class Binarizer:
    def __init__(self, quantiles = [0.2, 0.4, 0.6, 0.8], label = 'Overall Survival Status', 
                 miss_vals = [-7, -8, -9, np.NaN], overall_mi_intercept = False, overall_mi_ixn = False, specific_mi_intercept = True, specific_mi_ixn = True,
                 imputer = None, categorical_cols = [], numerical_cols = []):
        self.quantiles = quantiles
        self.label = label
        self.miss_vals = miss_vals
        self.overall_mi_intercept = overall_mi_intercept
        self.overall_mi_ixn = overall_mi_ixn
        self.specific_mi_intercept = specific_mi_intercept
        self.specific_mi_ixn = specific_mi_ixn
        self.imputer = imputer
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

    def _nansafe_equals(self, dframe, element):
        if np.isnan(element) :
            return dframe.isna()
        else:
            return dframe == element
    
    #uses an imputer (like mean) to impute missing values
    #(imputed value is based on imputer applied to the non missing training set values)
    def _impute_single_val(self, train_df, test_df):
        for c in train_df.columns:
            if c == self.label:
                continue
            val = self.imputer(train_df[c][~train_df[c].isin(self.miss_vals)])
            train_df.loc[train_df[c].isin(self.miss_vals), c] = val
            test_df.loc[test_df[c].isin(self.miss_vals), c] = val
        return train_df, test_df

    def binarize_and_augment(self, train_df, test_df, imputed_train_df = None, 
                             imputed_test_df = None, validation_size = 0):
        '''
        Core binarization method for M-GAM. Takes a training and test set with missing values, 
        bins features by training quantiles, and augments the dataset with missingness indicators and interactions.

        Includes optional support for imputing missing values alongside having indicators and interactions, 
        either by providing the dataset post-imputation alongside the dataset pre-imputation, or using 
        imputations provided by self.imputer if self.imputer was set during initialization.
        '''
        if imputed_train_df is not None and imputed_test_df is not None:
            return_tuple = self._imputed_binarize(train_df, test_df, imputed_train_df, imputed_test_df)
        elif self.imputer is not None:
            imputed_train_df, imputed_test_df = self._impute_single_val(train_df, test_df)
            return_tuple = self._imputed_binarize(train_df, test_df, imputed_train_df, imputed_test_df)
        else: 
            return_tuple = self._binarize(train_df, test_df)
        
        if validation_size == 0: 
            return return_tuple
        else:
            train_augmented_binned, test_augmented_binned, train_labels_augmented_binned, test_labels_augmented_binned = return_tuple
            n_train = train_augmented_binned.shape[0] - validation_size
            val_augmented_binned = train_augmented_binned[n_train:]
            val_labels_augmented_binned = train_labels_augmented_binned[n_train:]
            train_augmented_binned = train_augmented_binned[:n_train]
            train_labels_augmented_binned = train_labels_augmented_binned[:n_train]
            return (train_augmented_binned, 
                    val_augmented_binned, 
                    test_augmented_binned, 
                    train_labels_augmented_binned,
                    val_labels_augmented_binned,
                    test_labels_augmented_binned)
    
  
    def _imputed_binarize(self, train_df, test_df, imputed_train_df, imputed_test_df):
        n_train, _ = train_df.shape
        n_test, _ = test_df.shape
        train_binned, train_augmented_binned, test_binned, test_augmented_binned = {}, {}, {}, {}
        train_no_missing, test_no_missing = {}, {}
        miss_val_cols = []
        for c in train_df.columns:
            if c == self.label:
                continue
            has_missing = False
            missing_col_name = f'{c} missing'
            missing_row_train = np.zeros(n_train)
            missing_row_test = np.zeros(n_test)
            for v in self.miss_vals:
                if self.specific_mi_intercept:
                    new_col_name = f'{c} == {v}'
                    new_row_train = np.zeros(n_train)
                    new_row_train[self._nansafe_equals(train_df[c],v)] = 1
                    new_row_test = np.zeros(n_test)
                    new_row_test[self._nansafe_equals(test_df[c],v)] = 1
                    if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                        train_binned[new_col_name] = new_row_train
                        train_augmented_binned[new_col_name] = new_row_train
                        test_binned[new_col_name] = new_row_test
                        test_augmented_binned[new_col_name] = new_row_test
                        has_missing = True
                missing_row_train[self._nansafe_equals(train_df[c],v)] = 1
                missing_row_test[self._nansafe_equals(test_df[c],v)] = 1
            if self.overall_mi_intercept: 
                if missing_row_train.sum() > 0 or missing_row_test.sum() > 0: #has missingness
                    train_binned[missing_col_name] = missing_row_train
                    train_augmented_binned[missing_col_name] = missing_row_train
                    test_binned[missing_col_name] = missing_row_test
                    test_augmented_binned[missing_col_name] = missing_row_test
                    has_missing = True
            if has_missing:
                miss_val_cols.append(c)
        for c in self.numerical_cols:
            if c == self.label:
                continue
            for v in list(train_df[c].quantile(self.quantiles).unique()):
                if v in self.miss_vals:
                    continue
                else:
                    new_col_name = f'{c} <= {v}'

                    new_row_train = np.zeros(n_train)
                    new_row_train[train_df[c] <= v] = 1
                    new_row_train[train_df[c].isin(self.miss_vals)] = imputed_train_df[c][train_df[c].isin(self.miss_vals)] <= v

                    train_no_missing[new_col_name] = new_row_train
                    train_binned[new_col_name] = new_row_train
                    train_augmented_binned[new_col_name] = new_row_train
                    
                    new_row_test = np.zeros(n_test)
                    new_row_test[test_df[c] <= v] = 1
                    new_row_test[test_df[c].isin(self.miss_vals)] = imputed_test_df[c][test_df[c].isin(self.miss_vals)] <= v
                    test_no_missing[new_col_name] = new_row_test
                    test_binned[new_col_name] = new_row_test
                    test_augmented_binned[new_col_name] = new_row_test
        for c in self.categorical_cols:
            if c == self.label:
                continue
            for v in list(train_df[c].unique()):
                if v in self.miss_vals:
                    continue
                else: 
                    new_col_name = f'{c} == {v}'

                    new_row_train = np.zeros(n_train)
                    new_row_train[train_df[c] == v] = 1
                    new_row_train[train_df[c].isin(self.miss_vals)] = imputed_train_df[c][train_df[c].isin(self.miss_vals)] == v
                    train_no_missing[new_col_name] = new_row_train
                    train_binned[new_col_name] = new_row_train
                    train_augmented_binned[new_col_name] = new_row_train
                    
                    new_row_test = np.zeros(n_test)
                    new_row_test[test_df[c] == v] = 1
                    new_row_test[test_df[c].isin(self.miss_vals)] = imputed_test_df[c][test_df[c].isin(self.miss_vals)] == v
                    test_no_missing[new_col_name] = new_row_test
                    test_binned[new_col_name] = new_row_test
                    test_augmented_binned[new_col_name] = new_row_test
        for c_outer in miss_val_cols: #NOTE: miss_val_cols expects missingness interactions only to happen if there are missingness intercepts too
            if c_outer == self.label:
                continue
            for c_inner in self.numerical_cols:
                for v in train_df[c_inner].quantile(self.quantiles).unique():
                    if (v in self.miss_vals) or c_inner == self.label or c_inner == c_outer:
                        continue
                    else:
                        missing_ixn_name = f'{c_outer} missing & {c_inner} <= {v}'
                        missing_ixn_row_train = np.zeros(n_train)
                        missing_ixn_row_test = np.zeros(n_test)
                        for m_val in self.miss_vals:
                            if self.specific_mi_ixn: 
                                new_col_name = f'{c_outer}_missing_{m_val} & {c_inner} <= {v}'
        
                                new_row_train = np.zeros(n_train)
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] <= v)] = 1
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))
                                              & (imputed_train_df[c_inner] <= v)] = 0
        
                                new_row_test = np.zeros(n_test)
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] <= v)] = 1
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))
                                             & (imputed_test_df[c_inner] <= v)] = 0

                                if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                                    train_augmented_binned[new_col_name] = new_row_train
                                    test_augmented_binned[new_col_name] = new_row_test

                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] <= v)] = 1
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] <= v)] = 1
                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                        if self.overall_mi_ixn: 
                            if (missing_ixn_row_train.sum() > 0 or missing_ixn_row_test.sum() > 0): #has missingness
                                train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                                test_augmented_binned[missing_ixn_name] = missing_ixn_row_test
            for c_inner in self.categorical_cols:
                for v in train_df[c_inner].unique():
                    if (v in self.miss_vals) or c_inner == self.label or c_inner == c_outer:
                        continue
                    else:
                        missing_ixn_name = f'{c_outer} missing & {c_inner} == {v}'
                        missing_ixn_row_train = np.zeros(n_train)
                        missing_ixn_row_test = np.zeros(n_test)
                        for m_val in self.miss_vals:
                            if self.specific_mi_ixn: 
                                new_col_name = f'{c_outer}_missing_{m_val} & {c_inner} == {v}'
        
                                new_row_train = np.zeros(n_train)
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] == v)] = 1
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))
                                              & (imputed_train_df[c_inner] == v)] = 0
        
                                new_row_test = np.zeros(n_test)
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] == v)] = 1
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))
                                             & (imputed_test_df[c_inner] == v)] = 0

                                if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                                    train_augmented_binned[new_col_name] = new_row_train
                                    test_augmented_binned[new_col_name] = new_row_test

                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] == v)] = 1
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] == v)] = 1
                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                        if self.overall_mi_ixn: 
                            if (missing_ixn_row_train.sum() > 0 or missing_ixn_row_test.sum() > 0):
                                train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                                test_augmented_binned[missing_ixn_name] = missing_ixn_row_test

        # get feature clusters
        aug_clusters = {}
        bin_clusters = {}
        no_clusters = {}
        for c in train_df.columns:
            #TODO: test clustering system, define behaviour in cases of redundant column names
            aug_clusters[c] = [i for i, element in enumerate(train_augmented_binned) if element.startswith(c+' <') or element.startswith(c+' =')]
            bin_clusters[c] = [i for i, element in enumerate(train_binned) if element.startswith(c+' <') or element.startswith(c+' =')]    
            no_clusters[c] = [i for i, element in enumerate(train_no_missing) if element.startswith(c+' <') or element.startswith(c+' =')]
        

        train_binned[self.label] = train_df[self.label]
        test_binned[self.label] = test_df[self.label]
        train_no_missing[self.label] = train_df[self.label]
        test_no_missing[self.label] = test_df[self.label]
        train_augmented_binned[self.label] = train_df[self.label]
        test_augmented_binned[self.label] = test_df[self.label]

        return (pd.DataFrame(train_augmented_binned)[[c for c in train_augmented_binned.keys() if c != self.label]].values,
                pd.DataFrame(test_augmented_binned)[[c for c in test_augmented_binned.keys() if c != self.label]].values, 
                pd.DataFrame(train_augmented_binned)[self.label].values,
                pd.DataFrame(test_augmented_binned)[self.label].values
        )
    
    def _binarize(self, train_df, test_df):
        n_train, _ = train_df.shape
        n_test, _ = test_df.shape
        train_binned, train_augmented_binned, test_binned, test_augmented_binned = {}, {}, {}, {}
        train_no_missing, test_no_missing = {}, {}

        # This will be a dictionary telling us which column indices correspond to which
        # input variables
        dataset_structure_map = {}
        cur_new_col_index = 0

        thresh_vals = []

        miss_val_cols = []
        for c in train_df.columns:
            if c == self.label:
                continue
            has_missing = False
            missing_col_name = f'{c} missing'
            missing_row_train = np.zeros(n_train)
            missing_row_test = np.zeros(n_test)
            dataset_structure_map[c] = {}
            dataset_structure_map[c]['intercepts'] = {}
            dataset_structure_map[c]['interactions'] = {}
            dataset_structure_map[c]['bins'] = []
            
            for v in self.miss_vals:
                if self.specific_mi_intercept:
                    new_col_name = f'{c} == {v}'
                    new_row_train = np.zeros(n_train)
                    new_row_train[self._nansafe_equals(train_df[c],v)] = 1
                    new_row_test = np.zeros(n_test)
                    new_row_test[self._nansafe_equals(test_df[c],v)] = 1
                    if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                        train_binned[new_col_name] = new_row_train
                        train_augmented_binned[new_col_name] = new_row_train
                        dataset_structure_map[c]['intercepts'][v] = cur_new_col_index
                        thresh_vals.append(-1)
                        cur_new_col_index += 1
                        
                        test_binned[new_col_name] = new_row_test
                        test_augmented_binned[new_col_name] = new_row_test
                        has_missing = True
                missing_row_train[self._nansafe_equals(train_df[c],v)] = 1
                missing_row_test[self._nansafe_equals(test_df[c],v)] = 1
            if self.overall_mi_intercept: 
                if missing_row_train.sum() > 0 or missing_row_test.sum() > 0: #has missingness
                    train_binned[missing_col_name] = missing_row_train
                    train_augmented_binned[missing_col_name] = missing_row_train
                    dataset_structure_map[c]['intercepts']['any'] = cur_new_col_index
                    thresh_vals.append(-1)
                    cur_new_col_index += 1
                    test_binned[missing_col_name] = missing_row_test
                    test_augmented_binned[missing_col_name] = missing_row_test
                    has_missing = True
            if has_missing:
                miss_val_cols.append(c)
        for c in self.numerical_cols:
            if c == self.label:
                continue
            for v in list(train_df[c].quantile(self.quantiles).unique()):
                if (v in self.miss_vals) or np.isnan(v):
                    continue
                else:
                    thresh_vals.append(v)
                    new_col_name = f'{c} <= {v}'

                    new_row_train = np.zeros(n_train)
                    new_row_train[train_df[c] <= v] = 1
                    new_row_train[train_df[c].isin(self.miss_vals)] = 0
                    train_no_missing[new_col_name] = new_row_train
                    train_binned[new_col_name] = new_row_train
                    train_augmented_binned[new_col_name] = new_row_train
                    dataset_structure_map[c]['bins'] = dataset_structure_map[c]['bins'] + [cur_new_col_index]
                    cur_new_col_index += 1
                    
                    new_row_test = np.zeros(n_test)
                    new_row_test[test_df[c] <= v] = 1
                    new_row_test[test_df[c].isin(self.miss_vals)] = 0
                    test_no_missing[new_col_name] = new_row_test
                    test_binned[new_col_name] = new_row_test
                    test_augmented_binned[new_col_name] = new_row_test
        for c in self.categorical_cols:
            if c == self.label:
                continue
            for v in list(train_df[c].unique()):
                if (v in self.miss_vals) or np.isnan(v):
                    continue
                else: 
                    thresh_vals.append(v)
                    new_col_name = f'{c} == {v}'
                    # print(f"For col {new_col_name} adding {cur_new_col_index}")

                    new_row_train = np.zeros(n_train)
                    new_row_train[train_df[c] == v] = 1
                    new_row_train[train_df[c].isin(self.miss_vals)] = 0
                    train_no_missing[new_col_name] = new_row_train
                    train_binned[new_col_name] = new_row_train
                    train_augmented_binned[new_col_name] = new_row_train
                    dataset_structure_map[c]['bins'] = dataset_structure_map[c]['bins'] + [cur_new_col_index]
                    cur_new_col_index += 1
                    
                    new_row_test = np.zeros(n_test)
                    new_row_test[test_df[c] == v] = 1
                    new_row_test[test_df[c].isin(self.miss_vals)] = 0
                    test_no_missing[new_col_name] = new_row_test
                    test_binned[new_col_name] = new_row_test
                    test_augmented_binned[new_col_name] = new_row_test
        for c_outer in miss_val_cols:
            if c_outer == self.label:
                continue
            for c_inner in self.numerical_cols:
                for v in train_df[c_inner].quantile(self.quantiles).unique():
                    if (v in self.miss_vals) or c_inner == self.label or c_inner == c_outer:
                        continue
                    else:
                        missing_ixn_name = f'{c_outer} missing & {c_inner} <= {v}'
                        missing_ixn_row_train = np.zeros(n_train)
                        missing_ixn_row_test = np.zeros(n_test)
                        for m_val in self.miss_vals:
                            if self.specific_mi_ixn: 
                                new_col_name = f'{c_outer}_missing_{m_val} & {c_inner} <= {v}'
        
                                new_row_train = np.zeros(n_train)
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] <= v)] = 1
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
        
                                new_row_test = np.zeros(n_test)
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] <= v)] = 1
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                                if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                                    train_augmented_binned[new_col_name] = new_row_train
                                    test_augmented_binned[new_col_name] = new_row_test

                                    if 'interactions' not in dataset_structure_map[c_inner]:
                                        dataset_structure_map[c_inner]['interactions'] = {}
                                    if c_outer not in dataset_structure_map[c_inner]['interactions']:
                                        dataset_structure_map[c_inner]['interactions'][c_outer] = {}
                                    if m_val not in dataset_structure_map[c_inner]['interactions'][c_outer]:
                                        dataset_structure_map[c_inner]['interactions'][c_outer][m_val] = []

                                    thresh_vals.append(v)
                                    dataset_structure_map[c_inner]['interactions'][c_outer][m_val] = dataset_structure_map[c_inner]['interactions'][c_outer][m_val] + [cur_new_col_index]
                                    cur_new_col_index += 1

                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] <= v)] = 1
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] <= v)] = 1
                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                        if self.overall_mi_ixn: 
                            if (missing_ixn_row_train.sum() > 0 or missing_ixn_row_test.sum() > 0): #has missingness
                                train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                                test_augmented_binned[missing_ixn_name] = missing_ixn_row_test
            for c_inner in self.categorical_cols:
                for v in train_df[c_inner].unique():
                    if (v in self.miss_vals) or c_inner == self.label or c_inner == c_outer:
                        continue
                    else:
                        missing_ixn_name = f'{c_outer} missing & {c_inner} == {v}'
                        missing_ixn_row_train = np.zeros(n_train)
                        missing_ixn_row_test = np.zeros(n_test)
                        for m_val in self.miss_vals:
                            if self.specific_mi_ixn: 
                                new_col_name = f'{c_outer}_missing_{m_val} & {c_inner} == {v}'
        
                                new_row_train = np.zeros(n_train)
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] == v)] = 1
                                new_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
        
                                new_row_test = np.zeros(n_test)
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] == v)] = 1
                                new_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                                if new_row_train.sum() > 0 or new_row_test.sum() > 0: #has missingness
                                    train_augmented_binned[new_col_name] = new_row_train
                                    test_augmented_binned[new_col_name] = new_row_test

                                    if 'interactions' not in dataset_structure_map[c_inner]:
                                        dataset_structure_map[c_inner]['interactions'] = {}
                                    if c_outer not in dataset_structure_map[c_inner]['interactions']:
                                        dataset_structure_map[c_inner]['interactions'][c_outer] = {}
                                    if m_val not in dataset_structure_map[c_inner]['interactions'][c_outer]:
                                        dataset_structure_map[c_inner]['interactions'][c_outer][m_val] = []

                                    thresh_vals.append(v)
                                    dataset_structure_map[c_inner]['interactions'][c_outer][m_val] = dataset_structure_map[c_inner]['interactions'][c_outer][m_val] + [cur_new_col_index]
                                    cur_new_col_index += 1

                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner] == v)] = 1
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner] == v)] = 1
                            missing_ixn_row_train[(self._nansafe_equals(train_df[c_outer],m_val)) & (train_df[c_inner].isin(self.miss_vals))] = 0
                            missing_ixn_row_test[(self._nansafe_equals(test_df[c_outer],m_val)) & (test_df[c_inner].isin(self.miss_vals))] = 0

                        if self.overall_mi_ixn: 
                            if (missing_ixn_row_train.sum() > 0 or missing_ixn_row_test.sum() > 0):
                                train_augmented_binned[missing_ixn_name] = missing_ixn_row_train
                                test_augmented_binned[missing_ixn_name] = missing_ixn_row_test

        # get feature clusters
        aug_clusters = {}
        bin_clusters = {}
        no_clusters = {}
        for c in train_df.columns:
            #TODO: test clustering system, define behaviour in cases of redundant column names
            aug_clusters[c] = [i for i, element in enumerate(train_augmented_binned) if element.startswith(c+' <') or element.startswith(c+' =')]
            bin_clusters[c] = [i for i, element in enumerate(train_binned) if element.startswith(c+' <') or element.startswith(c+' =')]    
            no_clusters[c] = [i for i, element in enumerate(train_no_missing) if element.startswith(c+' <') or element.startswith(c+' =')]
        

        train_binned[self.label] = train_df[self.label]
        test_binned[self.label] = test_df[self.label]
        train_no_missing[self.label] = train_df[self.label]
        test_no_missing[self.label] = test_df[self.label]
        train_augmented_binned[self.label] = train_df[self.label]
        test_augmented_binned[self.label] = test_df[self.label]
        
        self.dataset_structure_map = dataset_structure_map
        self.thresh_vals = thresh_vals

        return (pd.DataFrame(train_augmented_binned)[[c for c in train_augmented_binned.keys() if c != self.label]].values,
                pd.DataFrame(test_augmented_binned)[[c for c in test_augmented_binned.keys() if c != self.label]].values, 
                pd.DataFrame(train_augmented_binned)[self.label].values,
                pd.DataFrame(test_augmented_binned)[self.label].values
        )