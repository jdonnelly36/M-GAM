import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mode
from scipy.stats.contingency import crosstab
from statsmodels.stats.multitest import multipletests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import MissingIndicator
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.low_rank_gaussian_copula import LowRankGaussianCopula

class GCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        self.gc = GaussianCopula(**gc_params)
        self.gc.fit(X)
        return self.gc

    def transform(self, X, y=None):
        X_imputed = self.gc.transform(X)
        return X_imputed

class LRGCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        print(gc_params)
        self.lrgc = LowRankGaussianCopula(**gc_params)
        self.lrgc.fit(X)
        return self.lrgc

    def transform(self, X, y=None):
        X_imputed = self.lrgc.transform(X)
        return X_imputed

# New transform to add indicator for each partially observed feature
# i.e. the Missing Indicator Method (MIM)
class MIM(MissingIndicator):
    def __init__(
        self,
        *,
        missing_values=np.nan,
        features="missing-only",
        alpha=0.05
    ):

        self.mim_features = features
        self.alpha=alpha

        if features == "dynamic":
            features = "missing-only"

        super().__init__(
            missing_values=missing_values, 
            features=features,
        )

    def fit(self, X, y=None):
        mask = super()._fit(X, y)
        self.mask = mask
        self.p_vals = []

        # print(f"Features with missing values: {self.features_}")

        if self.mim_features == "dynamic":
            if y is None:
                raise ValueError("features was set to dynamic, but y is None")

            else:
                # indicators_to_keep = []
                if len(self.features_) > 0:
                    for i in self.features_:                 

                        # Test for relation between groups
                        # Use t-test if y is continuous
                        # Use Chi-Square test if y is categorical
                        if y.dtype.name in ["int64", "object", "category"]:
                            # Make contingency table and do Chi-Squared test
                            _, table = crosstab(mask[:,i], y)
                            _, p_val, _, _ = chi2_contingency(table)
                        else:
                            # Group y by value of indicator and do two-sample t-test
                            y_groups = [y[mask[:,i]==k] for k in [0, 1]]
                            p_val = ttest_ind(*y_groups, equal_var=False).pvalue

                        # if p_val < self.alpha:
                        #     indicators_to_keep.append(i)

                        self.p_vals.append(p_val)

                    # self.features_ = np.array(indicators_to_keep)


                    # Keep indicator only if p_val is significant
                    # Use Benjamini-Hochbert correction on p-values
                    test_results, _, _, _ = multipletests(
                        pvals = self.p_vals,
                        alpha=self.alpha,
                        method="fdr_bh",
                    )
                    self.features_ = np.flatnonzero(test_results)

        return self

    def transform(self, X, y=None):


        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.mim_features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

        if self.mim_features in ["missing-only", "dynamic"]:
            if len(self.features_) < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        imputer_mask = self.mask

        if len(self.features_) < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        # print(f"Alpha: {self.alpha}\tKept features: {len(self.features_)}")
        # print(np.mean([
        #     pval for pval, j in zip(self.p_vals, range(X.shape[1])) if j not in self.features_
        # ]))
        
        # if len(immputer_mask.shape) == 3:
        #     imputer_mask = imputer_mask.squeeze()

        return imputer_mask

class Oracle_MIM(MissingIndicator):
    def __init__(
        self,
        *,
        indicators_to_keep,
        missing_values=np.nan,
    ):
        self.indicators_to_keep = indicators_to_keep
        super().__init__(
            missing_values=missing_values, 
            features="missing-only",
        )

    def fit(self, X, y=None):
        mask = super()._fit(X, y)
        self.mask = mask
        self.features_ = self.indicators_to_keep
        return self

    def transform(self, X, y=None):

        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, _ = self._get_missing_features_info(X)
        imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        imputer_mask = self.mask

        imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask


        
class Categorical_MIM(MIM):
    def __init__(
        self,
        *,
        missing_values=np.nan,
        features="missing-only",
        alpha=0.05
    ):
    
        # self.mim_features = features
        # self.alpha=alpha

        # if features == "dynamic":
        #     features = "missing-only"

        super().__init__(
            alpha=alpha,
            missing_values=missing_values, 
            features=features,
        )

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X, y=None):
        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.mim_features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

        if self.mim_features == "dynamic":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X.iloc[:, self.features_] = np.where(
                X.iloc[:, self.features_].isna(), 
                "UNK",
                X.iloc[:, self.features_],
            )          

            no_mim_features = list(set(range(X.shape[1])) - set(self.features_))
            X.iloc[:, no_mim_features] = np.where(
                X.iloc[:, no_mim_features].isna(), 
                mode(X.iloc[:, no_mim_features]).mode,
                X.iloc[:, no_mim_features],
            )

        elif self.mim_features == "missing-only":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X = np.where(
                X.isna(), 
                "UNK",
                X,
            )     


        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X = X.copy()
        
        if self.mim_features == "dynamic":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X.iloc[:, self.features_] = np.where(
                X.iloc[:, self.features_].isna(), 
                "UNK",
                X.iloc[:, self.features_],
            )          

            no_mim_features = list(set(range(X.shape[1])) - set(self.features_))
            X.iloc[:, no_mim_features] = np.where(
                X.iloc[:, no_mim_features].isna(), 
                mode(X.iloc[:, no_mim_features]).mode,
                X.iloc[:, no_mim_features],
            )

            X = X.to_numpy()

        elif self.mim_features == "missing-only":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X = np.where(
                X.isna(), 
                "UNK",
                X,
            )     


        return X