import copy
import sys
import uuid

import numpy as np
import pandas as pd  # type: ignore
from models.gain.gain_utils import (normalization, renormalization, rounding,
                                    rounding_automatic)


class Imputer:
    """Impute missing values using a variety of methods"""

    def __init__(self, imputation_method, random_state=0):
        self.imputation_method = imputation_method

        if imputation_method == "Mean":
            from sklearn.impute import SimpleImputer

            self.imputer = SimpleImputer(strategy="mean")

        elif imputation_method == "MissForest":
            # missingpy tries to import from sklearn.neighbors.base which
            # no longer exists in newer versions of sklearn
            try:
                import sklearn.neighbors._base  # type: ignore

                sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
            except ImportError:
                pass

            from missingpy import MissForest

            self.imputer = MissForest(random_state=random_state)

        elif imputation_method == "MIWAE":
            from models.miwae import MIWAE

            self.imputer = MIWAE(random_state=random_state)

        elif imputation_method == "GAIN":
            from models.gain.gain import GainImputation

            # generate a unique filename for the model checkpoint
            filename_00 = str(uuid.uuid4())
            self.imputer = GainImputation(
                "gain_" + filename_00, alpha=100, random_state=random_state
            )

        else:
            raise ValueError(f"Unrecognised imputation method {imputation_method}")

    def fit(self, data, cat_vars=None):
        # Convert to numpy and fit the imputation method
        data = copy.deepcopy(data)
        self.cat_vars = cat_vars
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        assert isinstance(data, np.ndarray)

        # Pass categorical features if this is supported by the chosen imputation method
        if self.imputation_method == "MissForest":
            self.imputer.fit(data, cat_vars=cat_vars)
        elif self.imputation_method == "GAIN":
            gain_normalized_data, self.gain_nomalization_params = normalization(data)
            self.imputer.fit(gain_normalized_data)
        else:
            self.imputer.fit(data)

    def impute(self, data, indicator=False) -> np.ndarray:
        data = copy.deepcopy(data)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        data_0 = np.nan_to_num(data)

        if self.imputation_method == "GAIN":
            gain_normalized_data, _ = normalization(data, self.gain_nomalization_params)
            imputed_data = self.imputer.transform(gain_normalized_data)
            imputed_data = renormalization(imputed_data, self.gain_nomalization_params)
            if self.cat_vars is not None:
                imputed_data = rounding(self.cat_vars, imputed_data)
            else:
                imputed_data = rounding_automatic(data_0, imputed_data)
        else:
            imputed_data = self.imputer.transform(data)

        if isinstance(imputed_data, pd.DataFrame):
            imputed_data = imputed_data.to_numpy()

        # mask = indicator of known fields i.e. np.array s.t.
        # 1 where data missing, 0 where data known
        mask = np.isnan(data)
        imputed_data = (1 - mask) * data_0 + mask * imputed_data
        # If user wants it, concatenate indicator of missing data to the result
        # (to let the classifier know which fields were imputed)
        if indicator:
            imputed_data = np.concatenate([imputed_data, np.isnan(data)], axis=1)
        return imputed_data
