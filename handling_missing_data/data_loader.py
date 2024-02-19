import json
from abc import ABCMeta, abstractmethod
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
from common import ExperimentParams
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore
from utilities import get_current_path

dataloaders: dict[str, type] = {}


class DataLoader:
    def __init__(self, dataset):
        if dataset not in dataloaders:
            raise ValueError(f"Unrecognised dataset type: {dataset}")
        self.dataloader = dataloaders[dataset]()

    def __getattr__(self, attrname):
        return getattr(self.dataloader, attrname)


class _DataLoaderBase(metaclass=ABCMeta):
    def __init__(self):
        self.datadir = ""
        self.outcome_col = ""
        self.cols_path = None

    def extract_cat_vars(self, one_hot: bool) -> tuple[list[int], list[int]]:
        """Returns the categorical variable column indices.

        They are returned for the two formats: categorical and ordinal.
        """
        return [], []

    @abstractmethod
    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        pass

    def _data_paths_natural(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        """Get paths to data with genuine missingness"""
        holdout = experiment.holdout_set
        validation = experiment.validation_set
        path_temp = get_current_path() / self.datadir
        train_path = path_temp / f"devel_{holdout}_train_{validation}.csv"
        val_path = path_temp / f"devel_{holdout}_val_{validation}.csv"
        test_path = path_temp / f"holdout_{holdout}.csv"

        return (train_path, val_path, test_path)

    def _data_paths_artificial(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        """Get paths to data with artificially introduced missingness"""
        train_percentage = experiment.train_percentage
        test_percentage = experiment.test_percentage

        if isclose(train_percentage, 0.0):
            train_percentage = 0
        elif isclose(train_percentage, 0.50):
            train_percentage = 0.5

        if isclose(test_percentage, 0.0):
            test_percentage = 0
        elif isclose(test_percentage, 0.50):
            test_percentage = 0.5

        missing_str = f"train_missing_{train_percentage}_test_missing_{test_percentage}"

        holdout = experiment.holdout_set
        validation = experiment.validation_set
        path_temp = get_current_path() / self.datadir
        train_path = path_temp / f"devel_{holdout}_train_{validation}_{missing_str}.csv"
        val_path = path_temp / f"devel_{holdout}_val_{validation}_{missing_str}.csv"
        test_path = path_temp / f"holdout_{holdout}_{missing_str}.csv"

        return (train_path, val_path, test_path)

    def load_data(self, experiment: ExperimentParams) -> tuple[pd.DataFrame, ...]:
        """Load training data with genuine missingness"""
        data_paths = self.data_paths(experiment)
        train = pd.read_csv(data_paths[0])
        val = pd.read_csv(data_paths[1])
        test = pd.read_csv(data_paths[2])

        train_x, train_y = (
            train.drop(columns=[self.outcome_col]),
            train[self.outcome_col],
        )
        val_x, val_y = (
            val.drop(columns=[self.outcome_col]),
            val[self.outcome_col],
        )
        test_x, test_y = (
            test.drop(columns=[self.outcome_col]),
            test[self.outcome_col],
        )
        return (train_x, train_y, val_x, val_y, test_x, test_y)

    def onehot_to_ord(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Transform data from one-hot encoded to ordinal encoded.

        Returns the transformed data and a boolean one_hot, which is True if
        still one-hot encoded (no transformation was necessary) and False if
        the data is transformed.

        This is the default method, in which we do not do any encoding.
        """
        return df, True

    def ord_to_onehot(self, data: np.ndarray) -> np.ndarray:
        """Transform data from ordinal encoded back to one-hot encoded.

        This method processes the imputed data, so it takes as input an
        ndarray rather than a DataFrame.

        This is the default method, in which we do not do any encoding.
        """
        return data

    # The following methods are used by BREAST_CANCER and SYNTHETIC_CATEGORICAL
    # to handle one-hot encoding of categorical variables.

    def onehot_to_ord_multicat(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Transform data from one-hot encoded to ordinal encoded.

        Returns the transformed data and a boolean is_one_hot, which is True if
        still one-hot encoded (no transformation was necessary) and False if
        the data is transformed.

        This method is called by `onehot_to_ord` for datasets which actually use
        one-hot encoding.
        """
        self.setup_encoders()
        cls = type(self)
        onehot_encoder = getattr(cls, "onehot_encoder")
        ordinal_encoder = getattr(cls, "ordinal_encoder")
        encoded_colnames = getattr(cls, "encoded_colnames")
        cat_colnames = getattr(cls, "cat_colnames")

        onehot_df = df[encoded_colnames]
        oh_decoded = onehot_encoder.inverse_transform(onehot_df)
        # silence warning in ordinal_encoder.transform
        oh_decoded_df = pd.DataFrame(oh_decoded, columns=cat_colnames, index=df.index)
        ord_df = ordinal_encoder.transform(oh_decoded_df)
        ord_df = pd.DataFrame(ord_df, columns=cat_colnames, index=df.index)
        rest_df = df.drop(encoded_colnames, axis=1)
        converted_df = pd.concat([rest_df, ord_df], axis=1)
        return converted_df, False

    def ord_to_onehot_multicat(self, data: np.ndarray) -> np.ndarray:
        """Transform data from ordinal encoded back to one-hot encoded.

        This method processes the imputed data, so it takes as input an
        ndarray rather than a DataFrame.

        This method is called by `ord_to_onehot` for datasets which actually use
        one-hot encoding.
        """
        self.setup_encoders()
        cls = type(self)
        onehot_encoder = getattr(cls, "onehot_encoder")
        ordinal_encoder = getattr(cls, "ordinal_encoder")
        encoded_colnames = getattr(cls, "encoded_colnames")
        cat_colnames = getattr(cls, "cat_colnames")
        cols = getattr(cls, "cols")
        full_colnames = cols["colnames"]

        # This differs from the corresponding code in onehot_to_ord_multicat
        # as we are starting with an ndarray without column names.
        df = pd.DataFrame(data, columns=full_colnames["encoded"])
        ord_df = df[cat_colnames]
        ord_decoded = ordinal_encoder.inverse_transform(ord_df)
        # silence warning in onehot_encoder.transform
        ord_decoded_df = pd.DataFrame(ord_decoded, columns=cat_colnames, index=df.index)
        onehot_df = onehot_encoder.transform(ord_decoded_df)
        onehot_df = pd.DataFrame(onehot_df, columns=encoded_colnames, index=df.index)
        rest_df = df.drop(cat_colnames, axis=1)
        converted_df = pd.concat([rest_df, onehot_df], axis=1)
        reordered_df = converted_df[full_colnames["onehot"]]
        return reordered_df.to_numpy()

    def get_cols_data(self):
        """Get data on categorical and ordinal columns"""
        cls = type(self)
        cols = getattr(cls, "cols")
        if cols is not None:
            return cols

        cols_path = getattr(cls, "cols_path")
        assert cols_path is not None

        jsonfile = get_current_path() / self.datadir / cols_path
        with open(jsonfile, encoding="UTF-8") as colsfile:
            cols = json.load(colsfile)
        setattr(cls, "cols", cols)

        return cols

    def setup_encoders(self) -> None:
        cls = type(self)
        if getattr(cls, "onehot_encoder") is not None:
            return

        factor_levels_path = getattr(cls, "factor_levels_path")
        jsonfile = get_current_path() / self.datadir / factor_levels_path
        with open(jsonfile, encoding="UTF-8") as colsfile:
            factor_levels = json.load(colsfile)
        # This is stored as [(factor, levels), (factor, levels), ...]
        factors = [fl[0] for fl in factor_levels]
        levels = [fl[1] for fl in factor_levels]

        # sklearn requires us to fit a non-empty DataFrame even if we specify all
        # the levels
        dummy_df = pd.DataFrame({fl[0]: [fl[1][0]] for fl in factor_levels})
        setattr(cls, "cat_colnames", factors)
        # building the model for transformations
        ohe = OneHotEncoder(categories=levels, sparse=False)
        setattr(cls, "onehot_encoder", ohe.fit(dummy_df))
        setattr(cls, "encoded_colnames", ohe.get_feature_names_out(factors))
        # building LabelEncoder dictionary model
        orde = OrdinalEncoder(categories=levels)
        setattr(cls, "ordinal_encoder", orde.fit(dummy_df))

        # We will also need the column names for the full dataset
        self.get_cols_data()


class DataLoaderMIMIC(_DataLoaderBase):
    def __init__(self):
        super().__init__()
        self.datadir = "DATA/MIMIC_III/MIMIC_subset_mcar"
        self.outcome_col = "outcome"

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_artificial(experiment)


dataloaders["MIMIC"] = DataLoaderMIMIC


class DataLoaderSynthetic(_DataLoaderBase):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC/SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE"
        )
        self.outcome_col = "output"

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_artificial(experiment)


dataloaders["SYNTHETIC"] = DataLoaderSynthetic

class DataLoaderSynthetic_MAR(DataLoaderSynthetic):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE"
        )


dataloaders["SYNTHETIC_MAR"] = DataLoaderSynthetic_MAR

class DataLoaderSynthetic_MAR_25(DataLoaderSynthetic):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE/0.25"
        )

dataloaders["SYNTHETIC_MAR_25"] = DataLoaderSynthetic_MAR_25

class DataLoaderSynthetic_MAR_50(DataLoaderSynthetic):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_1000_SAMPLES_25_FEATURES_25_INFORMATIVE/0.5"
        )

dataloaders["SYNTHETIC_MAR_50"] = DataLoaderSynthetic_MAR_50


class DataLoaderSyntheticCategorical(_DataLoaderBase):
    onehot_encoder = None
    ordinal_encoder = None
    encoded_colnames = None
    cat_colnames = None
    cols = None
    cols_path = "synthetic_categorical_cols.json"
    factor_levels_path = "synthetic_categorical_factor_levels.json"

    def __init__(self):
        super().__init__()
        self.datadir = "DATA/SYNTHETIC/SYNTHETIC_CATEGORICAL"
        self.outcome_col = "output"

    def extract_cat_vars(self, one_hot: bool) -> tuple[list[int], list[int]]:
        cols = self.get_cols_data()

        if one_hot:
            cols_onehot = cols["onehot"]
            return cols_onehot[0], cols_onehot[1]

        cols_encoded = cols["encoded"]
        return cols_encoded[0], cols_encoded[1]

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_artificial(experiment)

    def onehot_to_ord(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        return self.onehot_to_ord_multicat(df)

    def ord_to_onehot(self, data: np.ndarray) -> np.ndarray:
        return self.ord_to_onehot_multicat(data)


dataloaders["SYNTHETIC_CATEGORICAL"] = DataLoaderSyntheticCategorical

class DataLoaderSyntheticCategorical_MAR(DataLoaderSyntheticCategorical):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_CATEGORICAL"
        )


dataloaders["SYNTHETIC_CATEGORICAL_MAR"] = DataLoaderSyntheticCategorical_MAR

class DataLoaderSyntheticCategorical_MAR_25(DataLoaderSyntheticCategorical):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_CATEGORICAL/0.25"
        )


dataloaders["SYNTHETIC_CATEGORICAL_MAR_25"] = DataLoaderSyntheticCategorical_MAR_25

class DataLoaderSyntheticCategorical_MAR_50(DataLoaderSyntheticCategorical):
    def __init__(self):
        super().__init__()
        self.datadir = (
            "DATA/SYNTHETIC_MAR/SYNTHETIC_CATEGORICAL/0.5"
        )


dataloaders["SYNTHETIC_CATEGORICAL_MAR_50"] = DataLoaderSyntheticCategorical_MAR_50

class DataLoaderNHSX(_DataLoaderBase):
    cols = None
    cols_path = "NHSX_Covid_cols.json"

    def __init__(self):
        super().__init__()
        self.datadir = "DATA/NHSX_COVID19"
        self.outcome_col = "Death"

    def extract_cat_vars(self, one_hot: bool) -> tuple[list[int], list[int]]:
        cols = self.get_cols_data()

        return cols[0], cols[1]

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)


dataloaders["NHSX_COVID19"] = DataLoaderNHSX


class DataLoaderBreast(_DataLoaderBase):
    onehot_encoder = None
    ordinal_encoder = None
    encoded_colnames = None
    cat_colnames = None
    cols = None
    cols_path = "Breast_cancer_cols.json"
    factor_levels_path = "Breast_factor_levels.json"

    def __init__(self):
        super().__init__()
        self.datadir = "DATA/BREAST_CANCER"
        self.outcome_col = "Overall Survival Status"

    def extract_cat_vars(self, one_hot: bool) -> tuple[list[int], list[int]]:
        cols = self.get_cols_data()

        if one_hot:
            cols_onehot = cols["onehot"]
            return cols_onehot[0], cols_onehot[1]

        cols_encoded = cols["encoded"]
        return cols_encoded[0], cols_encoded[1]

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)

    def onehot_to_ord(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        return self.onehot_to_ord_multicat(df)

    def ord_to_onehot(self, data: np.ndarray) -> np.ndarray:
        return self.ord_to_onehot_multicat(data)


dataloaders["BREAST_CANCER"] = DataLoaderBreast

class DataLoaderBreast_MAR(_DataLoaderBase):
    onehot_encoder = None
    ordinal_encoder = None
    encoded_colnames = None
    cat_colnames = None
    cols = None
    cols_path = "Breast_cancer_cols.json"
    factor_levels_path = "Breast_factor_levels.json"

    def __init__(self):
        super().__init__()
        self.datadir = "DATA/BREAST_CANCER_MAR"
        self.outcome_col = "Overall Survival Status"

    def extract_cat_vars(self, one_hot: bool) -> tuple[list[int], list[int]]:
        cols = self.get_cols_data()

        if one_hot:
            cols_onehot = cols["onehot"]
            return cols_onehot[0], cols_onehot[1]

        cols_encoded = cols["encoded"]
        return cols_encoded[0], cols_encoded[1]

    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)

    def onehot_to_ord(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        return self.onehot_to_ord_multicat(df)

    def ord_to_onehot(self, data: np.ndarray) -> np.ndarray:
        return self.ord_to_onehot_multicat(data)


dataloaders["BREAST_CANCER_MAR"] = DataLoaderBreast_MAR

class DataLoaderBreast_MAR_pt4(DataLoaderBreast_MAR):
    def __init__(self):
        super().__init__()
        self.datadir = "DATA/BREAST_CANCER_MAR_pt4"

dataloaders["BREAST_CANCER_MAR_pt4"] = DataLoaderBreast_MAR_pt4

class DataLoaderBreast_MAR_25(DataLoaderBreast_MAR):
    def __init__(self):
        super().__init__()
        self.datadir = "DATA/BREAST_CANCER_MAR/0.25"

dataloaders["BREAST_CANCER_MAR_25"] = DataLoaderBreast_MAR_25

class DataLoaderBreast_MAR_50(DataLoaderBreast_MAR):
    def __init__(self):
        super().__init__()
        self.datadir = "DATA/BREAST_CANCER_MAR/0.5"

dataloaders["BREAST_CANCER_MAR_50"] = DataLoaderBreast_MAR_50

class DataLoaderPharyngitis(_DataLoaderBase): 
    def __init__(self):
        super().__init__()
        self.datadir = ("DATA/PHARYNGITIS")
        self.outcome_col = "radt"
    
    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)

dataloaders["PHARYNGITIS"] = DataLoaderPharyngitis

class DataLoaderFico(_DataLoaderBase): 
    def __init__(self):
        super().__init__()
        self.datadir = ("DATA/FICO")
        self.outcome_col = "PoorRiskPerformance"
    
    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)

dataloaders["FICO"] = DataLoaderFico

class DataLoaderFico_MAR(_DataLoaderBase): 
    def __init__(self):
        super().__init__()
        self.datadir = ("DATA/FICO_MAR")
        self.outcome_col = "PoorRiskPerformance"
    
    def data_paths(self, experiment: ExperimentParams) -> tuple[Path, ...]:
        return self._data_paths_natural(experiment)

dataloaders["FICO_MAR"] = DataLoaderFico_MAR

class DataLoaderFico_MAR_25(DataLoaderFico_MAR): 
    def __init__(self):
        super().__init__()
        self.datadir = ("DATA/FICO_MAR/0.25")
dataloaders["FICO_MAR_25"] = DataLoaderFico_MAR_25

class DataLoaderFico_MAR_50(DataLoaderFico_MAR): 
    def __init__(self):
        super().__init__()
        self.datadir = ("DATA/FICO_MAR/0.5")
dataloaders["FICO_MAR_50"] = DataLoaderFico_MAR_50
    