"""
Some definitions and functions common to both imputation and classification,
for example specifying command line options.
"""

import argparse
from itertools import product
from typing import Any, Generator, NamedTuple, Union

import numpy as np

ParamDict = dict[str, Any]
Number = Union[int, float]
ClassifierParamsGrid = dict[str, list[Number]]
ClassifierParams = dict[str, Number]

DATASETS = [
    "SYNTHETIC",
    "SYNTHETIC_CATEGORICAL",
    "SYNTHETIC_MAR",
    "SYNTHETIC_CATEGORICAL_MAR",
    "SYNTHETIC_MAR_25",
    "SYNTHETIC_CATEGORICAL_MAR_25",
    "SYNTHETIC_MAR_50",
    "SYNTHETIC_CATEGORICAL_MAR_50",
    "MIMIC",
    "NHSX_COVID19",
    "BREAST_CANCER",
    "FICO", 
    "FICO_MAR", 
    "BREAST_CANCER_MAR", 
    "BREAST_CANCER_MAR_pt4", 
    "FICO_MAR_25", 
    "FICO_MAR_50",
    "BREAST_CANCER_MAR_25", 
    "BREAST_CANCER_MAR_50", 
    "APS"
]
IMPUTATIONS = ["Mean", "MICE", "MissForest", "MIWAE", "GAIN"]
PERCENTAGES = {
    "SYNTHETIC": [0.25, 0.5],
    "SYNTHETIC_CATEGORICAL": [0.25, 0.5],
    "SYNTHETIC_MAR": [0, 0.25, 0.5],
    "SYNTHETIC_CATEGORICAL_MAR": [0.25, 0.5],
    "SYNTHETIC_MAR_25": [0, 0.25, 0.5],
    "SYNTHETIC_CATEGORICAL_MAR_25": [0.25, 0.5],
    "SYNTHETIC_MAR_50": [0, 0.25, 0.5],
    "SYNTHETIC_CATEGORICAL_MAR_50": [0.25, 0.5],
    "MIMIC": [0],
    "NHSX_COVID19": [0],
    "BREAST_CANCER": [0],
    "FICO": [0],
    "FICO_MAR": [0],
    "BREAST_CANCER_MAR": [0],
    "BREAST_CANCER_MAR_pt4": [0],
    "FICO_MAR_25": [0], 
    "FICO_MAR_50": [0],
    "BREAST_CANCER_MAR_25": [0],
    "BREAST_CANCER_MAR_50": [0], 
    'APS': [0]
}
N_HOLDOUT_SETS = 3
N_VAL_SETS = 5  # 5 validation sets per holdout set
N_IMPUTED_SETS = 10  # number of imputed sets used for multiple imputation


# We need "repeats" in here as we combine all of the multiple imputations when
# evaluating classifiers, so we have to be able to generate down to either
# repeats level or individual imputation level (`m`).
class ExperimentParams(NamedTuple):
    dataset: str
    imputation: str
    train_percentage: float
    test_percentage: float
    holdout_set: int
    validation_set: int
    repeats: int
    m: int


def build_parser_experiment(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        "-d",
        help="Experiments to analyse: SYNTHETIC, SYNTHETIC_CATEGORICAL, MIMIC, "
        "NHSX_COVID19, BREAST_CANCER; a space-separated string; default is all",
        type=str,
        default=" ".join(DATASETS),
    )
    parser.add_argument(
        "--imputation",
        "-i",
        help="Imputation methods to use: Mean, MICE, MissForest, MIWAE, GAIN; "
        "a space-separated string; default is all",
        type=str,
        default=" ".join(IMPUTATIONS),
    )
    parser.add_argument(
        "--train_percent",
        help="Proportions of missing data during training as a space-separated list; "
        "if not specified, sensible defaults will be used",
        type=str,
    )
    parser.add_argument(
        "--test_percent",
        help="Proportions of missing data during testing as a space-separated list; "
        "if not specified, sensible defaults will be used",
        type=str,
    )
    parser.add_argument(
        "--holdouts",
        help="Holdout sets to use as a space-separated list; if not specified, "
        "default is '0 1 2'",
        type=str,
        default=" ".join(map(str, range(N_HOLDOUT_SETS))),
    )
    parser.add_argument(
        "--val_sets",
        help="Validation sets to use as a space-separated list; if not specified, "
        "default is '0 1 2 3 4'",
        type=str,
        default=" ".join(map(str, range(N_VAL_SETS))),
    )
    parser.add_argument(
        "--repeats",
        help="Number of repeats to use for multiple imputation; "
        "if not specified, default is 10",
        type=int,
        default=N_IMPUTED_SETS,
    )


def split_list_arg(arg, convert):
    components = arg.split()
    return list(map(convert, components))


def split_range_arg(arg, convert=int, range_func=range):
    components = arg.split(",")
    if len(components) == 1:
        return [convert(components[0])]
    numerical_components = map(convert, components)
    # wandb cannot cope with numpy.float64 in a sweep_config, so we must
    # convert the output list into int or float before returning it; this is
    # a no-op when "range" is used, but it is important when round_arange is
    # used.
    return list(map(convert, range_func(*numerical_components)))


def round_arange(*args, decimals=3):
    unrounded = np.arange(*args)
    rounded = np.around(unrounded, decimals=decimals)
    return rounded


def process_args_experiment(args: argparse.Namespace, params: ParamDict) -> None:
    params["dataset"] = split_list_arg(args.dataset, str)
    params["imputation"] = split_list_arg(args.imputation, str)

    if args.train_percent is None:
        params["train_percentage"] = lambda dataset: PERCENTAGES[dataset]
    else:
        train_percents = split_list_arg(args.train_percent, float)
        params["train_percentage"] = lambda dataset: train_percents

    if args.test_percent is None:
        params["test_percentage"] = lambda dataset: PERCENTAGES[dataset]
    else:
        test_percents = split_list_arg(args.test_percent, float)
        params["test_percentage"] = lambda dataset: test_percents

    params["holdout_set"] = split_list_arg(args.holdouts, int)
    params["validation_set"] = split_list_arg(args.val_sets, int)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    params["repeats"] = args.repeats


def get_experiment_params(params: dict) -> Generator[ExperimentParams, None, None]:
    """An iterator giving the experiment parameters to use.

    Unfortunately we can't use a simple itertools.product over the params dict,
    as the test/train percentages depend on the dataset.
    """
    for dataset in params["dataset"]:
        for imputation, train, test, holdout, valid, m in product(
            params["imputation"],
            params["train_percentage"](dataset),
            params["test_percentage"](dataset),
            params["holdout_set"],
            params["validation_set"],
            range(params["repeats"]),
        ):
            yield ExperimentParams(
                dataset=dataset,
                imputation=imputation,
                train_percentage=train,
                test_percentage=test,
                holdout_set=holdout,
                validation_set=valid,
                repeats=params["repeats"],
                m=m,
            )


def get_experiment_params_repeats(
    params: dict,
) -> Generator[ExperimentParams, None, None]:
    """An iterator giving the experiment parameters to use, down to `repeats` level."""
    for dataset in params["dataset"]:
        for imputation, train, test, holdout, valid in product(
            params["imputation"],
            params["train_percentage"](dataset),
            params["test_percentage"](dataset),
            params["holdout_set"],
            params["validation_set"],
        ):
            yield ExperimentParams(
                dataset=dataset,
                imputation=imputation,
                train_percentage=train,
                test_percentage=test,
                holdout_set=holdout,
                validation_set=valid,
                repeats=params["repeats"],
                m=0,
            )
