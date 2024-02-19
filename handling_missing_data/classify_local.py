"""
Manage the classification process, storing results locally.
"""

import logging
from itertools import product
from typing import cast

import classification_functions
import matplotlib.pyplot as plt
from common import (ClassifierParams, ClassifierParamsGrid, ExperimentParams,
                    ParamDict, get_experiment_params_repeats)
from utilities import save_classification_results

logger = logging.getLogger("imputation")


def perform_classification(params: ParamDict) -> None:
    classifier = params["classifier"]

    for experiment in get_experiment_params_repeats(params):
        logger.info(
            "Beginning %s classification for experiment %s", classifier, experiment
        )

        parameter_grid = cast(ClassifierParamsGrid, params["classifier_parameters"])
        parameter_names = list(parameter_grid.keys())
        for cparam_values in product(*[parameter_grid[p] for p in parameter_names]):
            cparams = dict(zip(parameter_names, cparam_values))
            classify_experiment(experiment, classifier, cparams)


def classify_experiment(
    experiment: ExperimentParams,
    classifier: str,
    cparams: ClassifierParams,
) -> None:
    results = classification_functions.classify_experiment(
        experiment, classifier, cparams
    )
    if results is not None:
        save_classification_results(experiment, classifier, cparams, results)

        plt.close(results["clf_results_fig"]["val"])
        plt.close(results["clf_results_fig"]["test"])
