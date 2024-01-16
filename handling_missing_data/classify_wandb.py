"""
Manage the classification process using the wandb.ai service.
"""

import logging
from functools import partial
from typing import cast

import classification_functions
import matplotlib.pyplot as plt
import wandb
from common import (ClassifierParams, ClassifierParamsGrid, ExperimentParams,
                    Number, ParamDict, get_experiment_params_repeats)
from utilities import save_classification_results

logger = logging.getLogger("imputation")


def perform_classification(params: ParamDict) -> None:
    classifier = params["classifier"]

    for experiment in get_experiment_params_repeats(params):
        logger.info("Beginning %s classification for %s", classifier, experiment)

        tag = "_".join(
            [
                classifier,
                experiment.dataset,
                experiment.imputation,
                "train",
                str(experiment.train_percentage),
                "test",
                str(experiment.test_percentage),
                "h",
                str(experiment.holdout_set),
                "v",
                str(experiment.validation_set),
            ]
        )

        sweep_config = {
            "method": "grid",
            "metric": {"name": "loss", "goal": "minimize"},
            "name": tag,
            "parameters": add_values_level(params["classifier_parameters"]),
        }

        project_name = f"handling-missing-data-{experiment.dataset}-{classifier}"

        sweep_id = wandb.sweep(
            sweep_config,
            project=project_name,
            entity=params["wandb_entity"],
        )

        classify_function = partial(
            classify_experiment,
            experiment=experiment,
            params=params,
        )

        wandb.agent(sweep_id, classify_function)


def add_values_level(cparams: ClassifierParamsGrid) -> dict[str, ClassifierParamsGrid]:
    """Add extra "values" level to parameter grid dictionary.

    wandb requires the "parameters" entry to have a dict of dicts.  This function
    adds the extra level.
    """
    return {param: {"values": vals_list} for param, vals_list in cparams.items()}


def classify_experiment(
    experiment: ExperimentParams,
    params: ParamDict,
) -> None:
    classifier = params["classifier"]

    wandb.init()
    if wandb.run is None:
        logger.error("wandb failed to initialise")
        return

    wandb.run.name = "_".join([experiment.dataset, experiment.imputation, classifier])
    hparams = wandb.config._as_dict()
    wandb.config.update(experiment._asdict())
    wandb.config.update({"classifier": classifier})
    cparams: ClassifierParams = {
        k: cast(Number, hparams[k]) for k in params["classifier_parameters"].keys()
    }

    results = classification_functions.classify_experiment(
        experiment, classifier, cparams
    )

    if results is None:
        wandb.finish()
        return

    if params["save_locally"]:
        save_classification_results(experiment, classifier, cparams, results)

    logged_prediction_prob_val = wandb.Table(
        data=list(enumerate(results["predictions"]["val"])), columns=["id", "val"]
    )
    logged_prediction_prob_test = wandb.Table(
        data=list(enumerate(results["predictions"]["test"])), columns=["id", "test"]
    )

    wandb.log(
        {
            "prediction_prob_val": logged_prediction_prob_val,
            "prediction_prob_test": logged_prediction_prob_test,
            "val": results["clf_results"]["val"],
            "val_plots": results["clf_results_fig"]["val"],
            "test": results["clf_results"]["test"],
            "test_plots": results["clf_results_fig"]["test"],
            "loss": results["clf_results"]["val"]["auc"],
        }
    )

    plt.close(results["clf_results_fig"]["val"])
    plt.close(results["clf_results_fig"]["test"])

    wandb.finish()
