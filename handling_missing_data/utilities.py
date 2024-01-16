import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from common import ClassifierParams, ExperimentParams


def init_logger(logger):
    ch = logging.StreamHandler()
    ch.setLevel(logging.NOTSET)

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def get_current_path() -> Path:
    """
    Return the absolute path of the current directory as Path.
    """
    cur_frame = inspect.currentframe()
    if cur_frame is None:
        raise Exception("Cannot determine current directory")
    cur_file = Path(inspect.getfile(cur_frame))
    return cur_file.resolve().parent


def create_results_directory(
    experiment: ExperimentParams, classifier_params: ClassifierParams, classifier: str
) -> Path:
    """
    Create directory for results; this has a unique name within an appropriate
    subdirectory of the CLASSIFICATION_RESULTS directory.
    """
    base = generate_path(experiment, prefix="CLASSIFICATION_RESULTS", include_m=False)
    param_str = "_".join([f"{k[:2]}_{v}" for k, v in classifier_params.items()])
    date_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    directory = base / classifier / f"{param_str}_{date_str}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_classification_results(
    experiment: ExperimentParams,
    classifier: str,
    classifier_params: ClassifierParams,
    results: dict[str, dict[str, Any]],
) -> None:
    directory = create_results_directory(experiment, classifier_params, classifier)

    with open(directory / "config.json", "w", encoding="UTF-8") as outfile:
        config = experiment._asdict()
        config["classifier"] = classifier
        config.update(classifier_params)
        json.dump(config, outfile)

    with open(directory / "clf_results_val.json", "w", encoding="UTF-8") as outfile:
        json.dump(results["clf_results"]["val"], outfile)

    with open(directory / "clf_results_test.json", "w", encoding="UTF-8") as outfile:
        json.dump(results["clf_results"]["test"], outfile)

    results["clf_results_fig"]["val"].savefig(directory / "clf_results_val.png")
    results["clf_results_fig"]["test"].savefig(directory / "clf_results_test.png")
    np.savetxt(directory / "preds_val.csv", results["predictions"]["val"])
    np.savetxt(directory / "preds_test.csv", results["predictions"]["test"])


def format_value(val):
    if isinstance(val, float):
        return f"{val:.2}"
    return str(val)


def generate_path(params: ExperimentParams, prefix="", include_m=True):
    path = (
        Path(prefix)
        / params.dataset
        / params.imputation
        / f"train_per_{params.train_percentage}"
        / f"test_per_{params.test_percentage}"
        / f"holdout_{params.holdout_set}"
        / f"val_{params.validation_set}"
    )
    if include_m:
        path = path / f"m_{params.m}"

    return path
