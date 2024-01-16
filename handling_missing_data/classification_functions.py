import logging
from typing import Any, Optional

import classifiers
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from common import ClassifierParams, ExperimentParams
from data_loader import DataLoader
from imputation_utilities import load_imputed_data
from sklearn import metrics  # type: ignore

logger = logging.getLogger("imputation")


def remove_nan_rows(arr: np.ndarray, *arrs: tuple[np.ndarray]) -> list[np.ndarray]:
    """Remove rows in `arr` which contain a NaN, and corresponding rows in `arrs`.

    Returns the stripped arrays in the same order; a final array is also given
    listing the rows with NaNs.
    """

    # If the array is 2-dimensional, collapse each row
    if len(arr.shape) > 1:
        nan_rows = np.isnan(arr).any(axis=1)
    else:
        nan_rows = np.isnan(arr)
    outarrs = [arr[~nan_rows]]
    for extra_arr in arrs:
        outarrs.append(extra_arr[~nan_rows])
    outarrs.append(np.flatnonzero(nan_rows))
    return outarrs


def insert_nan_rows(arr: np.ndarray, nan_idx: np.ndarray) -> np.ndarray:
    outarr = arr
    for idx in nan_idx:
        outarr = np.insert(outarr, idx, np.nan)
    return outarr


def classify_experiment(
    experiment_params: ExperimentParams,
    classifier: str,
    classifier_params: ClassifierParams,
) -> Optional[dict[str, dict[str, Any]]]:
    logger.info("Classifying with params %s", classifier_params)
    logger.debug("Loading experiment data")
    data_loader = DataLoader(experiment_params.dataset)
    # data_load returns train_x, train_y0, val_x, val_y0, test_x, test_y0
    experiment_data = data_loader.load_data(experiment_params)

    results_val, results_test = [], []
    for m in range(experiment_params.repeats):
        experiment_params_m = experiment_params._replace(m=m)
        results_m = classify_single_imputation(
            experiment_params_m, classifier, classifier_params, experiment_data
        )
        if results_m is not None:
            results_val.append(results_m[0])
            results_test.append(results_m[1])

    if len(results_val) == 0:
        return None

    # Pooling: Average over results on multiple imputed datasets
    prediction_prob_val = np.nanmean(results_val, axis=0)
    prediction_prob_test = np.nanmean(results_test, axis=0)

    val_y0 = experiment_data[3]
    test_y0 = experiment_data[5]

    # we kill nan values in the prediction_prob vectors
    prediction_prob_val, val_y0, _ = remove_nan_rows(prediction_prob_val, val_y0)
    prediction_prob_test, test_y0, _ = remove_nan_rows(prediction_prob_test, test_y0)

    logger.debug("Evaluating classifiers")

    clf_results_val, clf_results_fig_val = evaluate_classifier(
        val_y0, prediction_prob_val
    )

    clf_results_test, clf_results_fig_test = evaluate_classifier(
        test_y0,
        prediction_prob_test,
    )

    results = {
        "clf_results": {"val": clf_results_val, "test": clf_results_test},
        "clf_results_fig": {"val": clf_results_fig_val, "test": clf_results_fig_test},
        "predictions": {"val": prediction_prob_val, "test": prediction_prob_test},
    }

    return results


def classify_single_imputation(
    experiment_params: ExperimentParams,
    classifier: str,
    classifier_params: ClassifierParams,
    experiment_data,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    imputed_train_x, imputed_val_x, imputed_test_x = load_imputed_data(
        experiment_params
    )

    logger.info("Classifying imputation %d", experiment_params.m)

    train_x, train_y0, val_x, val_y0, test_x, test_y0 = experiment_data

    imputed_train_x, train_y, _ = remove_nan_rows(imputed_train_x, train_y0)
    imputed_val_x, val_y, nan_val = remove_nan_rows(imputed_val_x, val_y0)
    imputed_test_x, test_y, nan_test = remove_nan_rows(imputed_test_x, test_y0)

    logger.debug("Classifying")

    model, fail_state = classifiers.fit_classifier(
        classifier, imputed_train_x, train_y, classifier_params
    )

    if fail_state == 1:
        logger.warning("Classifier failed on imputation %d", experiment_params.m)
        return None

    logger.debug("Getting prediction probabilities")
    # Make prediction probabilities
    prediction_prob_val = model.predict_proba(imputed_val_x)[:, 1]
    prediction_prob_test = model.predict_proba(imputed_test_x)[:, 1]

    prediction_prob_val = insert_nan_rows(prediction_prob_val, nan_val)
    prediction_prob_test = insert_nan_rows(prediction_prob_test, nan_test)

    return prediction_prob_val, prediction_prob_test


def evaluate_classifier(y_true, y_prob):
    """Evaluate the classifier performance

    Args:
        y_true (np.array or pd.Series): true classes
        y_prob (np.array or pd.Series): predicted probabilites from classifier
        use_wandb (bool, optional): Log to W&B. Defaults to False.
        wnb_prefix (str, optional): prefix for W&B stored results (eg. val, test).
           Defaults to ''.

    Returns:
        evaluation_metrics: dict contatining performance metrics of the classifier
        fig: matplotlib figure
    """
    # Get predicted class labels from probabilites
    y_pred = (y_prob >= 0.5).astype(np.int)
    evaluation_metrics = {}
    evaluation_metrics["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    evaluation_metrics["balanced_accuracy"] = metrics.balanced_accuracy_score(
        y_true, y_pred
    )
    evaluation_metrics["auc"] = metrics.roc_auc_score(y_true, y_prob)
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    evaluation_metrics["sensitivity"] = tp / (tp + fn)
    evaluation_metrics["specificity"] = tn / (tn + fp)
    evaluation_metrics["precision"] = tp / (tp + fp)
    evaluation_metrics["F1"] = metrics.f1_score(y_true, y_pred)
    evaluation_metrics["brier_score"] = metrics.brier_score_loss(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true")
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(
        ax=ax1
    )

    # Plot ROC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    ax2.plot(fpr, tpr, label="ROC")
    ax2.set_ylabel("sensitivity")
    ax2.set_xlabel("1-specificity")

    return evaluation_metrics, fig
