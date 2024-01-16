#! /usr/bin/env python3

import argparse
import logging
from typing import Callable

import classify_local
import classify_wandb
from common import (ClassifierParamsGrid, ParamDict, build_parser_experiment,
                    process_args_experiment, round_arange, split_list_arg,
                    split_range_arg)
from utilities import init_logger

logger = logging.getLogger("imputation")
init_logger(logger)


add_subparser: list[Callable[[argparse._SubParsersAction], None]] = []
process_subargs: dict[str, Callable[[argparse.Namespace], ClassifierParamsGrid]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    build_parser_experiment(parser)
    parser.add_argument(
        "--wandb",
        help="Use wandb.ai to log experiment results, otherwise store results locally",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_entity",
        help="Name of an entity to use with wandb.sweep",
        type=str,
    )
    parser.add_argument(
        "--save_locally",
        help="If using --wandb, also store experiment results locally",
        action="store_true",
    )
    parser.add_argument(
        "--loglevel",
        help="Logging level to use; if not specified, default is WARNING",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
    )

    # A classifier must be specified on the command line
    subparsers = parser.add_subparsers(
        description="Classifier",
        help="Name of classifier to use",
        dest="classifier",
        required=True,
    )

    for adder in add_subparser:
        adder(subparsers)

    return parser


def add_subparser_logreg(subparsers: argparse._SubParsersAction) -> None:
    parser_logreg = subparsers.add_parser(
        "LogisticRegression",
        help="Use Logistic Regression classifier; unless otherwise stated, for each "
        "classifier parameter, the option gives a list of values to be used in "
        "the grid search, and is given as a comma-separated list in the form "
        "expected by range() or numpy.arange(), for example, '2,6' means use "
        "the values 2, 3, 4, 5 in the grid search. Giving a single value (with "
        "no commas) will use that single value.",
    )
    parser_logreg.add_argument(
        "--iterations",
        help="Maximum number of iterations; the default is '50,260,50'",
        type=str,
        default="50,260,50",
    )


def process_subargs_logreg(args: argparse.Namespace) -> ClassifierParamsGrid:
    cparams: ClassifierParamsGrid = {}
    cparams["max_iter"] = split_range_arg(args.iterations)
    return cparams


add_subparser.append(add_subparser_logreg)
process_subargs["LogisticRegression"] = process_subargs_logreg


def add_subparser_rf(subparsers: argparse._SubParsersAction) -> None:
    parser_rf = subparsers.add_parser(
        "RandomForest",
        help="Use Random Forest classifier; unless otherwise stated, for each "
        "classifier parameter, the option gives a list of values to be used in "
        "the grid search, and is given as a comma-separated list in the form "
        "expected by range() or numpy.arange(), for example, '2,6' means use "
        "the values 2, 3, 4, 5 in the grid search. Giving a single value (with "
        "no commas) will use that single value.",
    )
    parser_rf.add_argument(
        "--n_estimators",
        help="Number of estimators; the default is '20,95,10'",
        type=str,
        default="20,95,10",
    )
    parser_rf.add_argument(
        "--max_depth",
        help="Maximum depth of trees; the default is '3,5'",
        type=str,
        default="3,5",
    )
    parser_rf.add_argument(
        "--min_samples_split",
        help="Minimum number of samples to be split into a further level; "
        "the default is '2,5'",
        type=str,
        default="2,5",
    )
    parser_rf.add_argument(
        "--min_samples_leaf",
        help="Minimum number of samples allowed in a leaf; the default is '2,5'",
        type=str,
        default="2,5",
    )


def process_subargs_rf(args: argparse.Namespace) -> ClassifierParamsGrid:
    cparams: ClassifierParamsGrid = {}
    cparams["n_estimators"] = split_range_arg(args.n_estimators)
    cparams["max_depth"] = split_range_arg(args.max_depth)
    cparams["min_samples_split"] = split_range_arg(args.min_samples_split)
    cparams["min_samples_leaf"] = split_range_arg(args.min_samples_leaf)
    return cparams


add_subparser.append(add_subparser_rf)
process_subargs["RandomForest"] = process_subargs_rf


def add_subparser_xgb(subparsers: argparse._SubParsersAction) -> None:
    parser_xgb = subparsers.add_parser(
        "XGBoost",
        help="Use XGBoost classifier; unless otherwise stated, for each classifier "
        "parameter, the option gives a list of values to be used in the grid search, "
        "and is given as a comma-separated list in the form expected by range() or "
        "numpy.arange(), for example, '3,6' means use the values 3, 4, 5 in the "
        "grid search.  Giving a single value (with no commas) will use that single "
        "value.",
    )
    parser_xgb.add_argument(
        "--n_estimators",
        help="Number of estimators; the default is '50,410,50'",
        type=str,
        default="50,410,50",
    )
    parser_xgb.add_argument(
        "--max_depth",
        help="Maximum depth of trees; the default is '3,6'",
        type=str,
        default="3,6",
    )
    parser_xgb.add_argument(
        "--subsample",
        help="Subsample ratio of the training instances; the default is '0.5,1.05,0.1'",
        type=str,
        default="0.5,1.05,0.1",
    )


def process_subargs_xgb(args: argparse.Namespace) -> ClassifierParamsGrid:
    cparams: ClassifierParamsGrid = {}
    cparams["n_estimators"] = split_range_arg(args.n_estimators)
    cparams["max_depth"] = split_range_arg(args.max_depth)
    cparams["subsample"] = split_range_arg(args.subsample, float, round_arange)
    return cparams


add_subparser.append(add_subparser_xgb)
process_subargs["XGBoost"] = process_subargs_xgb


def add_subparser_ngb(subparsers: argparse._SubParsersAction) -> None:
    parser_ngb = subparsers.add_parser(
        "NGBoost",
        help="Use NGBoost classifier; unless otherwise stated, for each classifier "
        "parameter, the option gives a list of values to be used in the grid search, "
        "and is given as a comma-separated list in the form expected by range() or "
        "numpy.arange(), for example, '3,6' means use the values 3, 4, 5 in the "
        "grid search.  Giving a single value (with no commas) will use that single "
        "value.",
    )
    parser_ngb.add_argument(
        "--n_estimators",
        help="Number of estimators; the default is '50,410,50'",
        type=str,
        default="50,410,50",
    )
    parser_ngb.add_argument(
        "--learning_rate",
        help="Learning rate to use, given as a space-separated list of rates; "
        "the default is '0.0001 0.001 0.01 0.1'",
        type=str,
        default="0.0001 0.001 0.01 0.1",
    )
    parser_ngb.add_argument(
        "--minibatch_frac",
        help="Subsample ratio of the training instances; the default is '0.5,1.05,0.1'",
        type=str,
        default="0.5,1.05,0.1",
    )


def process_subargs_ngb(args: argparse.Namespace) -> ClassifierParamsGrid:
    cparams: ClassifierParamsGrid = {}
    cparams["n_estimators"] = split_range_arg(args.n_estimators)
    cparams["learning_rate"] = split_list_arg(args.learning_rate, float)
    cparams["minibatch_frac"] = split_range_arg(
        args.minibatch_frac, float, round_arange
    )
    return cparams


add_subparser.append(add_subparser_ngb)
process_subargs["NGBoost"] = process_subargs_ngb


def add_subparser_nn(subparsers: argparse._SubParsersAction) -> None:
    parser_nn = subparsers.add_parser(
        "NeuralNetwork",
        help="Use multi-layer perceptron classifier; unless otherwise stated, for "
        "each classifier parameter, the option gives a list of values to be used "
        "in the grid search, and is given as a comma-separated list in the form "
        "expected by range() or numpy.arange(), for example, '3,6' means use the "
        "values 3, 4, 5 in the grid search.  Giving a single value (with no commas) "
        "will use that single value.",
    )
    parser_nn.add_argument(
        "--hidden_layers",
        help="Number of hidden layers; the default is '1,4'",
        type=str,
        default="1,4",
    )
    parser_nn.add_argument(
        "--neurons",
        help="Number of neurons per hidden layer; the default is '20,105,20'",
        type=str,
        default="20,105,20",
    )
    parser_nn.add_argument(
        "--learning_rate_init",
        help="Initial learning rate to use with the Adam optimiser, given as a "
        "space-separated list of rates; the default is '0.001 0.01 0.1'",
        type=str,
        default="0.001 0.01 0.1",
    )


def process_subargs_nn(args: argparse.Namespace) -> ClassifierParamsGrid:
    cparams: ClassifierParamsGrid = {}
    cparams["number_of_hidden_layers"] = split_range_arg(args.hidden_layers)
    cparams["neurons_per_hidden_layer"] = split_range_arg(args.neurons)
    cparams["learning_rate_init"] = split_list_arg(args.learning_rate_init, float)
    return cparams


add_subparser.append(add_subparser_nn)
process_subargs["NeuralNetwork"] = process_subargs_nn


def parse_args() -> ParamDict:
    parser = build_parser()
    args = parser.parse_args()

    return process_args(args)


def process_args(args: argparse.Namespace) -> ParamDict:
    """Process arguments into parameter dictionary"""

    params: ParamDict = {}
    process_args_experiment(args, params)

    classifier = args.classifier
    params["classifier"] = classifier
    params["classifier_parameters"] = process_subargs[classifier](args)

    params["wandb"] = args.wandb
    params["wandb_entity"] = args.wandb_entity
    params["save_locally"] = args.save_locally

    logger.setLevel(args.loglevel)

    return params


def main() -> None:
    params = parse_args()
    logger.debug("params = %s", params)
    if params["wandb"]:
        classify_wandb.perform_classification(params)
    else:
        classify_local.perform_classification(params)


if __name__ == "__main__":
    main()
