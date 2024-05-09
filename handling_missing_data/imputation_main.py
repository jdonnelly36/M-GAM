#! /usr/bin/env python3

import argparse
import logging

from common import (ParamDict, build_parser_experiment, get_experiment_params,
                    process_args_experiment)
from imputation_utilities import impute_and_store
from utilities import init_logger
import pandas as pd

logger = logging.getLogger("imputation")
init_logger(logger)


def parse_args():
    parser = argparse.ArgumentParser()
    build_parser_experiment(parser)
    parser.add_argument(
        "--loglevel",
        help="Logging level to use; if not specified, default is WARNING",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
    )

    args = parser.parse_args()
    return process_args(args)


def split_list_arg(arg, convert):
    components = arg.split()
    return list(map(convert, components))


def process_args(args):
    """Process arguments into parameter dictionary"""

    params: ParamDict = {}
    process_args_experiment(args, params)

    logger.setLevel(args.loglevel)

    return params


def perform_imputations(params):
    overall_time_states = None
    for experiment in get_experiment_params(params):
        logger.info("Beginning imputation for %s", experiment)
        try:
            overall_time_states = pd.read_csv(f'timing_stats_{experiment.dataset}_{experiment.imputation}_5_3.csv')
            mask_for_experiment = (
                (overall_time_states['m'] == experiment.m) &
                (overall_time_states['dataset'] == experiment.dataset) &
                (overall_time_states['holdout_set'] == experiment.holdout_set) &
                (overall_time_states['imputation'] == experiment.imputation) &
                (overall_time_states['validation_set'] == experiment.validation_set)
            )
            run_experiment = overall_time_states[mask_for_experiment].shape[0] == 0
        except:
            overall_time_states = None
        finally:
            if overall_time_states is None or run_experiment:
                time_stats = impute_and_store(experiment, random_state=experiment.m)
                if overall_time_states is None:
                    overall_time_states = time_stats
                else:
                    overall_time_states = pd.concat([overall_time_states, time_stats], axis=0)
                    overall_time_states.to_csv(f'timing_stats_{experiment.dataset}_{experiment.imputation}_5_3.csv', index=False)
            else:
                print(f"Skipping experiment for {experiment}")


def main():
    params = parse_args()
    perform_imputations(params)


if __name__ == "__main__":
    main()
