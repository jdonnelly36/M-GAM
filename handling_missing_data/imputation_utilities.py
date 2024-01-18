import logging
import os
import subprocess

import impute_values
import numpy as np
import pandas as pd  # type: ignore
from data_loader import DataLoader
from utilities import ExperimentParams, generate_path
import timeit

logger = logging.getLogger("imputation")


def load_imputed_data(params: ExperimentParams):
    """
    Returns:
        np.array: imputed_train_x
        np.array: imputed_val_x
        np.array: imputed_test_x
    """
    imputed_dir = generate_path(params, prefix="IMPUTED_DATA")
    imputed_train_x = np.load(imputed_dir / "imputed_train_x.npy")
    imputed_val_x = np.load(imputed_dir / "imputed_val_x.npy")
    imputed_test_x = np.load(imputed_dir / "imputed_test_x.npy")

    return imputed_train_x, imputed_val_x, imputed_test_x


def impute_and_store(params: ExperimentParams, random_state: int = 0) -> None:
    if params.imputation == "MICE":
        res = run_MICE(params, random_state)
        return res

    imputed_dir = generate_path(params, prefix="IMPUTED_DATA")
    os.makedirs(imputed_dir, exist_ok=True)

    out_train_path = imputed_dir / "imputed_train_x.npy"
    out_val_path = imputed_dir / "imputed_val_x.npy"
    out_test_path = imputed_dir / "imputed_test_x.npy"

    '''if out_train_path.exists() and out_val_path.exists() and out_test_path.exists():
        logger.info("Already Imputed")
        return'''

    # Load incomplete data
    data_loader = DataLoader(params.dataset)
    train_x, _, val_x, _, test_x, _ = data_loader.load_data(params)

    one_hot = True
    if params.imputation == "MissForest":
        # We might not use one-hot encoding, depending on the dataset
        train_x, _ = data_loader.onehot_to_ord(train_x)
        val_x, _ = data_loader.onehot_to_ord(val_x)
        test_x, one_hot = data_loader.onehot_to_ord(test_x)

    cat_ind, ord_ind = data_loader.extract_cat_vars(one_hot=one_hot)

    if len(cat_ind) > 0 or len(ord_ind) > 0:
        cat_vars = cat_ind + ord_ind
    else:
        cat_vars = None

    # Perform imputation
    imp = impute_values.Imputer(params.imputation, random_state=random_state)
    start_time = timeit.default_timer()
    imp.fit(train_x, cat_vars=cat_vars)
    elapsed_fit = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    logger.info("Imputing training data")
    imputed_train_x = imp.impute(train_x)
    elapsed_train = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    logger.info("Imputing validation data")
    imputed_val_x = imp.impute(val_x)
    elapsed_val = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    logger.info("Imputing test data")
    imputed_test_x = imp.impute(test_x)
    elapsed_test = timeit.default_timer() - start_time

    # Convert back to one-hot encoding if needed
    if params.imputation == "MissForest":
        imputed_train_x = data_loader.ord_to_onehot(imputed_train_x)
        imputed_val_x = data_loader.ord_to_onehot(imputed_val_x)
        imputed_test_x = data_loader.ord_to_onehot(imputed_test_x)

    np.save(imputed_dir / "imputed_train_x", imputed_train_x)
    np.save(imputed_dir / "imputed_val_x", imputed_val_x)
    np.save(imputed_dir / "imputed_test_x", imputed_test_x)

    time_stats = {
        'time_to_fit': elapsed_fit,
        'time_for_train': elapsed_train,
        'time_for_val': elapsed_val,
        'time_for_test': elapsed_test,
        'random_state': random_state
    }
    print("Params: ", params)
    param_dict = params._asdict()
    for param_name in param_dict:
        param_val = param_dict[param_name]
        print(param_name, param_val)
        time_stats[param_name] = param_val

    return pd.DataFrame(time_stats, index=[0])


def run_MICE(params: ExperimentParams, random_state: int) -> None:
    data_loader = DataLoader(params.dataset)
    train_path, val_path, test_path = data_loader.data_paths(params)

    # avoid bug in MICE when the seed is too large
    random_state = np.mod(random_state, 2147483)

    imputed_dir = generate_path(params, prefix="IMPUTED_DATA")

    parts = ["train", "val", "test"]

    if all((imputed_dir / f"imputed_{part}_x.npy").exists() for part in parts):
        logger.info("Already Imputed")
        return

    imputed_dir = generate_path(params, prefix="IMPUTED_DATA")

    # Generate arguments for MICE R script
    args = [
        "--train",
        train_path,
        "--val",
        val_path,
        "--test",
        test_path,
        "--outdir",
        str(imputed_dir),
        "--random_state",
        str(random_state),
    ]

    # Add categorical vars
    cat_ind, ord_ind = data_loader.extract_cat_vars(one_hot=True)
    if len(cat_ind) > 0:
        cat_var_ind = [str(x + 1) for x in cat_ind]
        args.extend(["--bin_feat", "-".join(cat_var_ind)])
    if len(ord_ind) > 0:
        ord_var_ind = [str(x + 1) for x in ord_ind]
        args.extend(["--ord_feat", "-".join(ord_var_ind)])

    # Generate command for running MICE R script
    script_path = "MICE_impute.R"
    command = ["Rscript", "--vanilla", script_path] + args
    try:
        start_time = timeit.default_timer()
        subprocess.run(command, check=True)
        elapsed_total = timeit.default_timer() - start_time
    except subprocess.CalledProcessError:
        logger.error("MICE_impute.R exited with an error; experiment %s", params)
        return

    # Convert to .npy format
    for part in parts:
        imputed = pd.read_csv(imputed_dir / f"imputed_{part}_x.csv").to_numpy()
        np.save(imputed_dir / f"imputed_{part}_x.npy", imputed)
        (imputed_dir / f"imputed_{part}_x.csv").unlink()

    
    time_stats = {
        'time_overall': elapsed_total,
        'random_state': random_state
    }
    param_dict = params._asdict()
    for param_name in param_dict:
        param_val = param_dict[param_name]
        time_stats[param_name] = param_val
    
    return pd.DataFrame(time_stats, index=[0])
