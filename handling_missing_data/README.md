# Classification of datasets with imputed missing values
DOI: 10.5281/zenodo.8234032

This repository has data and scripts to perform imputation on datasets
with missing data, and then to classify the resulting imputed
datasets.  This repository forms part of the supplementary material
for the paper:

Shadbahr, T. and Roberts, M. and Stanczuk, J. and Gilbey, J. and Teare, P. et al., "The Impact of Imputation Quality on Machine Learning Classifiers for Datasets with Missing Values"

## System requirements

This software should run on any operating system, though it has only
been tested on Linux and macOS systems.  Having a CUDA-compatible GPU
will make some of the computations faster.

The code requires Python 3 to be installed; it has been tested with
Python 3.9 and 3.10, but is likely to work with slightly older Python
versions.  Some of the packages used have not yet been updated to work
with Python 3.11, though.

For imputation using MICE, R needs to be installed.

## Installing the required packages

It is recommended to run this code in a virtual environment.  On a
UNIX-like system (macOS or Linux), run these three commands in a
suitable terminal:

        python3 -m venv imputation-env

        source imputation-env/bin/activate

        pip install -r requirements.txt

to create your virtual environment and install the required Python
packages.  (You may have to use `python` rather than `python3` in the
first command, depending on the setup of your system.)  This is likely
to take 3-5 minutes.

To install the required packages for performing imputation using MICE,
run R and issue the following command (inside R):

         install.packages(c('optparse', 'mice', 'stringr', 'tidyverse', 'testthat'))

This is likely to take about 1 minute or less.

### Package versions

The Python package versions used are recorded in the
`requirements.txt` file.  The R package versions of the above-listed
packages used were:

* R: 4.2.0
* optparse: 1.7.3
* mice: 3.14.0
* stringr: 1.4.1
* tidyverse: 1.3.2
* testthat: 3.1.4

## The datasets

There are four datasets in the directory `DATA`.  Each of the datasets
has been split into three folds; one fold is used as a holdout set and
the other two form the development set, which is then further split
for five-fold cross validation.  The resulting files have names such
as:

* `holdout_0.csv`
* `devel_2_train_2.csv`
* `devel_1_val_1.csv`

and are generated from the provided original dataset using a Jupyter
notebook included in the directory.  The details of the original
dataset appear in the paper, and there are additional README files in
the `NHSX_COVID19` and `BREAST_CANCER` directories.  The original
`MIMIC-III` dataset can be generated using scripts in a separate
repository; see the Jupyter notebook for more details.

While the `NHSX_COVID19` and `BREAST_CANCER` datasets have natural
missingness, the `MIMIC-III`, `SYNTHETIC` and `SYNTHETIC_CATEGORICAL`
datasets have different levels of missingness artificially introduced.
The resulting filenames also describe the amount of missingness, for
example:

* `holdout_1_train_missing_0.25_test_missing_0.5.csv`
* `devel_2_val_0_train_missing_0.5_test_missing_0.5.csv`

There is no need to recreate these CSV files to perform the imputation
and classification experiments, but if desired, they can be recreated
by running the Jupyter notebooks in the subdirectories of `DATA`.

## Performing imputation

The script `imputation_main.py` performs all of the imputations
required and saves the results in `IMPUTED_DATA`.  Running:

        ./imputation_main.py --help

will display the available command-line options, including options to
choose which dataset to impute and which imputation method to use.

For a short demo of imputation, the following command will perform a
small number of imputations on one set of data using all of the
methods:

        ./imputation_main.py --dataset SYNTHETIC --train_percent 0.25 --test_percent 0.25 --holdouts 0 --val_sets 0 --loglevel INFO

This will take about 40 minutes with a CPU and perform 10 imputations
with each method; for a shorter demo, using the option `--repeats 2`
will reduce this to just two imputations with each method.  It will be
somewhat faster with a GPU.  The results will be imputed datasets in
the `IMPUTED_DATA` directory, mostly stored as NumPy `.npy` files, but
for MICE as `.csv` files.  MissForest is likely to give lots of
warnings; this is due to the `missingpy` package not having been
updated for the latest version of scikit-learn.

## Performing classification

The script `classification_main.py` performs all of the imputations
required and saves the results in `IMPUTED_DATA`.  Running:

        ./classification_main.py --help

will display the available command-line options, including options to
choose which dataset to impute, which imputation method to use and
which classification method to use.  There is also an option to upload
the results to the wandb.ai site.

It is necessary to specify a classification method.  Each method
offers different options to specifier classifier parameters.  The
following command will show those command-line options specific to the
Random Forest classifier:

        ./classification_main.py RandomForest --help

A similar command can be used for the other classifiers.

It is obviously necessary to do the imputation step on the data before
performing the classication.  For a short demo of classification,
using the imputations performed above, the following command will 
classify the imputations using the Random Forest classifier with a
reduced grid search size (for speed):

        ./classification_main.py --dataset SYNTHETIC --train_percent 0.25 --test_percent 0.25 --holdouts 0 --val_sets 0 --loglevel INFO RandomForest --n_estimators 20,95,20 --min_samples_split 2,4 --min_samples_leaf 2,4

This will take about 3 minutes.  The results are stored in
subdirectories of `CLASSIFICATION_RESULTS`.  A similarly reduced
XGBoost run takes about 7 minutes; NGBoost takes about 12 minutes
while NeuralNetwork takes about 3 minutes.

## Reproducing the paper's results

Running `imputation_main.py` without any options followed by
`classification_main.py CLASSIFIER` with the four different
classifiers will reproduce all of the results of our paper.
