# Interpretable Generalized Additive Models for Datasets with Missing Values

This package contains the code used for `Interpretable Generalized Additive Models for Datasets with Missing Values` (https://arxiv.org/abs/2412.02646).

### Quickstart Guide

#### Installation: 

```bash
git clone https://github.com/jdonnelly36/M-GAM.git
cd M-GAM
pip install -r requirements.txt
```

#### Tutorial: 

See `quickstart.ipynb` for a tutorial walkthrough of how to use M-GAM. 

### Replication Information

The code contains two primary experimental directories: `handling_missing_data` and `missingness-experiments`. The former provides the code used to preprocess and split all datasets considered (in the `DATA` directory), as well as code used to time the imputation of missing values using various methods. The code in this folder is heavily borrowed from `The impact of imputation quality on machine learning classifiers for datasets with missing values` (https://gitlab.developers.cam.ac.uk/maths/cia/covid-19-projects/handling_missing_data). In order to run and time imputations, use the script `run_imputations.sh`

The second experimental directory, `missingness-experiments`, contains code to fit and measure various classifiers over this data, including M-GAMs. `nature-mice-imputations` provides scripts to compute and plot data relating to sparsity-accuracy curves; `mice_slurm_data.py` and `baseline_slurm_data.py` should be run to generate all required data, which can then be visualized by the various scripts in this directory titled `fig...`. In the `nature-mice-timing-baselines` directory we provide code to fit and time a wide variety of non-GAM baselines by running `mice_slurm_timing_data.py`. This data can then be aggregated into figures using `generate_figures.py`.

The backbone algorithm for M-GAM comes from `Fast Sparse Classification for Generalized Linear and Additive Models` (https://github.com/interpretml/fastSparse). 
