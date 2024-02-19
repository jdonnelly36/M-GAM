The data here comes from the documented experiments at https://gitlab.developers.cam.ac.uk/maths/cia/covid-19-projects/handling_missing_data/-/blob/main/DATA/BREAST_CANCER/Breast_cancer_complete_used_onehot.csv?ref_type=heads , which were used for the paper "The Impact of Imputation Quality on Machine Learning Classifiers for Datasets with Missing Values". 

Specifically, the val_0 folder was obtained by running the command python imputation_main.py --dataset BREAST_CANCER -i MICE --holdouts 0 --val_sets 0 --loglevel INFO in the cloned repository. The full test_per_0 foler was obtained with: 

`python imputation_main.py --dataset BREAST_CANCER -i MICE --loglevel INFO`

The original dataframe, with the same train/test splits, is devel_0_train_0.csv, devel_0_val_0.csv, and holdout_0.csv . We need these dataframes to recover the labels for each example (and they give us the column names too). 

(TODO: review the documentation around the train/val/holdout splits). 

An alternative is to run MICE ourselves from scratch, but it seems like better replication practice to use exactly this paper's methodology, at least for its datasets. This methodology corresponds to generating multiple imputations for the dataset which do not use y. 