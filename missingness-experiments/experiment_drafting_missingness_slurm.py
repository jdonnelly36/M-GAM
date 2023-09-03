#!/home/users/jcd97/missing_data/bin/python
#SBATCH --job-name=missing_data # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=missing_data_%j.out
#SBATCH --ntasks=1                 # Run on a single Node
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=60gb                     # Job memory request
#not SBATCH  -x linux[41-60],gpu-compute[1-7]
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

from tqdm import tqdm
import sys
import numpy as np, scipy.stats as st
sys.path.append('./')
from experiment_utils import *
from MAR_model import MAR_model
import time

def ci(m_list, conf=0.95):
    interval = st.t.interval(conf, len(m_list)-1, loc=np.mean(m_list), scale=st.sem(m_list))
    return (interval[1] - interval[0]) / 2

def acc_by_support(missing_props, mode='MAR', n_samples=1000, n_features=10, num_bootstraps=20):
    original_accs = []
    missing_accs = []
    augmented_accs = []
    # Generate our data ----------
    X_orig, y_orig = generate_data(num_features=n_features, num_samples=n_samples, 
        dgp_function=my_linear_dgp, noise_scale=0.5)
    
    # Split it into train and test -----------------
    train_prop = 0.6
    val_prop = 0.2
    dataset_labels = np.random.choice([0, 1, 2], size=(X_orig.shape[0]), p=[train_prop, val_prop, 1-train_prop-val_prop])
    train_indices = dataset_labels == 0
    val_indices = dataset_labels == 1
    test_indices = dataset_labels == 2
    legend = []

    for missing_prop in tqdm(missing_props):
        X, y = X_orig.copy(), y_orig.copy()
        if mode == 'MAR':
            missingness_model = MAR_model([np.random.randint(2, size=(n_features)).astype(float) for i in range(n_features)], 
                                        missing_prop)
        else:
            missingness_model = None
        
        if mode == 'MAR':
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        else:
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=None)
        df = pd.DataFrame(X_missing)
        df['y'] = y
        binned_df = process_data_with_missingness(df)
        X_augmented = binned_df.loc[:, binned_df.columns != 'y'].values
        print("X_augmented.shape", X_augmented.shape)

        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        # Get best accuracy for the augmented data -----
        X_train = X_augmented[train_indices, :]
        X_val = X_augmented[val_indices, :]
        X_test = X_augmented[test_indices, :]

        '''fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)'''
                                                        
        fit_model, test_accs, support_sizes = get_reg_accu_paths(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)
        #coef_matrix = fit_model.coeff().todense()
        '''print(f"For {n_samples} samples with {n_features} features and {missing_prop} missingness -----")
        num_w_missing = np.sum(np.sum((coef_matrix[:, n_features:] != 0), axis=1) > 0)
        print(f"{num_w_missing} out of {coef_matrix.shape[0]} models had non-zero coefficients on missingness terms")'''
        #plt.savefig(f'model_viz_{n_samples}_samples_{n_features}_features_{missing_prop}_missing.png')
        #plt.clf()
        #augmented_accs.append(test_accs)

        #plt.errorbar(missing_props, [np.mean(o) for o in original_accs], [ci(o) for o in original_accs], capsize=4)
        #plt.errorbar(missing_props, [np.mean(o) for o in missing_accs], [ci(o) for o in missing_accs], capsize=4)
        plt.errorbar(support_sizes, [np.mean(o) for o in test_accs], [ci(o) for o in test_accs], capsize=4)
        legend.append(f"{missing_prop} Missing ({X_augmented.shape[-1]} Binary Features)")
    plt.legend(legend)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Support Size")
    if missingness_model is None:
        plt.title("Accuracy by Support Size Under MCAR")
        plt.savefig(f'MCAR_acc_by_support_{n_samples}_samples_{n_features}_features.png')
    else:
        plt.title("Accuracy by Support Size Under MAR")
        plt.savefig(f'MAR_acc_by_support_{n_samples}_samples_{n_features}_features.png')

def acc_by_missingness(missing_props, mode='MAR', n_samples=1000, n_features=10, num_bootstraps=20):
    augmented_accs = []
    # Generate our data ----------
    X_orig, y_orig = generate_data(num_features=n_features, num_samples=n_samples, 
        dgp_function=my_linear_dgp, noise_scale=0.5)
    
    # Split it into train and test -----------------
    train_prop = 0.6
    val_prop = 0.2
    dataset_labels = np.random.choice([0, 1, 2], size=(X_orig.shape[0]), p=[train_prop, val_prop, 1-train_prop-val_prop])
    train_indices = dataset_labels == 0
    val_indices = dataset_labels == 1
    test_indices = dataset_labels == 2
    legend = []
    prop_models_with_missing = []

    for missing_prop in tqdm(missing_props):
        X, y = X_orig.copy(), y_orig.copy()
        if mode == 'MAR':
            missingness_model = MAR_model([np.random.randint(2, size=(n_features)).astype(float) for i in range(n_features)], 
                                        missing_prop)
        else:
            missingness_model = None
        
        if mode == 'MAR':
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        else:
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=None)
        df = pd.DataFrame(X_missing)
        df['y'] = y
        binned_df, num_nonmissing_ftrs = process_data_with_missingness(df)
        X_augmented = binned_df.loc[:, binned_df.columns != 'y'].values
        print("X_augmented.shape", X_augmented.shape)

        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        # Get best accuracy for the augmented data -----
        X_train = X_augmented[train_indices, :]
        X_val = X_augmented[val_indices, :]
        X_test = X_augmented[test_indices, :]

        fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)
        coef_matrix = fit_model.coeff().todense()
        print(f"For {n_samples} samples with {n_features} features and {num_nonmissing_ftrs} missingness ftrs -----")
        num_w_missing = np.sum(np.sum((coef_matrix[:, -num_nonmissing_ftrs:] != 0), axis=1) > 0)
        print(f"{num_w_missing} out of {coef_matrix.shape[0]} models had non-zero coefficients on missingness terms")
        prop_models_with_missing.append(coef_matrix.shape[0] / num_w_missing)
        #plt.savefig(f'model_viz_{n_samples}_samples_{n_features}_features_{missing_prop}_missing.png')
        #plt.clf()
        augmented_accs.append(test_accs)

    plt.errorbar(missing_props, [np.mean(o) for o in augmented_accs], [ci(o) for o in test_accs], capsize=4)
    #legend = [f"{m} Missing ({X_augmented.shape[-1]} Binary Features)" for m in missing_props]
    #plt.legend(legend)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Missing Proportion")
    if missingness_model is None:
        plt.title("Accuracy by Missing Proportion")
        plt.savefig(f'MCAR_acc_by_missing_prop_{n_samples}_samples_{n_features}_features.png')
    else:
        plt.title("Accuracy by Missing Proportion")
        plt.savefig(f'MAR_acc_by_missing_prop_{n_samples}_samples_{n_features}_features.png')
    plt.clf()

    
    plt.plot(missing_props, prop_models_with_missing)
    #legend = [f"{m} Missing ({X_augmented.shape[-1]} Binary Features)" for m in missing_props]
    #plt.legend(legend)
    plt.ylabel("Proportion of Models Using Missingness")
    plt.xlabel("Missing Data Proportion")
    if missingness_model is None:
        plt.title("Accuracy by Missing Proportion")
        plt.savefig(f'MCAR_missing_use_by_missing_prop_{n_samples}_samples_{n_features}_features.png')
    else:
        plt.title("Accuracy by Missing Proportion")
        plt.savefig(f'MAR_missing_use_by_missing_prop_{n_samples}_samples_{n_features}_features.png')
    plt.clf()

def time_by_samples(n_samples, missing_prop=0.3, n_features=10, num_bootstraps=20):
    original_times = [[] for i in range(len(n_samples))]
    missing_times = [[] for i in range(len(n_samples))]
    augmented_times = [[] for i in range(len(n_samples))]
    augmented_smart_times = [[] for i in range(len(n_samples))]


    for sample_ind, sample_count in enumerate(tqdm(n_samples)):
        X_orig, y_orig = generate_data(num_features=n_features, num_samples=sample_count, 
            dgp_function=my_linear_dgp, noise_scale=0.5)
        missingness_model = MAR_model([np.random.randint(2, size=(n_features)).astype(float) for i in range(n_features)], 
                                        missing_prop)
        
        # Generate our data ----------
        X, y = X_orig.copy(), y_orig.copy()
        X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        X_augmented = add_missingness_terms(X_missing.copy())
        X_augmented_smart = add_missingness_terms(X_missing.copy(), use_inclusion_bound=True)

        X[X == -1] = 0
        X_missing[X_missing == -1] = 0
        X_augmented[X_augmented == -1] = 0
        X_augmented_smart[X_augmented_smart == -1] = 0

        for i in range(num_bootstraps):
            bootstrap_X, bootstrap_y = resample(X, y, replace=True, random_state=i)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            original_times[sample_ind].append(time.time() - start)

            bootstrap_X, bootstrap_y = resample(X_missing, y, replace=True, random_state=i)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            missing_times[sample_ind].append(time.time() - start)

            bootstrap_X, bootstrap_y = resample(X_augmented, y, replace=True, random_state=i)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            augmented_times[sample_ind].append(time.time() - start)

            bootstrap_X, bootstrap_y = resample(X_augmented_smart, y, replace=True, random_state=i)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            augmented_smart_times[sample_ind].append(time.time() - start)

    plt.errorbar(n_samples, [np.mean(o) for o in original_times], [ci(o) for o in original_times], capsize=4)
    plt.errorbar(n_samples, [np.mean(o) for o in missing_times], [ci(o) for o in missing_times], capsize=4)
    plt.errorbar(n_samples, [np.mean(o) for o in augmented_times], [ci(o) for o in augmented_times], capsize=4)
    plt.errorbar(n_samples, [np.mean(o) for o in augmented_smart_times], [ci(o) for o in augmented_smart_times], capsize=4)
    plt.legend(["No Missing Data", "Obfuscated", "Augmented", "Augmented w/ Bound"])
    plt.ylabel("Runtime (Seconds)")
    plt.xlabel("Number of samples")
    plt.xscale('log')
    if missingness_model is None:
        plt.title("Runtime by Number of Samples MCAR")
        plt.savefig(f'MCAR_runtime_by_samples_{n_features}_features_{missing_prop}_missing.png')
    else:
        plt.title("Runtime by Number of Samples MAR")
        plt.savefig(f'MAR_runtime_by_samples_{n_features}_features_{missing_prop}_missing.png')

if __name__ == '__main__':
    for model in ['MAR', 'MCAR']:
        for n_features in [5, 10, 30]:
            for n_samples in [100, 1_000, 10_000]:
                num_bootstraps = 100

                missing_props = [i / 5 for i in range(21)]
                acc_by_missingness(missing_props, model, n_samples, n_features, num_bootstraps)
                plt.clf()

                missing_props = [i / 5 for i in range(6)]
                acc_by_support(missing_props, model, n_samples, n_features, num_bootstraps)
                plt.clf()

    '''for missing_props in [0.1, 0.2, 0.5]:
        for n_features in [5, 10, 30]:
            n_samples = [100, 1_000, 10_000]
            num_bootstraps = 20

            time_by_samples(n_samples, missing_prop=missing_props, n_features=n_features, num_bootstraps=20)
            plt.clf()'''