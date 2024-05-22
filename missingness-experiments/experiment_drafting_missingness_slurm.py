#SBATCH --job-name=missing_data # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=missing_data_%j.out
#SBATCH --ntasks=1                 # Run on a single Node
#SBATCH --cpus-per-task=16          # All nodes have 16+ cores; about 20 have 40+
#SBATCH --mem=100gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

from statistics import correlation
from tqdm import tqdm
import sys
import numpy as np, scipy.stats as st
sys.path.append('./')
from experiment_utils import *
from MAR_model import MAR_model
import time
from datetime import date

def ci(m_list, conf=0.95):
    interval = st.t.interval(conf, len(m_list)-1, loc=np.mean(m_list), scale=st.sem(m_list))
    return (interval[1] - interval[0]) / 2

def acc_by_support(missing_props, mode='MAR', n_samples=1000, n_features=10, num_bootstraps=20,
                    correlations=None, x1_noise_center=0):
    original_accs = []
    missing_accs = []
    augmented_accs = []

    # For similicity, assuming all our features will have variance=1
    if correlations is None:
        X_cov = np.ones((n_features, n_features))
    else:
        X_cov = correlations
        
    # Generate our data ----------
    X_orig, y_orig = generate_data(num_features=n_features, num_samples=n_samples, 
        dgp_function=my_linear_dgp, noise_scale=0.01, X_cov_matrix=X_cov, x1_noise_center=x1_noise_center)
    
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
            #[np.random.choice([0, 1], size=(n_features), p=[0.8, 0.2]).astype(float) for i in range(n_features)], 
            missingness_model = MAR_model([np.array([1] + [0 for _ in range(n_features-1)])] + 
                                        [np.zeros(n_features) for _ in range(n_features - 1)],
                                        missing_prop, eligible_cols=[0])
            '''def missingness_model(X):
                mask = np.zeros_like(X)

                # For each feature in X, get the probability of removal
                unajusted_predictions = (X[:, 0] + 5) * X[:, 1]
                positive_threshold = np.quantile(unajusted_predictions, 1 - missing_prop)
                cur_mask = np.where(unajusted_predictions.flatten() > positive_threshold, 0, 1)
                mask[:, 0] = cur_mask
                return mask'''
        else:
            missingness_model = None
        
        if mode == 'MAR':
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        else:
            X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=None)
        df = pd.DataFrame(X_missing)
        df['y'] = y

        imputed_df, num_imputed_nonmissing = process_data_with_missingness(impute_mean_values(df))
        binned_df, num_nonmissing_ftrs = process_data_with_missingness(df)
        X_imputed = imputed_df.loc[:, imputed_df.columns != 'y'].values
        X_augmented = binned_df.loc[:, binned_df.columns != 'y'].values
        print("X_augmented.shape", X_augmented.shape)

        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        # Get best accuracy for imputed data -----------
        X_train = X_imputed[train_indices, :]
        X_val = X_imputed[val_indices, :]
        X_test = X_imputed[test_indices, :]

        fit_model, test_accs, support_sizes, lambdas = get_reg_accu_paths(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)
        plt.errorbar(lambdas, [np.mean(o) for o in test_accs], [ci(o) for o in test_accs], capsize=4)
        legend.append(f"Impu w/{missing_prop} Missing ({X_augmented.shape[-1]} Binary Features)")

        # Get best accuracy for the augmented data -----
        X_train = X_augmented[train_indices, :]
        X_val = X_augmented[val_indices, :]
        X_test = X_augmented[test_indices, :]

        '''fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)'''
                                                        
        fit_model, test_accs, support_sizes, lambdas = get_reg_accu_paths(X_train, X_val, X_test, 
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
        plt.errorbar(lambdas, [np.mean(o) for o in test_accs], [ci(o) for o in test_accs], capsize=4)
        legend.append(f"Aug w/{missing_prop} Missing ({X_augmented.shape[-1]} Binary Features)")
    plt.legend(legend)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Lambda")
    plt.xscale('log')
    if missingness_model is None:
        plt.title("Accuracy by Regularization Under MCAR")
        plt.savefig(f'{date.today()}MCAR_{x1_noise_center}_x1_bias_acc_by_reg_{n_samples}_samples_{n_features}_features_easy_mar.png')
    else:
        plt.title("Accuracy by Regularization Under MAR")
        plt.savefig(f'{date.today()}MAR_{x1_noise_center}_x1_bias_acc_by_reg_{n_samples}_samples_{n_features}_features_easy_mar.png')

def acc_by_missingness(missing_props, modes=['MAR', 'MCAR'], n_samples=1000, n_features=10, num_bootstraps=20,
                    correlations=None, x1_noise_center=0):
    
    # For similicity, assuming all our features will have variance=1
    if correlations is None:
        X_cov = np.ones((n_features, n_features))
    else:
        X_cov = correlations
    # Generate our data ----------
    X_orig, y_orig = generate_data(num_features=n_features, num_samples=n_samples, 
        dgp_function=my_linear_dgp, noise_scale=0.01, X_cov_matrix=X_cov, x1_noise_center=x1_noise_center)
    
    # Split it into train and test -----------------
    train_prop = 0.6
    val_prop = 0.2
    dataset_labels = np.random.choice([0, 1, 2], size=(X_orig.shape[0]), p=[train_prop, val_prop, 1-train_prop-val_prop])
    train_indices = dataset_labels == 0
    val_indices = dataset_labels == 1
    test_indices = dataset_labels == 2
    legend = []

    vals_by_missingness = {}
    for mode in modes:
        vals_by_missingness[mode] = {
            'missing_use': [],
            'accuracy': [],
            'mean_value_impu_accuracy': []
        }
    for mode in modes:
        prop_models_with_missing = []
        augmented_accs = []
        imputed_accs = []
   
        for missing_prop in tqdm(missing_props):
            X, y = X_orig.copy(), y_orig.copy()
            if mode == 'MAR':
                missingness_model = MAR_model([np.array([1] + [0 for _ in range(n_features-1)])] + 
                                        [np.zeros(n_features) for _ in range(n_features - 1)],
                                        missing_prop, eligible_cols=[0])
            else:
                missingness_model = None
            
            if mode == 'MAR':
                X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
            else:
                X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=None)
            df = pd.DataFrame(X_missing)
            df['y'] = y
            imputed_df, num_imputed_nonmissing = process_data_with_missingness(impute_mean_values(df))
            print(f"For impute: {num_imputed_nonmissing} of {imputed_df.shape[-1] -1} ftrs are nonmissing")
            binned_df, num_nonmissing_ftrs = process_data_with_missingness(df)

            X_imputed = imputed_df.loc[:, imputed_df.columns != 'y'].values
            X_augmented = binned_df.loc[:, binned_df.columns != 'y'].values

            y_train = y[train_indices]
            y_val = y[val_indices]
            y_test = y[test_indices]


            # Get best accuracy for imputed data -----------
            X_train = X_imputed[train_indices, :]
            X_val = X_imputed[val_indices, :]
            X_test = X_imputed[test_indices, :]

            fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                            y_train, y_val, y_test, 
                                                            num_bootstraps=num_bootstraps)
            imputed_accs.append(test_accs)

            # Get best accuracy for the augmented data -----
            X_train = X_augmented[train_indices, :]
            X_val = X_augmented[val_indices, :]
            X_test = X_augmented[test_indices, :]

            fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                            y_train, y_val, y_test, 
                                                            num_bootstraps=num_bootstraps)
            coef_matrix = fit_model.coeff().todense()
            #print(coef_matrix.shape)
            num_missing_by_model = np.sum((coef_matrix[:, -num_nonmissing_ftrs:] != 0), axis=1)
            num_nonmissing_by_model = np.sum((coef_matrix != 0), axis=1)
            missing_ratio = np.reshape(num_missing_by_model, (-1)) / np.reshape(num_nonmissing_by_model, (-1))
            missing_ratio[np.isnan(missing_ratio)] = 0
            #print(np.reshape(num_missing_by_model, (-1)), np.reshape(num_nonmissing_by_model, (-1)))
            #print(missing_ratio)
            prop_models_with_missing.append(list(missing_ratio))
            #plt.savefig(f'model_viz_{n_samples}_samples_{n_features}_features_{missing_prop}_missing.png')
            #plt.clf()
            augmented_accs.append(test_accs)

        vals_by_missingness[mode]['missing_use'] = prop_models_with_missing
        vals_by_missingness[mode]['accuracy'] = augmented_accs
        vals_by_missingness[mode]['mean_value_impu_accuracy'] = imputed_accs

    for mode in modes:
        plt.errorbar(missing_props, 
                    [np.mean(o) for o in vals_by_missingness[mode]['accuracy']], 
                    [ci(o) for o in vals_by_missingness[mode]['accuracy']], capsize=4)

    for mode in modes:
        plt.errorbar(missing_props, 
                    [np.mean(o) for o in vals_by_missingness[mode]['mean_value_impu_accuracy']], 
                    [ci(o) for o in vals_by_missingness[mode]['mean_value_impu_accuracy']], capsize=4)
                    
    plt.legend([f"Augmentation ({m})" for m in modes] + [f"Mean Val Imputation ({m})" for m in modes])
    plt.ylabel("Test Accuracy")
    plt.xlabel("Missing Proportion")
    plt.title("Test Accuracy by Missing Proportion")
    plt.savefig(f'{date.today()}_{x1_noise_center}_x1_bias_acc_by_missing_prop_{n_samples}_samples_{n_features}_features_easy_mar.png')
    plt.clf()

    for mode in modes:
        #print(vals_by_missingness[mode]['missing_use'])
        plt.plot(missing_props, [np.mean(o) for o in vals_by_missingness[mode]['missing_use']])
    plt.legend(modes)
    plt.ylabel("Average Proportion of Features Used That Depend on Missingness")
    plt.xlabel("Missing Data Proportion")
    
    plt.title("Missing Feature Usage by Missing Proportion")
    plt.savefig(f'missing_use_by_missing_prop_{n_samples}_samples_{n_features}_features_easy_mar.png')
    plt.clf()

def time_by_samples(n_samples, missing_prop=0.3, n_features=10, num_bootstraps=20,
                    correlations=None):
    original_times = [[] for i in range(len(n_samples))]
    missing_times = [[] for i in range(len(n_samples))]
    augmented_times = [[] for i in range(len(n_samples))]
    augmented_smart_times = [[] for i in range(len(n_samples))]


    for sample_ind, sample_count in enumerate(tqdm(n_samples)):

        # For similicity, assuming all our features will have variance=1
        if correlations is None:
            X_cov = np.ones((n_features, n_features))
        else:
            X_cov = correlations
        X_orig, y_orig = generate_data(num_features=n_features, num_samples=sample_count, 
            dgp_function=my_linear_dgp, noise_scale=0.01, X_cov_matrix=X_cov, x1_noise_center=x1_noise_center)
        missingness_model = MAR_model([np.array([1] + [0 for _ in range(n_features-1)])] + 
                                        [np.zeros(n_features) for _ in range(n_features - 1)],
                                        missing_prop, eligible_cols=[0])
        
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
        plt.savefig(f'{date.today()}_MCAR_runtime_by_samples_{n_features}_features_{missing_prop}_missing.png')
    else:
        plt.title("Runtime by Number of Samples MAR")
        plt.savefig(f'{date.today()}_MAR_runtime_by_samples_{n_features}_features_{missing_prop}_missing.png')

if __name__ == '__main__':
    np.random.seed(0)
    x1_noise_center = 3
    for x1_noise_center in [0, 1, 3, 6, 10]:
        n_samples = 100
        n_features = 3
        num_bootstraps = 50

        for n_samples in [100, 200, 500, 1000]:
            for model in ['MAR', 'MCAR']:
                missing_props = [i / 4 for i in range(3)]
                acc_by_support(missing_props, model, n_samples, n_features, num_bootstraps,
                    x1_noise_center=x1_noise_center)
                plt.clf()

            missing_props = [i / 20 for i in range(20)]
            correlations = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
            acc_by_missingness(missing_props, ['MAR', 'MCAR'], n_samples, n_features, num_bootstraps,
                                correlations=correlations, x1_noise_center=x1_noise_center)

        for n_samples in [100, 200, 500, 1000]:
            acc_by_missingness(missing_props, ['MAR', 'MCAR'], n_samples, n_features, num_bootstraps,
                                correlations=correlations, x1_noise_center=x1_noise_center)
            plt.clf()
        for n_samples in [50, 100]:
            for n_features in [3]:
                num_bootstraps = 25
                for model in ['MAR', 'MCAR']:
                    missing_props = [i / 4 for i in range(3)]
                    acc_by_support(missing_props, model, n_samples, n_features, num_bootstraps,
                        x1_noise_center=x1_noise_center)
                    plt.clf()

    
    for n_samples in [100, 200, 500, 1_000]:
        for n_features in [3]:
            num_bootstraps = 100

            missing_props = [i / 20 for i in range(20)]
            acc_by_missingness(missing_props, ['MAR', 'MCAR'], n_samples, n_features, num_bootstraps,
                x1_noise_center=x1_noise_center)
            plt.clf()

    for n_samples in [1_000, 100, 500]:
        for n_features in [3, 5, 10]:
            num_bootstraps = 100
            for model in ['MAR', 'MCAR']:
                missing_props = [i / 5 for i in range(5)]
                acc_by_support(missing_props, model, n_samples, n_features, num_bootstraps,
                    x1_noise_center=x1_noise_center)
                plt.clf()

    '''for missing_props in [0.1, 0.2, 0.5]:
        for n_features in [5, 10, 30]:
            n_samples = [100, 1_000, 10_000]
            num_bootstraps = 20

            time_by_samples(n_samples, missing_prop=missing_props, n_features=n_features, num_bootstraps=20)
            plt.clf()'''