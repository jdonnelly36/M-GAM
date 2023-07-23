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
sys.path.append('/home/users/jcd97/code/missing_data/fastsparsemissing/missingness-experiments')
from experiment_utils import *
from MAR_model import MAR_model
import time

def acc_by_missingness(missing_props, n_samples=1000, n_features=10, num_bootstraps=20):
    original_accs = []
    missing_accs = []
    augmented_accs = []

    for missing_prop in tqdm(missing_props):
        missingness_model = MAR_model([np.random.randint(2, size=(n_features)).astype(float) for i in range(n_features)], 
                                        missing_prop)
        
        # Generate our data ----------
        X, y = generate_data(num_features=n_features, num_samples=n_samples, 
            dgp_function=my_linear_dgp, noise_scale=0.5)
        X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        X_augmented = add_missingness_terms(X_missing.copy())

        X[X == -1] = 0
        X_missing[X_missing == -1] = 0
        X_augmented[X_augmented == -1] = 0

        # Split it into train and test -----------------
        train_prop = 0.6
        val_prop = 0.2
        dataset_labels = np.random.choice([0, 1, 2], size=(X.shape[0]), p=[train_prop, val_prop, 1-train_prop-val_prop])
        train_indices = dataset_labels == 0
        val_indices = dataset_labels == 1
        test_indices = dataset_labels == 2

        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        # Get best accuracy for the original data -----
        X_train = X[train_indices, :]
        X_val = X[val_indices, :]
        X_test = X[test_indices, :]

        fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test,
                                                        num_bootstraps=num_bootstraps)
        original_accs.append(test_accs)

        # Get best accuracy for the obfuscated data -----
        X_train = X_missing[train_indices, :]
        X_val = X_missing[val_indices, :]
        X_test = X_missing[test_indices, :]

        fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)
        missing_accs.append(test_accs)

        # Get best accuracy for the augmented data -----
        X_train = X_augmented[train_indices, :]
        X_val = X_augmented[val_indices, :]
        X_test = X_augmented[test_indices, :]

        fit_model, best_ind, test_accs = choose_best_gam(X_train, X_val, X_test, 
                                                        y_train, y_val, y_test, 
                                                        num_bootstraps=num_bootstraps)
        augmented_accs.append(test_accs)

    plt.errorbar(missing_props, [np.mean(o) for o in original_accs], [np.std(o) for o in original_accs], capsize=4)
    plt.errorbar(missing_props, [np.mean(o) for o in missing_accs], [np.std(o) for o in missing_accs], capsize=4)
    plt.errorbar(missing_props, [np.mean(o) for o in augmented_accs], [np.std(o) for o in augmented_accs], capsize=4)
    plt.legend(["No Missing Data", "Obfuscated", "Augmented"])
    plt.ylabel("Test Accuracy")
    plt.xlabel("Missing Probability")
    if missingness_model is None:
        plt.title("Accuracy by Proportion Missing Under MCAR")
        plt.savefig(f'MCAR_acc_by_missingness_{n_samples}_samples_{n_features}_features.png')
    else:
        plt.title("Accuracy by Proportion Missing Under MAR")
        plt.savefig(f'MAR_acc_by_missingness_{n_samples}_samples_{n_features}_features.png')

def time_by_samples(n_samples, missing_prop=0.3, n_features=10, num_bootstraps=20):
    original_times = [[] for i in range(len(n_samples))]
    missing_times = [[] for i in range(len(n_samples))]
    augmented_times = [[] for i in range(len(n_samples))]

    for sample_count in tqdm(n_samples):
        missingness_model = MAR_model([np.random.randint(2, size=(n_features)).astype(float) for i in range(n_features)], 
                                        missing_prop)
        
        # Generate our data ----------
        X, y = generate_data(num_features=n_features, num_samples=sample_count, 
            dgp_function=my_linear_dgp, noise_scale=0.5)
        X_missing = obfuscate_data(X.copy(), obfuscation_rate=missing_prop, missingness_model=missingness_model.get_mask)
        X_augmented = add_missingness_terms(X_missing.copy())

        X[X == -1] = 0
        X_missing[X_missing == -1] = 0
        X_augmented[X_augmented == -1] = 0

        for i in range(num_bootstraps):
            bootstrap_X, bootstrap_y = resample(X, y, replace=True)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            original_times[i].append(time.time() - start)

            bootstrap_X, bootstrap_y = resample(X_missing, y, replace=True)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            missing_times[i].append(time.time() - start)

            bootstrap_X, bootstrap_y = resample(X_augmented, y, replace=True)
            start = time.time()
            _ = fastsparsegams.fit(
                bootstrap_X, bootstrap_y, loss="Exponential", max_support_size=20, algorithm="CDPSI"
            )
            augmented_times[i].append(time.time() - start)

    plt.errorbar(missing_props, [np.mean(o) for o in original_times], [np.std(o) for o in original_times], capsize=4)
    plt.errorbar(missing_props, [np.mean(o) for o in missing_times], [np.std(o) for o in missing_times], capsize=4)
    plt.errorbar(missing_props, [np.mean(o) for o in augmented_times], [np.std(o) for o in augmented_times], capsize=4)
    plt.legend(["No Missing Data", "Obfuscated", "Augmented"])
    plt.ylabel("Runtime")
    plt.xlabel("Number of samples")
    if missingness_model is None:
        plt.title("Runtime by Number of Samples MCAR")
        plt.savefig(f'MCAR_runtime_by_samples_{n_samples}_samples_{n_features}_features.png')
    else:
        plt.title("Runtime by Number of Samples MAR")
        plt.savefig(f'MAR_runtime_by_samples_{n_samples}_samples_{n_features}_features.png')

if __name__ == '__main__':
    missing_props = [i / 20 for i in range(20)]
    n_samples = 1000
    n_features = 10
    num_bootstraps = 20

    acc_by_missingness(missing_props, n_samples, n_features, num_bootstraps)

    missing_props = 0.2
    n_samples = [10, 100, 1_000, 10_000, 100_000]
    n_features = 10
    num_bootstraps = 20

    time_by_samples(n_samples, missing_prop=missing_props, n_features=n_features, num_bootstraps=20)