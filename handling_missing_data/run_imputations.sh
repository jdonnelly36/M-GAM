#!/usr/bin/env bash
#SBATCH --job-name=ibreca # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=i_breca_10_%j.out
#not SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb                     # Job memory request
#SBATCH  -x linux46 
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
# source /usr/xtmp/jcd97/imputation-env/bin/activate
source /home/users/ham51/imputation-env/bin/activate

# srun -u imputation_main.py -i MICE
# srun -u imputation_main.py -i MICE -d FICO_MAR_25
# srun -u imputation_main.py -i MICE -d FICO_MAR_50
# srun -u imputation_main.py -i MICE -d 'FICO FICO_MAR_25 FICO_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
srun -u imputation_main.py -i MICE -d 'BREAST_CANCER BREAST_CANCER_MAR_25 BREAST_CANCER_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
# srun -u imputation_main.py -i MICE -d FICO_MAR
# srun -u imputation_main.py -i MICE -d 'SYNTHETIC_MAR_50 SYNTHETIC_MAR SYNTHETIC_MAR_25'
# srun -u imputation_main.py -i MICE -d 'SYNTHETIC_CATEGORICAL_MAR SYNTHETIC_CATEGORICAL_MAR_25 SYNTHETIC_CATEGORICAL_MAR_50'
# srun -u imputation_main.py -i MICE -d 'SYNTHETIC_CATEGORICAL_MAR_25 SYNTHETIC_CATEGORICAL_MAR_50'
# srun -u imputation_main.py -i MICE -d BREAST_CANCER_MAR_pt4
# srun -u imputation_main.py -i MICE -d BREAST_CANCER_MAR
# srun -u imputation_main.py -i MICE -d BREAST_CANCER_MAR_25
# srun -u imputation_main.py -i MICE -d BREAST_CANCER_MAR_50
# srun -u imputation_main.py -i MICE -d MIMIC_
# srun -u imputation_main.py -i MICE -d 'SYNTHETIC SYNTHETIC_CATEGORICAL'