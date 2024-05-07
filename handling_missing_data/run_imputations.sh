#!/usr/bin/env bash
#SBATCH --job-name=i_ADULT # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=i_ADULT_%j.out
#not SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
# source /usr/xtmp/jcd97/imputation-env/bin/activate
source /home/users/ham51/imputation-env/bin/activate


# srun -u imputation_main.py -i MICE -d 'PHARYNGITIS_MAR_25 PHARYNGITIS_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
# srun -u imputation_main.py -i MICE -d 'CKD_MAR_25 CKD_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
# srun -u imputation_main.py -i MICE -d 'HEART_DISEASE_MAR_25 HEART_DISEASE_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
# srun -u imputation_main.py -i MICE -d 'MIMIC_MAR_25 MIMIC_MAR_50' --holdouts '0 1 2 3 4 5 6 7 8 9'
srun -u imputation_main.py -i MICE -d 'ADULT' --holdouts '0 1 2 3 4 5 6 7 8 9'