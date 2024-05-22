#!/usr/bin/env bash
#SBATCH --job-name=i_gpu_MAR_general # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=logs/i_gpu_MAR_general_%j.out
#not SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH -x linux46
#not SBATCH --gres=gpu:p100:1
#SBATCH --partition=compsci
#SBATCH --array=0-279%10              # should equal num_imputers * num_datasets * num_holdouts - 1
# source /usr/xtmp/jcd97/imputation-env/bin/activate
source /home/users/ham51/imputation-env/bin/activate


imputers=(MissForest Mean)
datasets=(MIMIC_MAR_25 MIMIC_MAR_50 ADULT_MAR_25 ADULT_MAR_50 HEART_DISEASE_MAR_25 HEART_DISEASE_MAR_50 CKD_MAR_25 CKD_MAR_50 PHARYNGITIS_MAR_25 PHARYNGITIS_MAR_50 BREAST_CANCER_MAR_25 BREAST_CANCER_MAR_50 FICO_MAR_25 FICO_MAR_50)

num_imputers=${#imputers[@]}
imputer_idx=$((SLURM_ARRAY_TASK_ID % num_imputers))
imputer=${imputers[$imputer_idx]}

task_id=$((SLURM_ARRAY_TASK_ID / num_imputers))
num_datasets=${#datasets[@]}
dataset_idx=$((task_id % num_datasets))
dataset=${datasets[$dataset_idx]}

task_id=$((task_id / num_datasets))
holdout=$((task_id % num_datasets))

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Imputer: $imputer"
echo "Dataset: $dataset"
echo "Holdout: $holdout"



srun -u imputation_main.py -i "$imputer" -d "$dataset" --holdouts "$holdout"
echo "Done"