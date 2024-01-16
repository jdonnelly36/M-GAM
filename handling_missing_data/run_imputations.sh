#!/usr/bin/env bash
#SBATCH --job-name=impute # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=imutations_%j.out
#not SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb                     # Job memory request
#SBATCH  -x linux46 
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
source /usr/xtmp/jcd97/imputation-env/bin/activate

srun -u imputation_main.py -i MICE
