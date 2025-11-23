#!/bin/bash
#SBATCH --job-name=optuna-age
#SBATCH --output=logs/optuna_age_%j.out
#SBATCH --error=logs/optuna_age_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# Optional: email notifications
## #SBATCH --mail-type=BEGIN,END,FAIL
## #SBATCH --mail-user=your.email@domain.com

set -euo pipefail

# Activate environment
# Adjust this to your own env / module setup
source ~/miniconda3/bin/activate optuna-env

# Make sure logs directory exists
mkdir -p logs

# Define input paths (adapt to your project structure)
FEATURES_CSV=data/features.csv
TARGETS_CSV=data/targets.csv
OUTPUT_DIR=results/optuna_age

python optuna_mlp_age_regression.py \
    --features_csv "${FEATURES_CSV}" \
    --targets_csv "${TARGETS_CSV}" \
    --target_column age \
    --n_trials 30 \
    --test_size 0.2 \
    --output_dir "${OUTPUT_DIR}"
