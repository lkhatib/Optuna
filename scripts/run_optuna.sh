#!/bin/bash
#SBATCH --chdir=/projects/thdmi/5country/pangenome_filtered/age_prediction
#SBATCH --output=/projects/thdmi/5country/pangenome_filtered/age_prediction/optuna.out
#SBATCH --time 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="lakhatib@ucsd.edu"
#SBATCH --mem 512G
#SBATCH --partition=highmem 
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1 

source ~/miniconda3/bin/activate optuna-env

python optuna_script-dd.py