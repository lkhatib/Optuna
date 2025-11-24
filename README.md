# Age Prediction from Clinical & Microbiome Data  
### Hyperparameter Tuning with Optuna + Keras (MLP Regression)

This repository provides a reproducible example of using **Optuna** to tune a **Keras MLP regression model** for predicting **age** from clinical, microbiome, or other biological datasets.  

---

## Repository Structure

```
Optuna/
├── env
│ └── optuna-env.yml
├── scripts/
│ ├── run_optuna.sh
│ └── optuna_script-dd.py
└── README.md
```
---

## Instructions

### 1. Install Dependencies

conda create -n optuna-env python=3.10  
conda activate optuna-env  

---

### 2. Prepare Your Data

**features.csv**  
- Rows = samples  
- Columns = features (microbial abundances, clinical variables, etc.)  
- Index = sample IDs  

**targets.csv**  
Must contain a numeric target column, e.g.:

sample_id | age  
--------- | ----  
S01       | 55  
S02       | 42  
S03       | 63  

---

### 3. Run the Optuna Training Script

python optuna_mlp_age_regression.py \  
  --features_csv data/features.csv \  
  --targets_csv data/targets.csv \  
  --target_column age \  
  --n_trials 30 \  
  --test_size 0.2 \  
  --output_dir results/  

Results are saved in:  
**results/optuna_summary.json**

---

## Model Architecture

The script builds a tunable **MLP regression model** with:

- Variable number of layers  
- Adjustable hidden dimensions  
- Dropout  
- Optional batch normalization  
- Adam or RMSprop optimizer  
- Tunable learning rate  

All hyperparameters are optimized with **Optuna**.

---

## Hyperparameter Optimization

Optuna optimizes validation **MAE** using TPE sampling.

Tuned parameters include:

- n_layers  
- hidden_dim  
- activation  
- dropout_rate  
- batchnorm  
- optimizer  
- learning_rate  
- batch_size  
- epochs  

---

## Working with Microbiome Data

If microbiome features are used, preprocessing may include:

- CLR transform  
- Multiplicative replacement  
- BIOM → feature matrix conversion  

Optional imports in the script allow:

from skbio.stats.composition import clr, multiplicative_replacement  
from biom import load_table  

---

## Running on HPC (SLURM)

Submit the example SLURM job:

sbatch run_optuna.slurm  

Modify environment setup, resource requests, and paths as needed.

---

## Example Optuna Summary (JSON)

{
  "best_val_mae": 5.82,
  "best_params": {
    "n_layers": 3,
    "hidden_dim": 128,
    "activation": "relu",
    "dropout_rate": 0.2,
    "batchnorm": true,
    "optimizer": "adam",
    "learning_rate": 0.0007,
    "batch_size": 64,
    "epochs": 120
  },
  "n_trials": 30
}

---
