# Optuna - Hyperparameter Optimization Framework using SLURM

Framework for running Optuna on a SLURM server. For more information on running Optuna, see [Optuna GitHub Repository](https://github.com/optuna/optuna)

**1) Install an Optuna environment using the .yml file that I've provided in the repo:**

`conda env create -f optuna-env.yml`

**2) Clone this repo.**

`git clone https://github.com/lkhatib/Optuna.git`

**3) Modify scripts.** Templates are provided in Optuna/scripts

**4)Run scripts.** 

`sbatch run_optuna.sh`

**5)Check output for best MAE and best hyperparameters.** Example output provided in output.out
