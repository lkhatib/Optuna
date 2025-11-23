#!/usr/bin/env python
"""
Optuna + Keras age regression example on microbiome / clinical data.

This script demonstrates how to:
- Load feature and target data from disk
- Optionally process a BIOM table with CLR transform
- Standardize features
- Build a Keras MLP regression model
- Tune hyperparameters with Optuna

Example
-------
python optuna_mlp_age_regression.py \
    --features_csv data/features.csv \
    --targets_csv data/targets.csv \
    --target_column age \
    --n_trials 30 \
    --output_dir results/

If you're working with a BIOM table, you can preprocess it separately
into a feature matrix and pass that CSV via --features_csv.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

# Optional microbiome imports – only needed if you integrate BIOM directly
try:
    from skbio.stats.composition import clr, multiplicative_replacement
    from biom import load_table
except ImportError:
    clr = None
    multiplicative_replacement = None
    load_table = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_features_and_targets(
    features_csv: Path,
    targets_csv: Path,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix X and target vector y from CSV files.

    Parameters
    ----------
    features_csv : Path
        Path to CSV file with features (rows = samples, columns = features).
    targets_csv : Path
        Path to CSV file with targets.
    target_column : str
        Column name in targets_csv containing the target (e.g., 'age').

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector aligned to X.
    """
    logger.info(f"Loading features from {features_csv}")
    X = pd.read_csv(features_csv, index_col=0)

    logger.info(f"Loading targets from {targets_csv}")
    y_df = pd.read_csv(targets_csv, index_col=0)

    if target_column not in y_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {targets_csv}")

    # Align on index (sample IDs)
    common_idx = X.index.intersection(y_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping sample IDs between features and targets.")

    X = X.loc[common_idx].copy()
    y = y_df.loc[common_idx, target_column].astype(float)

    logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------
def build_model(
    trial: optuna.Trial,
    input_dim: int,
) -> Sequential:
    """Build a Keras MLP model using hyperparameters from Optuna trial."""
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    use_batchnorm = trial.suggest_categorical("batchnorm", [True, False])

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model = Sequential()
    model.add(Dense(hidden_dim, activation=activation, input_shape=(input_dim,)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers - 1):
        model.add(Dense(hidden_dim, activation=activation))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Regression output
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=optimizer,
        loss="mae",   # mean absolute error
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------
def create_objective(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    """Create an Optuna objective function closure."""

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 50, 150)

        model = build_model(trial, input_dim=X_train.shape[1])

        early_stopping = EarlyStopping(
            monitor="val_mae",
            patience=15,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[early_stopping],
        )

        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        trial.set_user_attr("best_val_mae", float(val_mae))
        return float(val_mae)

    return objective


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------
def run_optuna(
    features_csv: Path,
    targets_csv: Path,
    target_column: str,
    n_trials: int,
    test_size: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for age regression.

    Returns the best trial's metrics and hyperparameters.
    """
    X_df, y_series = load_features_and_targets(features_csv, targets_csv, target_column)

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df,
        y_series.values,
        test_size=test_size,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_val = scaler.transform(X_val_df.values)

    logger.info("Starting Optuna study...")
    objective = create_objective(X_train, X_val, y_train, y_val)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_val_mae = study.best_value
    best_params = study.best_params
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_val_mae": best_val_mae,
        "best_params": best_params,
        "n_trials": n_trials,
    }

    summary_path = output_dir / "optuna_summary.json"
    logger.info(f"Saving results to {summary_path}")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna + Keras age regression on clinical / microbiome data."
    )
    parser.add_argument(
        "--features_csv",
        type=Path,
        required=True,
        help="Path to features CSV file (rows=samples, cols=features).",
    )
    parser.add_argument(
        "--targets_csv",
        type=Path,
        required=True,
        help="Path to targets CSV file with target column (e.g., age).",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="age",
        help="Name of the target column in targets_csv.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used as validation set.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory to save Optuna results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_optuna(
        features_csv=args.features_csv,
        targets_csv=args.targets_csv,
        target_column=args.target_column,
        n_trials=args.n_trials,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
