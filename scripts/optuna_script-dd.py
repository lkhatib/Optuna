import optuna
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from skbio.stats.composition import clr, multiplicative_replacement
from biom import Table, load_table
import dask.dataframe as dd  # Use dask for memory-efficient data processing

print("Running script..")

# Load the data with Dask to handle large datasets
ft = dd.read_csv('/projects/thdmi/5country/pangenome_filtered/data/thdmi_zebra_feature-table_valid_covariates.tsv', sep='\t', dtype={'Unnamed: 0': 'object'}).set_index('Unnamed: 0') 
print(f"Loaded feature table: # of columns={len(ft.columns)}")

md = dd.read_csv(
    '/projects/thdmi/5country/pangenome_filtered/data/thdmi_metadata_valid_covariates.tsv',
    sep='\t', 
    dtype=str
).set_index('Unnamed: 0')

# Convert data to float32 to save memory
ft = ft.astype(np.float32)

print(f"Converted ft to np.float32")

def multiplicative_clr(ft):
    ft_array = np.array(ft)
    # Replace zeros using multiplicative replacement
    metagenomic_counts = multiplicative_replacement(ft_array)
    # Apply CLR transformation
    clr_transformed_data = clr(metagenomic_counts)
    ft = pd.DataFrame(clr_transformed_data, index=ft.index, columns=ft.columns)
    return ft

# Apply the clr transformation
ft = multiplicative_clr(ft)

print(f"Multiplicative clr completed.")

# Convert 'age' column from metadata
ft['age'] = md['host_age'].compute() 

print(f"Preparing data...")
# Separating features and target
X = ft.drop('age', axis=1)  # Drop the 'age' column, assuming the rest are microbiome features
y = ft['age']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the features using StandardScaler (convert to float32 for efficiency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Define an objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    epochs = trial.suggest_int('epochs', 50, 100)  # Reduced max epochs to control memory usage
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.4)  # Reduced dropout for memory efficiency

    # Create the model with the suggested hyperparameters
    model = Sequential()
    model.add(Dense(256, activation=activation, input_shape=(X_train_scaled.shape[1],)))  # Reduced layer size
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    # Implement early stopping to avoid overfitting and save memory
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model using batch data (to reduce memory usage)
    model.fit(X_train_scaled, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=0,
              callbacks=[early_stopping])

    # Evaluate the model on validation data
    score = model.evaluate(X_test_scaled, y_test, verbose=0)
    return score[1]  # Return MAE as the objective to minimize

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # Reduced number of trials for efficiency

# Print the best score and parameters
print(f'Best MAE: {study.best_value}')
print(f'Best Hyperparameters: {study.best_params}')
