import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from pyswarm import pso  # Particle Swarm Optimization

# Load dataset
data_path = "C:/Traffic_pred_Project/Data/processed_traffic_data.csv"
df = pd.read_csv(data_path)

# Select target variables
target_columns = ["Traffic_Volume", "Congestion_Level", "Average_Speed"]

# Prepare data for LSTM
sequence_length = 4
def create_sequences(data, labels, seq_length):
    sequences, target_labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        target_labels.append(labels[i + seq_length])
    return np.array(sequences), np.array(target_labels)


features_scaled = df.drop(columns= target_columns).values
# features = features[X_scaler.feature_names_in_] #To maintain order
labels_scaled = df[target_columns].values



X, y = create_sequences(features_scaled, labels_scaled, sequence_length)

# Split data
split = int(0.8 * len(X))
split_end = int(0.9 * len(X))
X_train, X_test = X[:split], X[split:split_end]
y_train, y_test = y[:split], y[split:split_end]

# Define function to train & evaluate model
def train_and_evaluate(params):
    lstm_units, dropout_rate, batch_size, learning_rate = params
    print("Evaluating hyperparameters:", params)

    model = Sequential([
        Input(shape=(sequence_length, X.shape[2])),
        LSTM(int(lstm_units), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(int(lstm_units), return_sequences=False),
        Dropout(dropout_rate),
        Dense(len(target_columns))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), 
              verbose=0, validation_data=(X_test, y_test),
              callbacks=[early_stopping])  # Added EarlyStopping

    loss = model.evaluate(X_test, y_test, verbose=0)  # Return multiple metrics
    return loss

# Define hyperparameter search space
lb = [10, 0.1, 16, 0.0001]  # Lower bounds (LSTM units, dropout, batch size, learning rate)
ub = [100, 0.5, 128, 0.005]  # Upper bounds

# Run PSO optimization
best_params, best_loss = pso(train_and_evaluate, lb, ub, swarmsize=15, maxiter=22)

print("Best hyperparameters found by PSO:", best_params)

# Train final model with optimized hyperparameters
lstm_units, dropout_rate, batch_size, learning_rate = best_params

final_model = Sequential([
    Input(shape=(sequence_length, X.shape[2])),
    LSTM(int(lstm_units), return_sequences=True),
    Dropout(dropout_rate),
    LSTM(int(lstm_units), return_sequences=False),
    Dropout(dropout_rate),
    Dense(len(target_columns))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
final_model.compile(optimizer=optimizer, loss='mse')  
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Final model training with best hyperparameters...")
final_model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), 
                verbose=1, validation_data=(X_test, y_test),
                callbacks=[early_stopping])

# Save model
model_path = "C:/Traffic_pred_Project/Model/lstm_model_optimized.h5"
final_model.save(model_path)
