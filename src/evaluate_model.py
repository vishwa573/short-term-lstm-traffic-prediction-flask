import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data_path = "C:/Traffic_pred_Project/Data/processed_traffic_data.csv"
df = pd.read_csv(data_path)

target_columns = ["Traffic_Volume", "Congestion_Level", "Average_Speed"]

# Prepare sequences
sequence_length = 4
def create_sequences(data, labels, seq_length):
    sequences, target_labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        target_labels.append(labels[i + seq_length])
    return np.array(sequences), np.array(target_labels)

# Prepare data
features = df.drop(columns = target_columns).values
labels = df[target_columns].values

X, y = create_sequences(features, labels, sequence_length)

# print("\nTarget Values After Scaling (Before Training):")
# print(pd.DataFrame(labels).describe())



# Split data
split = int(0.8 * len(X))
split_end = int(0.9 * len(X))
X_train, X_test = X[:split], X[split_end:]
y_train, y_test = y[:split], y[split_end:]

# Load trained model
model_path = "C:/Traffic_pred_Project/Model/lstm_model_optimized.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer="adam", loss="mse")

# Predict
y_pred = model.predict(X_test)
# print("Predicted Values:")
# print("Raw Model Predictions (Before Inverse Scaling):")
# print(y_pred[:10])  # Print first 10 predictions
# print("Shape of Predictions:", y_pred.shape)
# print("Mean:", np.mean(y_pred, axis=0))
# print("Std Dev:", np.std(y_pred, axis=0))
# print("Min:", np.min(y_pred, axis=0))
# print("Max:", np.max(y_pred, axis=0))

# Inverse transform predictions
# Load saved scalers
feature_scaler_path = r"C:/Traffic_pred_Project/scalar/X_scaler.npy"
target_scaler_path =  r"C:/Traffic_pred_Project/scalar/y_scaler.npy"

X_scaler = np.load(feature_scaler_path, allow_pickle=True).item()
y_scaler = np.load(target_scaler_path, allow_pickle=True).item()
y_test_original = y_scaler.inverse_transform(y_test)
y_pred_original = y_scaler.inverse_transform(y_pred)

print("\nPredicted Values After Inverse Transform:")
print(pd.DataFrame(y_pred_original).describe())

y_test_sample = y_test_original[:10]  # First 10 actual values
y_pred_sample = y_pred_original[:10]  # First 10 predicted values

for i in range(len(y_test_sample)):
    print(f"Actual: {y_test_sample[i]}, Predicted: {y_pred_sample[i]}")

# Compute performance metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

r2_avg_speed = r2_score(y_test_original[:, 2], y_pred_original[:, 2])
r2_traffic_volume = r2_score(y_test_original[:, 0], y_pred_original[:, 0])
r2_congestion_level = r2_score(y_test_original[:, 1], y_pred_original[:, 1])

print("Model Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"R² Score for Average Speed: {r2_avg_speed}")
print(f"R² Score for Traffic Volume: {r2_traffic_volume}")
print(f"R² Score for Congestion Level: {r2_congestion_level}")

# Visualization
plt.figure(figsize=(12, 4))

# Average Speed
plt.subplot(1, 3, 1)
plt.hist(y_test_original[:, 2], bins=30, alpha=0.5, label="Actual", color='blue')
plt.hist(y_pred_original[:, 2], bins=30, alpha=0.5, label="Predicted", color='red')
plt.xlabel("Average Speed")
plt.legend()

# Traffic Volume
plt.subplot(1, 3, 2)
plt.hist(y_test_original[:, 0], bins=30, alpha=0.5, label="Actual", color='blue')
plt.hist(y_pred_original[:, 0], bins=30, alpha=0.5, label="Predicted", color='red')
plt.xlabel("Traffic Volume")
plt.legend()

# Congestion Level
plt.subplot(1, 3, 3)
plt.hist(y_test_original[:, 1], bins=30, alpha=0.5, label="Actual", color='blue')
plt.hist(y_pred_original[:, 1], bins=30, alpha=0.5, label="Predicted", color='red')
plt.xlabel("Congestion Level")
plt.legend()

plt.tight_layout()
plt.show()
