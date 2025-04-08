import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

print(os.getcwd())

# Load dataset
df = pd.read_csv(r"C:\Traffic_pred_Project\Data\updated_dataset.csv")
#df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

# Drop categorical columns that can't be directly used in LSTM
df = df.drop(columns=["date","Area Name", "Road/Intersection Name"])
df["Roadwork and Construction Activity"] = df["Roadwork and Construction Activity"].map({"No": 0, "Yes": 1})
df = pd.get_dummies(df, columns=["Weather Conditions"])
df.columns = df.columns.str.strip().str.replace(" ", "_") 

# Separate features (X) and targets (y)
feature_cols = [col for col in df.columns if col not in ["Traffic_Volume", "Congestion_Level","Average_Speed"]]
target_cols = ["Traffic_Volume","Congestion_Level", "Average_Speed" ]

X = df[feature_cols].values
y = df[target_cols].values

# Normalize features
X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# # Print column names
# print("Column names in X_scaled DataFrame:")
# print(X_scaled_df.columns.tolist())


# Normalize target variables
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Combine back into a DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled[target_cols] = y_scaled


df_scaled.to_csv(r"C:\Traffic_pred_Project\Data\processed_traffic_data.csv", index=False)
# target_columns = ["Traffic_Volume", "Congestion_Level", "Average_Speed"]
# df_test = df_scaled.drop(columns = target_cols)
df_scaled.head(20).to_csv(r"C:\Traffic_pred_Project\Data\test_data.csv", index=False)
np.save(r"C:\Traffic_pred_Project\scalar\X_scaler.npy", X_scaler)
np.save(r"C:\Traffic_pred_Project\scalar\y_scaler.npy", y_scaler)

print("Preprocessing complete. Processed data saved.")
