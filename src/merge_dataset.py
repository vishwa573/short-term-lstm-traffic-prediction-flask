import pandas as pd

# Load datasets
df = pd.read_csv(r"C:\Traffic_pred_Project\Data\Banglore_traffic_Dataset.csv")
holidays = pd.read_csv(r"C:\Traffic_pred_Project\Data\Public_holidays.csv")

# Convert to datetime format
df["date"] = pd.to_datetime(df["date"])
holidays["date"] = pd.to_datetime(holidays["date"],format="%d/%m/%Y")
holidays["is_public_holiday"] = 1

# Merge data
df = df.merge(holidays, on="date", how="left")
df["is_public_holiday"] = df["is_public_holiday"].fillna(0).astype(int)

# Save cleaned dataset
df.to_csv(r"C:\Traffic_pred_Project\Data\updated_dataset.csv", index=False)
print("Data preprocessing complete! Saved ")
