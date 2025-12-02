#!/usr/bin/env python3
"""
Script to merge prices.csv into GetData_with_features.csv
Performs a left join on the 'item' column, filling missing prices with 0
"""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path("UpdatedData")
features_file = data_dir / "GetData_with_features.csv"
prices_file = data_dir / "prices.csv"

print("Loading GetData_with_features.csv...")
# Load the main features file
df_features = pd.read_csv(features_file)

print("Loading prices.csv...")
# Load the prices file
df_prices = pd.read_csv(prices_file)

# Convert price to numeric, coercing errors to NaN
df_prices['price'] = pd.to_numeric(df_prices['price'], errors='coerce')

print(f"Features file shape: {df_features.shape}")
print(f"Prices file shape: {df_prices.shape}")

# Perform left join on 'item' column
print("Merging data...")
df_merged = df_features.merge(df_prices, on='item', how='left')

# Fill missing prices with 0
print("Filling missing prices with 0...")
df_merged['price'] = df_merged['price'].fillna(0).astype(float)

print(f"Merged file shape: {df_merged.shape}")
print(f"Items with price > 0: {(df_merged['price'] > 0).sum()}")
print(f"Items with price = 0: {(df_merged['price'] == 0).sum()}")

# Save the merged result back to GetData_with_features.csv
print("Saving merged data to GetData_with_features.csv...")
df_merged.to_csv(features_file, index=False)

print("Merge completed successfully!")

