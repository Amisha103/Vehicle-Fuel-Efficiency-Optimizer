import pandas as pd
import os

print("\n🚀 Loading datasets...")

# Load both datasets (make sure both files are in the 'data' folder)
numerical_df = pd.read_csv("data/dual_fuel_engine_dataset.csv")
categorical_df = pd.read_csv("data/dual_fuel_engine_dataset_categorical.csv")

print(f"✅ Numerical dataset shape: {numerical_df.shape}")
print(f"✅ Categorical dataset shape: {categorical_df.shape}")

# Rename columns to avoid duplicates
numerical_df.columns = [f"num_{col}" for col in numerical_df.columns]
categorical_df.columns = [f"cat_{col}" for col in categorical_df.columns]

# Combine both datasets safely
print("\n🔗 Combining datasets...")
combined_df = pd.concat([numerical_df, categorical_df], axis=1)
print(f"✅ Combined dataset shape: {combined_df.shape}")

# Handle missing values
print("\n🧹 Handling missing values...")
total_missing = combined_df.isnull().sum().sum()
print(f"Total missing values before cleaning: {total_missing}")

# Fill missing values
for col in combined_df.columns:
    col_dtype = combined_df[col].dtypes

    # Numeric columns → fill with mean
    if pd.api.types.is_numeric_dtype(col_dtype):
        if combined_df[col].isnull().any():
            mean_val = combined_df[col].mean()
            combined_df[col].fillna(mean_val, inplace=True)

    # Categorical columns → fill with mode
    else:
        if combined_df[col].isnull().any():
            mode_val = combined_df[col].mode()
            if not mode_val.empty:
                combined_df[col].fillna(mode_val.iloc[0], inplace=True)
            else:
                combined_df[col].fillna("Unknown", inplace=True)

print(f"✅ Missing values after cleaning: {combined_df.isnull().sum().sum()}")

# Save processed dataset
output_path = "data/processed_engine_data.csv"
combined_df.to_csv(output_path, index=False)

print(f"\n💾 Preprocessing complete! Cleaned dataset saved to: {output_path}")
