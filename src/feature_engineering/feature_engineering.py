
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Get the project directory
project_dir = Path(__file__).resolve().parents[2]

# 2. Load the cleaned dataset
data_path = (
    project_dir / "data" / "processed" /
    "cleaned_loan_data.csv"
)

df = pd.read_csv(data_path)

print("Cleaned dataset loaded successfully!")
print("Original shape:", df.shape)


# 3. Investigate the existing loan-to-income ratio
calculated_ratio = (
    df["loan_amnt"] /
    df["person_income"].replace(0, np.nan)
)

difference = (
    calculated_ratio - df["loan_percent_income"]
).abs()

print("\nLoan-to-income ratio investigation:")
print("Maximum difference:", difference.max())


# 4. Select useful input features
selected_features = [
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

target_column = "loan_status"


# 5. Create the feature dataset
feature_df = df[selected_features + [target_column]].copy()

# Calculate a consistent loan-to-income ratio
feature_df["loan_percent_income"] = (
    feature_df["loan_amnt"] /
    feature_df["person_income"].replace(0, np.nan)
)

# Arrange the columns
feature_columns = [
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

feature_df = feature_df[feature_columns + [target_column]]


# 6. Display selected features
print("\nSelected input features:")

for column in feature_columns:
    print("-", column)

print("\nTarget variable:", target_column)

print("\nNew loan-to-income ratio:")
print(feature_df["loan_percent_income"].head())


# 7. Check the feature dataset
print("\nFeature dataset shape:", feature_df.shape)

print("\nMissing values:")
print(feature_df.isnull().sum())


# 8. Save the feature dataset
output_path = (
    project_dir / "data" / "processed" /
    "loan_features.csv"
)

feature_df.to_csv(output_path, index=False)

print("\nFeature engineering completed!")
print("Feature dataset saved at:")
print(output_path)
