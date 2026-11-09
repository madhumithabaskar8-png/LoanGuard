
import pandas as pd
from pathlib import Path

# --------------------------------
# 1. Load original dataset
# --------------------------------

project_dir = Path(__file__).resolve().parents[2]

input_path = project_dir / "data" / "raw" / "credit_risk_dataset.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# --------------------------------
# 2. Remove duplicate records
# --------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate records found:", duplicate_count)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# --------------------------------
# 3. Handle invalid age records
# --------------------------------

invalid_age_count = (
    (df["person_age"] < 18) |
    (df["person_age"] > 100)
).sum()

print("\nInvalid age records found:", invalid_age_count)

df = df[
    (df["person_age"] >= 18) &
    (df["person_age"] <= 100)
].copy()

print("Shape after age cleaning:", df.shape)

# --------------------------------
# 4. Check missing values
# --------------------------------

print("\nMissing values after cleaning:")

print(df.isnull().sum())

# --------------------------------
# 5. Save cleaned dataset
# --------------------------------

output_path = (
    project_dir / "data" / "processed" /
    "cleaned_loan_data.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

# --------------------------------
# 6. Final dataset summary
# --------------------------------

print("\n--- FINAL CLEANING SUMMARY ---")

print("Original records: 32581")
print("Final records:", len(df))
print("Total columns:", len(df.columns))

print("\nRemaining duplicates:", df.duplicated().sum())

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)
