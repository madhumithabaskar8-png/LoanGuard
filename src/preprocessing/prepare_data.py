
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. Find the main project directory
current_path = Path(__file__).resolve()

project_dir = next(
    parent for parent in current_path.parents
    if (parent / "data").is_dir()
)


# 2. Load feature dataset
data_path = (
    project_dir / "data" / "processed" / "loan_features.csv"
)

df = pd.read_csv(data_path)

print("Feature dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 3. Separate input features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)


# 4. Identify numerical and categorical columns
numerical_features = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]


# 5. Split the dataset before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# 6. Numerical preprocessing pipeline
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# 7. Categorical preprocessing pipeline
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


# 8. Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# 9. Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted preprocessing
X_test_processed = preprocessor.transform(X_test)


# 10. Display preprocessing results
print("\n--- PREPROCESSING RESULTS ---")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nRemaining missing values in processed training data:",
      np.isnan(X_train_processed).sum())

print("Remaining missing values in processed testing data:",
      np.isnan(X_test_processed).sum())


# 11. Save preprocessing pipeline and datasets
models_dir = project_dir / "models"
processed_dir = project_dir / "data" / "processed"

models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    preprocessor,
    models_dir / "preprocessor.pkl"
)

np.save(processed_dir / "X_train.npy", X_train_processed)
np.save(processed_dir / "X_test.npy", X_test_processed)
np.save(processed_dir / "y_train.npy", y_train.to_numpy())
np.save(processed_dir / "y_test.npy", y_test.to_numpy())


# 12. Display final summary
print("\nPreprocessing completed successfully!")

print("Saved preprocessing pipeline:")
print(models_dir / "preprocessor.pkl")

print("\nTraining and testing datasets saved successfully!")
