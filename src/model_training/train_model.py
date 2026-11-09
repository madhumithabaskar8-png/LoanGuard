
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Locate the project directory
current_path = Path(__file__).resolve()
project_dir = next(
    parent for parent in current_path.parents
    if (parent / "data").is_dir()
)

processed_dir = project_dir / "data" / "processed"
models_dir = project_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# 2. Load preprocessed training and testing data
X_train = np.load(processed_dir / "X_train.npy")
X_test = np.load(processed_dir / "X_test.npy")
y_train = np.load(processed_dir / "y_train.npy")
y_test = np.load(processed_dir / "y_test.npy")

print("Preprocessed datasets loaded successfully!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 3. Define the machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

# 4. Train and evaluate each model
for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test records
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Default", "Default"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model
    model_filename = (
        "logistic_regression.pkl"
        if model_name == "Logistic Regression"
        else "random_forest.pkl"
    )

    joblib.dump(model, models_dir / model_filename)

    print("\nModel saved:", models_dir / model_filename)

    # Store evaluation results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

# 5. Compare both models
results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

# 6. Save comparison results
results_path = project_dir / "reports" / "model_comparison.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(results_path, index=False)

print("\nModel comparison saved at:")
print(results_path)

print("\nModel training completed successfully!")
