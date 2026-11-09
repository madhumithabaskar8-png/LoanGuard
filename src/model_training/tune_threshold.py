
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Locate project directory
current_path = Path(__file__).resolve()

project_dir = next(
    parent for parent in current_path.parents
    if (parent / "data").is_dir()
)

processed_dir = project_dir / "data" / "processed"
models_dir = project_dir / "models"
reports_dir = project_dir / "reports"

# 2. Load preprocessed data
X_train = np.load(processed_dir / "X_train.npy")
X_test = np.load(processed_dir / "X_test.npy")
y_train = np.load(processed_dir / "y_train.npy")
y_test = np.load(processed_dir / "y_test.npy")

# 3. Split training data into training and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("Training records:", len(X_fit))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

# 4. Train a temporary Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest for threshold tuning...")
model.fit(X_fit, y_fit)

# 5. Get validation probabilities
val_probabilities = model.predict_proba(X_val)[:, 1]

# 6. Test different thresholds
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

results = []

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, val_predictions),
        "Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, val_predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

# 7. Select threshold with the highest validation F1-score
best_index = results_df["F1 Score"].idxmax()
best_threshold = float(
    results_df.loc[best_index, "Threshold"]
)

print("\nSelected threshold:", best_threshold)

# 8. Train final model using the complete training set
final_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# 9. Evaluate once on the untouched test set
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST RESULTS")
print("Threshold:", best_threshold)

print("Accuracy:",
      accuracy_score(y_test, test_predictions))

print("Precision:",
      precision_score(y_test, test_predictions, zero_division=0))

print("Recall:",
      recall_score(y_test, test_predictions, zero_division=0))

print("F1 Score:",
      f1_score(y_test, test_predictions, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_test, test_probabilities))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Default", "Default"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# 10. Save final model and selected threshold together
joblib.dump(
    {
        "model": final_model,
        "threshold": best_threshold
    },
    models_dir / "loanguard_model.pkl"
)

# 11. Save threshold comparison
reports_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    reports_dir / "threshold_comparison.csv",
    index=False
)

print("\nFinal model saved successfully!")
print(models_dir / "loanguard_model.pkl")

print("\nThreshold comparison saved successfully!")
print(reports_dir / "threshold_comparison.csv")
