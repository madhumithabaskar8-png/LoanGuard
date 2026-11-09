
import pandas as pd
import joblib
from pathlib import Path

# 1. Locate project directory
current_path = Path(__file__).resolve()

project_dir = next(
    parent for parent in current_path.parents
    if (parent / "models").is_dir()
)

# 2. Load saved preprocessing pipeline and final model
preprocessor = joblib.load(
    project_dir / "models" / "preprocessor.pkl"
)

saved_model = joblib.load(
    project_dir / "models" / "loanguard_model.pkl"
)

model = saved_model["model"]
threshold = saved_model["threshold"]

# 3. Collect applicant details
print("\n========== LOANGUARD ==========")
print("Loan Repayment Risk Prediction")
print("===============================\n")

income = float(input("Enter annual income: "))
home = input(
    "Home ownership (RENT/MORTGAGE/OWN/OTHER): "
).strip().upper()

employment = float(
    input("Enter employment length in years: ")
)

intent = input(
    "Loan purpose (EDUCATION/MEDICAL/VENTURE/PERSONAL/"
    "DEBTCONSOLIDATION/HOMEIMPROVEMENT): "
).strip().upper()

loan_amount = float(input("Enter requested loan amount: "))

previous_default = input(
    "Previous default on file? (Y/N): "
).strip().upper()

credit_history = int(
    input("Enter credit history length in years: ")
)

# 4. Basic input validation
if income <= 0:
    raise ValueError("Annual income must be greater than zero.")

if loan_amount <= 0:
    raise ValueError("Loan amount must be greater than zero.")

if employment < 0 or credit_history < 0:
    raise ValueError("Employment and credit history cannot be negative.")

if home not in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
    raise ValueError("Invalid home ownership category.")

if intent not in [
    "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
]:
    raise ValueError("Invalid loan purpose.")

if previous_default not in ["Y", "N"]:
    raise ValueError("Enter Y or N for previous default.")

# 5. Calculate loan-to-income ratio
loan_percent_income = loan_amount / income

# 6. Create applicant DataFrame
applicant = pd.DataFrame([{
    "person_income": income,
    "person_home_ownership": home,
    "person_emp_length": employment,
    "loan_intent": intent,
    "loan_amnt": loan_amount,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": previous_default,
    "cb_person_cred_hist_length": credit_history
}])

# 7. Apply the same preprocessing used during training
applicant_processed = preprocessor.transform(applicant)

# 8. Predict default probability
default_probability = model.predict_proba(
    applicant_processed
)[0][1]

# 9. Apply selected threshold
prediction = int(default_probability >= threshold)

# 10. Display result
print("\n========== PREDICTION RESULT ==========")

print(f"Estimated default probability: {default_probability:.2%}")
print(f"Selected threshold: {threshold:.2f}")

if prediction == 1:
    print("Predicted Category: HIGH RISK")
else:
    print("Predicted Category: LOW RISK")

print("\nNote: This is an educational decision-support prediction.")
print("It is not an automatic loan approval or rejection.")
