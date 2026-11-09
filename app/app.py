
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

# 1. Locate project directory
project_dir = Path(__file__).resolve().parents[1]

# 2. Initialize Flask

app = Flask(__name__)
# 3. Load saved preprocessing pipeline
preprocessor = joblib.load(
    project_dir / "models" / "preprocessor.pkl"
)

# 4. Load final trained model and threshold
saved_model = joblib.load(
    project_dir / "models" / "loanguard_model.pkl"
)

model = saved_model["model"]
threshold = saved_model["threshold"]


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:
            # 5. Collect applicant information
            income = float(request.form["income"])
            home = request.form["home"].strip().upper()
            employment = float(request.form["employment"])
            intent = request.form["intent"].strip().upper()
            loan_amount = float(request.form["loan_amount"])
            previous_default = (
                request.form["previous_default"].strip().upper()
            )
            credit_history = int(request.form["credit_history"])

            # 6. Validate inputs
            if income <= 0 or loan_amount <= 0:
                raise ValueError(
                    "Income and loan amount must be greater than zero."
                )

            if employment < 0 or credit_history < 0:
                raise ValueError(
                    "Employment and credit history cannot be negative."
                )

            if home not in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
                raise ValueError("Invalid home ownership category.")

            if intent not in [
                "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
            ]:
                raise ValueError("Invalid loan purpose.")

            if previous_default not in ["Y", "N"]:
                raise ValueError("Invalid previous default value.")

            # 7. Calculate loan-to-income ratio
            loan_percent_income = loan_amount / income

            # 8. Prepare applicant data
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

            # 9. Apply preprocessing
            applicant_processed = preprocessor.transform(applicant)

            # 10. Predict default probability
            probability = model.predict_proba(
                applicant_processed
            )[0][1]

            # 11. Apply selected threshold
            prediction = int(probability >= threshold)

            if prediction == 1:
                category = "HIGH RISK"
            else:
                category = "LOW RISK"

            # 12. Prepare result for frontend
            result = {
                "category": category,
                "probability": round(probability * 100, 2),
                "threshold": round(threshold * 100, 2),
                "loan_ratio": round(loan_percent_income * 100, 2)
            }

        except (ValueError, TypeError) as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
