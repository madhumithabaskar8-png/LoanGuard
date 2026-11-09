
# LoanGuard – Loan Repayment Risk Prediction

## Project Overview

LoanGuard is a Machine Learning-based web application
that predicts the repayment risk category of loan applicants.

It uses applicant information to classify loan
repayment risk into:

- LOW RISK
- HIGH RISK

The application is designed as a decision-support
prototype and does not make actual lending decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Flask
- HTML
- CSS
- Joblib

## Machine Learning Model

LoanGuard uses a trained Random Forest Classifier.

The model uses a classification threshold of 0.55
to determine the predicted risk category.

## Applicant Details

The application collects:

- Annual Income
- Home Ownership
- Employment Length
- Loan Purpose
- Loan Amount
- Previous Default History
- Credit History Length

## Project Features

- Applicant information input
- Data preprocessing
- Loan repayment risk prediction
- LOW RISK / HIGH RISK classification
- Simple web interface

## Project Structure

```text
LoanGuard/
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run Locally

1. Clone the repository.
2. Install the required packages.
3. Activate your virtual environment.
4. Run the Flask application.

```bash
pip install -r requirements.txt
python app/app.py
```

5. Open http://127.0.0.1:5000 in your browser.

## Disclaimer

LoanGuard is an educational machine learning project.
Its predictions should not be treated as actual
financial or lending advice.
