
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Locate the project folder
project_dir = Path(__file__).resolve().parents[2]

# 2. Load the cleaned dataset
data_path = project_dir / "data" / "processed" / "cleaned_loan_data.csv"
df = pd.read_csv(data_path)

# 3. Create the reports folder if it does not exist
reports_dir = project_dir / "reports"
reports_dir.mkdir(exist_ok=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 4. Count the loan status categories
status_counts = df["loan_status"].value_counts().sort_index()

labels = ["No Default (0)", "Default (1)"]
counts = [
    status_counts.get(0, 0),
    status_counts.get(1, 0)
]

# 5. Calculate percentages
total = sum(counts)

print("\nLoan Status Distribution:")
for label, count in zip(labels, counts):
    percentage = (count / total) * 100
    print(f"{label}: {count} applicants ({percentage:.2f}%)")

# 6. Create a bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(labels, counts)

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

# Display the count above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# 7. Save the chart
chart_path = reports_dir / "loan_status_distribution.png"
plt.savefig(chart_path)

plt.show()

print("\nChart saved successfully at:", chart_path)
