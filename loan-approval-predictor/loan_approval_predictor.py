"""
Loan Approval Predictor — a beginner machine learning project.

A bank wants to guess, before giving a loan, whether an applicant will
repay it. This project builds a Logistic Regression model that predicts
"approved" (1) or "rejected" (0) from five applicant features.

The data is generated inside this script with NumPy, so it runs fully
offline — no download is needed. The labels come from a simple realistic
rule (good credit score + low debt + stable income = likely approved)
with a bit of randomness, so the data behaves like real bank data.

Run it with:  python loan_approval_predictor.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 1. Generate a realistic synthetic dataset
# ---------------------------------------------------------------------------
# random_state makes the numbers the same every time the script runs,
# so results are reproducible.
rng = np.random.default_rng(42)
N = 1000  # number of fake loan applicants

# Each feature is drawn from a realistic range:
#   income        : monthly income in dollars        (2,000 - 12,000)
#   credit_score  : credit score                     (300 - 850)
#   loan_amount   : requested loan in dollars        (5,000 - 100,000)
#   years_employed: years at the current job         (0 - 30)
#   existing_debt : other monthly debt in dollars    (0 - 5,000)
income = rng.uniform(2000, 12000, N)
credit_score = rng.uniform(300, 850, N)
loan_amount = rng.uniform(5000, 100000, N)
years_employed = rng.uniform(0, 30, N)
existing_debt = rng.uniform(0, 5000, N)

# The label comes from a simple scoring rule, the way a bank might think:
# a high credit score helps, a high income helps, a big loan hurts, and
# a lot of existing debt hurts. A little random noise is added so the
# data is not perfectly separable (like real life).
score = (
    0.6 * (credit_score - 300) / 550      # credit score: 0 to 0.6
    + 0.3 * (income - 2000) / 10000       # income:       0 to 0.3
    - 0.4 * loan_amount / 100000          # loan size:    0 to -0.4
    - 0.3 * existing_debt / 5000          # debt:         0 to -0.3
    + 0.1 * years_employed / 30           # job stability: 0 to 0.1
    + rng.normal(0, 0.15, N)              # random noise
)
approved = (score > 0.35).astype(int)     # 1 = approved, 0 = rejected

data = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "years_employed": years_employed,
    "existing_debt": existing_debt,
    "approved": approved,
})

print("Dataset shape:", data.shape)
print("Approval rate: {:.1f}%".format(100 * data["approved"].mean()))
print(data.head(), "\n")

# ---------------------------------------------------------------------------
# 2. Split into features (X) and target (y), then train / test
# ---------------------------------------------------------------------------
X = data.drop("approved", axis=1)   # the 5 input columns
y = data["approved"]                # the answer we want to predict

# 80% of the rows train the model, 20% test it. stratify=y keeps the same
# approved/rejected ratio in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Scale the features
# ---------------------------------------------------------------------------
# Income is in thousands, credit score in hundreds — without scaling, the
# big numbers would dominate the model. StandardScaler gives every feature
# mean 0 and standard deviation 1. fit on TRAIN only, then transform both.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train a Logistic Regression model
# ---------------------------------------------------------------------------
# Logistic Regression draws a straight line (in 5 dimensions) that best
# separates approved from rejected applicants, and outputs a probability.
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate on the test set (data the model has never seen)
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_pred)))
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Which features mattered most? A large positive coefficient pushes the
# model toward "approved", a large negative one toward "rejected".
print("Feature importance (model coefficients):")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name:15s} {coef:+.3f}")

# ---------------------------------------------------------------------------
# 6. Predict for a brand-new applicant
# ---------------------------------------------------------------------------
def predict_applicant(income, credit_score, loan_amount, years_employed, existing_debt):
    """Predict approval for one new applicant and print the probability."""
    applicant = pd.DataFrame([{
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "years_employed": years_employed,
        "existing_debt": existing_debt,
    }])
    applicant_scaled = scaler.transform(applicant)
    probability = model.predict_proba(applicant_scaled)[0][1]
    decision = "APPROVED" if probability >= 0.5 else "REJECTED"
    print(f"\nNew applicant -> {decision} "
          f"(approval probability: {probability:.1%})")
    return decision


# A strong applicant: high income, great credit score, small loan, no debt.
predict_applicant(income=9000, credit_score=780, loan_amount=15000,
                  years_employed=8, existing_debt=200)

# A weak applicant: low income, poor credit score, big loan, heavy debt.
predict_applicant(income=2500, credit_score=420, loan_amount=80000,
                  years_employed=1, existing_debt=4500)
