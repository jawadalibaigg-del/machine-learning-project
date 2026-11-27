"""
Customer Churn Predictor

Churn means a customer leaves a service (cancels a subscription, closes
an account). This project trains two classifiers to predict churn and
compares them, so we can see which one is better at spotting the
customers who are about to leave.

The data is synthetic: it is generated with make_classification, so the
script runs with no downloads. Each customer has 8 features (for
example monthly charges, tenure, number of support calls) and one label:
1 = churned, 0 = stayed.

Models compared:
  1. Logistic Regression - a simple linear classifier
  2. Random Forest - an ensemble of decision trees

Run it with:  python customer_churn_predictor.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Feature names, in the same order as the columns make_classification
# produces. They make the output readable: "tenure_months" is clearer
# than "feature 3".
FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "contract_length",
    "num_services",
    "late_payments",
    "satisfaction_score",
]


def build_dataset():
    """Create 1,000 fake customers and split them into features (X) and label (y)."""
    X, y = make_classification(
        n_samples=1000,        # 1,000 customers
        n_features=8,          # 8 facts per customer
        n_informative=5,       # 5 of the 8 features actually affect churn
        n_redundant=2,         # 2 features are copies/mixes of the informative ones
        n_classes=2,           # two outcomes: churn (1) or stay (0)
        weights=[0.75, 0.25],  # 25% of customers churn, 75% stay (like real life)
        random_state=42,       # fixed seed, so the data is the same every run
    )
    return X, y


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    """Train one model, predict on the test set, and print its scores."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")

    # ROC-AUC needs probabilities, not just 0/1 predictions. Both models
    # here have predict_proba, so we can use it.
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print("                 Predicted Stay  Predicted Churn")
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"Actual Stay            {tn}              {fp}")
    print(f"Actual Churn           {fn}              {tp}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=["Stayed", "Churned"]
    ))

    return accuracy_score(y_test, predictions)


def main():
    print("Customer Churn Predictor")
    print("=" * 40)

    # 1. Build the data.
    X, y = build_dataset()
    print(f"Customers: {len(y)}  |  Churned: {int(y.sum())}  |  Stayed: {int((1 - y).sum())}")

    # 2. Split into train (80%) and test (20%). stratify=y keeps the same
    # 25% churn share in both parts, so the test set is fair.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scale the features. Logistic Regression learns much better when
    # all features are on a similar scale; trees do not need it, but
    # scaling does not hurt them, so we use the same scaled data for both.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train and compare the two models.
    results = {}
    results["Logistic Regression"] = train_and_evaluate(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_scaled, X_test_scaled, y_train, y_test,
    )
    results["Random Forest"] = train_and_evaluate(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train_scaled, X_test_scaled, y_train, y_test,
    )

    # 5. Say which model won, and show which features mattered most.
    best = max(results, key=results.get)
    print(f"\nBest model: {best} (accuracy {results[best]:.4f})")

    # Train one more Random Forest on the full data just to read the
    # feature importances: how much each feature helped the trees decide.
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train_scaled, y_train)
    print("\nMost important features (Random Forest):")
    order = np.argsort(forest.feature_importances_)[::-1]
    for rank in order:
        print(f"  {FEATURE_NAMES[rank]:<20} {forest.feature_importances_[rank]:.3f}")


if __name__ == "__main__":
    main()
