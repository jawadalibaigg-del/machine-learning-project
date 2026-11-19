"""Diabetes Risk Predictor - a beginner machine learning project.

Predicts diabetes disease progression (a number where higher = worse
progression one year later) from 10 health measurements such as age,
BMI, blood pressure and blood test results.

Dataset: the classic Diabetes dataset built into scikit-learn
(442 patients, 10 features) - no download needed.

Models compared:
- Linear Regression (baseline)
- Random Forest Regressor

Run:
    pip install -r requirements.txt
    python diabetes_predictor.py
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # 1. Load the data
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="progression")

    print("Dataset shape:", X.shape)
    print("Features:", list(X.columns))
    print()

    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train and compare two models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = []
    best_name, best_model, best_r2 = None, None, float("-inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        results.append((name, r2, mae))
        print(f"{name:18}  R2 score: {r2:.3f}   Mean absolute error: {mae:.1f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2

    print(f"\nBest model: {best_name} (R2 = {best_r2:.3f})")

    # 4. Which measurements matter most? (Random Forest feature importance)
    if hasattr(best_model, "feature_importances_"):
        print("\nMost important health measurements:")
        importance = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        for feature, value in importance.head(5).items():
            print(f"  {feature:6} {'#' * int(value * 50)} {value:.2f}")

    # 5. Predict progression for the first 5 test patients
    print("\nSample predictions (first 5 test patients):")
    sample_predictions = best_model.predict(X_test.head())
    for actual, predicted in zip(y_test.head(), sample_predictions):
        print(f"  Actual: {actual:6.1f}   Predicted: {predicted:6.1f}")


if __name__ == "__main__":
    main()
