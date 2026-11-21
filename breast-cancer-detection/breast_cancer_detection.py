"""
Breast Cancer Detection
=======================
Classify breast tumours as malignant or benign using Logistic Regression
on scikit-learn's built-in Wisconsin Breast Cancer dataset.

The dataset is bundled with scikit-learn, so the script runs fully
offline - no downloads needed.

What it does:
1. Loads the breast cancer dataset (569 samples, 30 measurements).
2. Splits the data into training and test sets.
3. Scales the features and trains a Logistic Regression model.
4. Reports accuracy, a classification report and a confusion matrix.
5. Lists the measurements that most influence the prediction.
6. Shows a prediction for one example patient.

How to run:
    pip install -r requirements.txt
    python breast_cancer_detection.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the dataset and return features, labels and names."""
    data = load_breast_cancer()
    return data.data, data.target, data.target_names, data.feature_names


def train_model(X_train, y_train):
    """Build and train a scaled Logistic Regression model."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(X_train, y_train)
    return model


def main():
    X, y, target_names, feature_names = load_data()
    print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} measurements")
    print(f"Classes: {list(target_names)} (0 = malignant, 1 = benign)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} patients")
    print(f"Test set: {len(X_test)} patients\n")

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.1%})\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()

    # Which measurements matter most? Logistic Regression coefficients
    # show how strongly each scaled feature pushes the prediction.
    coefficients = model.named_steps["logisticregression"].coef_[0]
    ranked = sorted(
        zip(feature_names, coefficients), key=lambda pair: abs(pair[1]), reverse=True
    )
    print("Top 5 most influential measurements:")
    for name, weight in ranked[:5]:
        direction = "towards benign" if weight > 0 else "towards malignant"
        print(f"  {name}: {weight:.2f} ({direction})")
    print()

    # Example: predict for the first patient in the test set.
    example = X_test[0].reshape(1, -1)
    prediction = model.predict(example)[0]
    probability = model.predict_proba(example)[0]
    print("Example patient from the test set:")
    print(f"  Actual:    {target_names[y_test[0]]}")
    print(f"  Predicted: {target_names[prediction]}")
    print(f"  Confidence: {max(probability):.1%} (malignant {probability[0]:.1%}, "
          f"benign {probability[1]:.1%})")


if __name__ == "__main__":
    main()
