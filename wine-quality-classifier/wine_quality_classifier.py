"""Wine Quality Classifier — predict a wine's cultivar from its chemistry.

Uses scikit-learn's built-in Wine dataset (178 samples, 13 chemical
features, 3 cultivars). No downloads needed: the data ships with
scikit-learn.

Trains a Random Forest classifier, evaluates it on a held-out test set,
and prints a classification report plus the most important features.
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Wine dataset and return features, labels and names."""
    data = load_wine()
    return data.data, data.target, data.target_names, data.feature_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier on the training split."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, target_names):
    """Print accuracy, classification report and confusion matrix."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=target_names))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))
    return accuracy


def show_feature_importance(model, feature_names, top_n=5):
    """Print the most important chemical features."""
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    print(f"\nTop {top_n} most important features:")
    for name, importance in importances[:top_n]:
        print(f"  {name}: {importance:.3f}")


def main():
    X, y, target_names, feature_names = load_data()
    print(f"Dataset: {X.shape[0]} wines, {X.shape[1]} chemical features")
    print(f"Cultivars: {', '.join(target_names)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, test samples: {len(X_test)}\n")

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, target_names)
    show_feature_importance(model, feature_names)

    # Example: classify a single wine from the test set.
    sample = X_test[0].reshape(1, -1)
    predicted = model.predict(sample)[0]
    print(
        f"\nExample prediction: first test wine -> "
        f"{target_names[predicted]} (actual: {target_names[y_test[0]]})"
    )


if __name__ == "__main__":
    main()
