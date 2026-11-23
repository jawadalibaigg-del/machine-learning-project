"""Handwritten Digit Classifier - recognize digits 0-9 from pixel data.

Uses the built-in Digits dataset from scikit-learn: 1,797 images of
handwritten digits (8x8 pixels each, 64 features). A Random Forest
classifier learns to recognize the digit in each image.

No data download is needed - the dataset ships with scikit-learn.
"""

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    # Load the built-in Digits dataset (1,797 samples, 64 pixel features).
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset: {X.shape[0]} handwritten digit images, {X.shape[1]} pixel features")
    print(f"Digits: {sorted(set(y))}")
    print(f"Images are {digits.images[0].shape[0]}x{digits.images[0].shape[1]} pixels")

    # Split into training (80%) and test (20%) sets, keeping class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}, test samples: {len(X_test)}")

    # Train a Random Forest classifier.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("Confusion matrix (rows = actual digit, columns = predicted digit):")
    print(confusion_matrix(y_test, predictions))

    # Show a few example predictions, including one image as ASCII art.
    print("\nExample predictions:")
    for index in range(5):
        actual = y_test[index]
        predicted = predictions[index]
        mark = "OK" if actual == predicted else "WRONG"
        print(f"  Sample {index + 1}: actual = {actual}, predicted = {predicted} [{mark}]")

    # Print the first test image as ASCII art so you can "see" the digit.
    print("\nFirst test image (actual digit: "
          f"{y_test[0]}, predicted: {predictions[0]}):")
    print_digit_as_ascii(X_test[0])


def print_digit_as_ascii(pixels):
    """Print an 8x8 digit image as ASCII art (darker = denser pixels)."""
    shades = " .:-=+*#%@"
    for row in range(8):
        line = ""
        for col in range(8):
            value = pixels[row * 8 + col]
            # Pixel values range 0-16; map them onto the shade characters.
            line += shades[min(int(value / 16 * (len(shades) - 1)), len(shades) - 1)]
        print("  " + line)


if __name__ == "__main__":
    main()
