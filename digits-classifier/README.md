# Handwritten Digit Classifier

A machine learning project that recognizes handwritten digits (0–9) from their pixel data — using a Random Forest classifier.

## Dataset

The built-in **Digits dataset** from scikit-learn: 1,797 images of handwritten digits, each 8×8 pixels (64 features), labelled 0–9. The data ships with scikit-learn, so no download is needed.

## Features

- Loads the Digits dataset with `sklearn.datasets.load_digits`
- Splits the data into training (80%) and test (20%) sets, keeping the digit balance with `stratify`
- Trains a Random Forest classifier (100 trees)
- Reports accuracy, a full classification report and a confusion matrix
- Shows example predictions and prints a digit image as ASCII art so you can "see" what the model saw

## How to run

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the script:

```bash
python digits_classifier.py
```

## Example output

```
Dataset: 1797 handwritten digit images, 64 pixel features
Digits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Images are 8x8 pixels

Training samples: 1437, test samples: 360

Accuracy: 96.11%

Classification report:
              precision    recall  f1-score   support
           0       0.97      0.97      0.97        36
           1       0.90      0.97      0.93        36
           2       1.00      0.97      0.99        35
           ...
```

## What I learned

- Loading a built-in image dataset with scikit-learn
- Working with image data as flat feature vectors (64 pixels per image)
- Splitting data with `train_test_split` (with `stratify` to keep class balance)
- Training a Random Forest classifier
- Evaluating a model with accuracy, classification reports and confusion matrices
- Visualizing a small image as ASCII art
