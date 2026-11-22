# Wine Quality Classifier

A machine learning project that predicts which cultivar (type) a wine belongs to based on its chemical analysis — using a Random Forest classifier.

## Dataset

The built-in **Wine dataset** from scikit-learn: 178 wine samples, 13 chemical features (alcohol, malic acid, ash, magnesium, phenols, flavanoids, colour intensity, hue, and more) and 3 cultivars (classes 0, 1 and 2). The data ships with scikit-learn, so no download is needed.

## Features

- Loads the Wine dataset with `sklearn.datasets.load_wine`
- Splits the data into training (75%) and test (25%) sets
- Trains a Random Forest classifier (100 trees)
- Reports accuracy, a full classification report and a confusion matrix
- Ranks the chemical features by importance
- Shows an example prediction on a single wine

## How to run

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the script:

```bash
python wine_quality_classifier.py
```

## Example output

```
Dataset: 178 wines, 13 chemical features
Cultivars: class_0, class_1, class_2

Training samples: 133, test samples: 45

Accuracy: 100.00%

Classification report:
              precision    recall  f1-score   support
     class_0       1.00      1.00      1.00        15
     class_1       1.00      1.00      1.00        18
     class_2       1.00      1.00      1.00        12
...
Top 5 most important features:
  flavanoids: 0.193
  color_intensity: 0.174
  proline: 0.155
```

## What I learned

- Loading a built-in dataset with scikit-learn
- Splitting data with `train_test_split` (with `stratify` to keep class balance)
- Training a Random Forest classifier
- Evaluating a model with accuracy, classification reports and confusion matrices
- Reading feature importances to see which inputs matter most
