# Breast Cancer Detection

A machine learning classifier that predicts whether a breast tumour is
**malignant** or **benign** from 30 cell measurements, using Logistic
Regression on scikit-learn's built-in Wisconsin Breast Cancer dataset.

## Features

- Uses the dataset bundled with scikit-learn — no downloads needed, runs offline
- Train/test split (80/20) with stratified sampling
- Feature scaling with `StandardScaler` inside a pipeline
- Reports accuracy, a full classification report and a confusion matrix
- Ranks the measurements that most influence the prediction
- Shows a worked prediction for one example patient, with confidence

## How to run

```bash
pip install -r requirements.txt
python breast_cancer_detection.py
```

## Requirements

- Python 3.9+
- scikit-learn

See `requirements.txt` for the exact dependencies.

## Results

On the held-out test set (114 patients) the model reaches roughly 97–98%
accuracy. The most influential measurements are typically the mean radius,
mean concave points and mean area of the cell nucleus.

## What was learned

- Loading a built-in dataset with `sklearn.datasets.load_breast_cancer`
- Splitting data with `train_test_split` and why `stratify` matters
- Why features should be scaled before Logistic Regression
- Building a model with `make_pipeline` so scaling and training stay together
- Reading a classification report, confusion matrix and model coefficients
