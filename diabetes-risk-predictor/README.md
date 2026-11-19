# Diabetes Risk Predictor

A beginner machine learning project that predicts diabetes disease progression from 10 health measurements (age, BMI, blood pressure, blood test results, and more).

## Dataset
The Diabetes dataset built into scikit-learn — 442 patients, 10 features. No download needed, so the project runs anywhere.

## What the script does
1. Loads and explores the dataset with pandas
2. Splits the data into training and test sets
3. Trains and compares two models:
   - **Linear Regression** (baseline)
   - **Random Forest Regressor**
4. Reports R² score and mean absolute error for each model
5. Shows which health measurements matter most (feature importance)
6. Prints sample predictions vs actual values

## How to run
```bash
pip install -r requirements.txt
python diabetes_predictor.py
```

## What I learned
- Loading a real dataset with scikit-learn
- Train/test split and why it matters
- Comparing models with R² and MAE
- Feature importance in tree-based models
