# Loan Approval Predictor

A machine learning project that predicts whether a bank should approve a loan application, using **Logistic Regression**.

## What it does

Given five facts about a loan applicant — monthly income, credit score, requested loan amount, years employed, and existing monthly debt — the model predicts **approved (1)** or **rejected (0)** and gives the approval probability.

## The data

The dataset is **generated inside the script** with NumPy (1,000 fake applicants), so the project runs fully offline — no download needed. Labels come from a realistic scoring rule (good credit score + high income + low debt = likely approved) plus a little random noise, so the data behaves like real bank data.

## How to run

```bash
pip install -r requirements.txt
python loan_approval_predictor.py
```

## What the script does, step by step

1. **Generates data** — 1,000 applicants with realistic feature ranges.
2. **Splits the data** — 80% train, 20% test (`train_test_split`, stratified).
3. **Scales the features** — `StandardScaler` so income (thousands) does not dominate credit score (hundreds).
4. **Trains Logistic Regression** — learns a decision boundary between approved and rejected.
5. **Evaluates** — accuracy, confusion matrix, and classification report on the unseen test set; prints feature coefficients to show which factors mattered most.
6. **Predicts for new applicants** — two example applicants (one strong, one weak) with approval probabilities.

## Results (example run)

- Test accuracy: ~90%
- Credit score and existing debt are the strongest factors, which matches how real banks decide.

## Requirements

- numpy, pandas, scikit-learn (see `requirements.txt`)

## What was learned

- Generating realistic synthetic data with NumPy
- Train/test split and why the test set must stay unseen
- Feature scaling with `StandardScaler` (fit on train only!)
- Logistic Regression for binary classification
- Accuracy, confusion matrix, precision, recall, F1-score
- Reading model coefficients as feature importance
