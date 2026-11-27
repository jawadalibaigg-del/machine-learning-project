# Customer Churn Predictor

Predict whether a customer will leave (churn) or stay, and compare two classifiers to see which one is better at spotting churners.

## What is churn?
Churn is when a customer stops using a service — cancels a subscription, closes an account, switches to a competitor. Companies want to find the customers who are about to leave **before** they leave, so they can offer them a discount or better support.

## How it works
1. **Synthetic data** — 1,000 fake customers are generated with `make_classification` (no downloads needed). Each customer has 8 features (tenure, monthly charges, support calls, satisfaction score...) and a label: `1 = churned`, `0 = stayed`. About 25% churn, like in real life.
2. **Train/test split** — 80% of customers train the models, 20% test them (`stratify` keeps the 25% churn share in both parts).
3. **Feature scaling** — `StandardScaler` puts all features on the same scale so Logistic Regression can learn properly.
4. **Two models trained and compared:**
   - Logistic Regression
   - Random Forest (100 trees)
5. **Evaluation** — accuracy, ROC-AUC, confusion matrix and classification report for each model, plus the most important features from the Random Forest.

## How to run
```bash
pip install -r requirements.txt
python customer_churn_predictor.py
```

## Example output
```
Customer Churn Predictor
========================================
Customers: 1000  |  Churned: 250  |  Stayed: 750

===== Logistic Regression =====
Accuracy:  0.8200
ROC-AUC:   0.8331

===== Random Forest =====
Accuracy:  0.9150
ROC-AUC:   0.9688

Best model: Random Forest (accuracy 0.9150)

Most important features (Random Forest):
  satisfaction_score   0.241
  support_calls        0.160
  monthly_charges      0.153
  contract_length      0.141
  ...
```

## What was learned
- **Synthetic data** with `make_classification` — practising ML without downloading a real dataset
- **Train/test split** with `stratify` — testing on data the model has never seen
- **Feature scaling** (`StandardScaler`) — why some models need features on the same scale
- **Logistic Regression** — a simple linear classifier that outputs probabilities
- **Random Forest** — an ensemble of many decision trees voting together
- **Confusion matrix** — true/false positives and negatives, not just one accuracy number
- **ROC-AUC** — how well a model ranks churners above stayers, across all thresholds
- **Feature importances** — which customer facts matter most for the prediction
