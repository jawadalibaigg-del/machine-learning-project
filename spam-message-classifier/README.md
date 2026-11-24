# Spam Message Classifier

A machine learning project that tells spam SMS messages apart from genuine ones, using TF-IDF features and a Multinomial Naive Bayes classifier.

## Dataset

A small hand-built dataset of **72 labelled messages** (36 spam, 36 ham) written in the style of the classic SMS Spam Collection. It is built into the script itself, so no download is needed and the project runs offline.

## Features

- Builds a labelled dataset of spam and ham messages inside the script
- Splits the data into training (75%) and test (25%) sets with `stratify`, keeping the spam/ham balance
- Converts text into numbers with **TF-IDF** (`TfidfVectorizer`, English stop words removed)
- Trains a **Multinomial Naive Bayes** classifier — the classic model for text classification
- Reports accuracy and a full classification report (precision, recall, F1)
- Tests the model on 4 fresh example messages and prints SPAM / HAM for each

## How to run

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the script:

```bash
python spam_classifier.py
```

## Example output

```
Dataset: 72 messages
Spam: 36, Ham: 36

Training samples: 54, test samples: 18

Accuracy: 83.33%

Classification report:
              precision    recall  f1-score   support

         ham       0.88      0.78      0.82         9
        spam       0.80      0.89      0.84         9

    accuracy                           0.83        18
   macro avg       0.84      0.83      0.83        18
weighted avg       0.84      0.83      0.83        18

Example predictions:
  [SPAM] Congratulations! You have won a FREE trip to Dubai, call now to claim
  [HAM ] Hey, are you coming to the study group tonight?
  [SPAM] URGENT: verify your account now or it will be suspended
  [HAM ] Can you bring the charger when you come over?
```

## What I learned

- Text has to become numbers before a model can use it — TF-IDF scores each word by how often it appears in a message versus how rare it is across all messages
- Removing stop words ("the", "you", "and") keeps the model focused on the words that actually signal spam
- Naive Bayes works well on text because it learns how likely each word is in spam versus ham, then combines those clues
- `stratify` in `train_test_split` keeps the spam/ham ratio the same in both sets, so the test result is honest
- Small datasets can still give good test scores — but with only 18 test messages, one mistake drops accuracy by ~5.5%, so always check the report, not just the headline number
