"""Spam Message Classifier — tell spam SMS messages apart from real ones.

A small, self-contained dataset of labelled messages (spam / ham) is
built into the script, so nothing needs to be downloaded. The text is
turned into features with TF-IDF and a Multinomial Naive Bayes model
learns which words signal spam.

Run it and it will print the accuracy, a classification report and a
few example predictions.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# (message, label) — 1 = spam, 0 = ham (a genuine message).
# A small hand-built sample in the style of the classic SMS Spam dataset.
DATA = [
    # --- spam ---
    ("WINNER!! You have won a $1000 cash prize! Call now to claim", 1),
    ("Congratulations! You have been selected for a FREE holiday. Text YES to claim", 1),
    ("URGENT: Your bank account has been locked. Verify now at http://bit.ly/verify", 1),
    ("You won a brand new iPhone! Click the link to collect your prize", 1),
    ("FREE entry into our weekly competition! Text WIN to 80085 now", 1),
    ("Your loan has been approved! Get $5000 cash today, no credit check", 1),
    ("Claim your free gift card now before the offer expires!", 1),
    ("You are our lucky winner! Reply YES to receive your $500 voucher", 1),
    ("Limited time offer: 90% off designer watches. Order now!", 1),
    ("Your mobile number has won £2000. Call 09001234567 to claim now", 1),
    ("Double your money in 7 days! Guaranteed returns, invest now", 1),
    ("Congratulations, you have won a free cruise! Call this number immediately", 1),
    ("Get rich quick! Earn $500 a day from home, no experience needed", 1),
    ("Your package delivery failed. Click here to reschedule and pay the fee", 1),
    ("You have 1 new voicemail from an unknown number. Call now to listen", 1),
    ("Act now! Your account will be closed in 24 hours unless you verify", 1),
    ("FREE ringtones! Download the latest hits now, text RING to 80810", 1),
    ("You have been chosen to test our new product and get paid $100", 1),
    ("Win a luxury car! Enter the draw today, only £1 per entry", 1),
    ("Your subscription has expired. Renew now to avoid losing access", 1),
    ("Exclusive deal just for you: buy one get three free, today only", 1),
    ("You have won a shopping voucher worth $250! Click to redeem", 1),
    ("Make money fast with crypto! Join now and get a $50 bonus", 1),
    ("Your tax refund is waiting. Submit your bank details to receive it", 1),
    ("You have won $10,000 in our lottery! Claim your prize today", 1),
    ("Act fast! Your credit card has been charged $999, call to cancel", 1),
    ("Get a FREE Samsung phone today, just pay the shipping cost", 1),
    ("You are pre-approved for a $20,000 loan, apply now", 1),
    ("Congratulations! Your number was picked in today's lucky draw", 1),
    ("Earn $1000 weekly working from your phone, sign up free", 1),
    ("Your Netflix account is on hold, update your payment info here", 1),
    ("Claim your cashback of $150 now, limited time offer", 1),
    ("You have an unpaid parking fine of $80, pay now to avoid court", 1),
    ("Invest in Bitcoin now and triple your money this month", 1),
    ("FREE! Get your horoscope and lucky numbers by texting LUCKY", 1),
    ("Your Amazon order could not be delivered, confirm your address", 1),
    # --- ham ---
    ("Hey, are we still meeting for lunch tomorrow at 12?", 0),
    ("Can you pick up some milk and bread on your way home?", 0),
    ("Happy birthday! Hope you have a wonderful day", 0),
    ("The meeting has been moved to 3pm in the main conference room", 0),
    ("Don't forget to bring your notebook to class tomorrow", 0),
    ("I just finished the report, I will send it to you tonight", 0),
    ("What time does the movie start? Should we book tickets now?", 0),
    ("Thanks for your help today, I really appreciate it", 0),
    ("Mom called and said dinner is ready, come home soon", 0),
    ("Can you send me the photos from yesterday's trip?", 0),
    ("The train is delayed by 20 minutes, I will be a bit late", 0),
    ("Good luck with your exam tomorrow, you will do great!", 0),
    ("Let's go for a walk in the park this evening if it doesn't rain", 0),
    ("I left the keys under the doormat, let me know when you get in", 0),
    ("Did you watch the match last night? What an amazing finish!", 0),
    ("Please call me when you are free, I need to ask you something", 0),
    ("The homework is due on Friday, don't forget to submit it", 0),
    ("I am cooking pasta for dinner, do you want some?", 0),
    ("Your parcel has arrived at the post office, you can collect it", 0),
    ("See you at the gym at 7am tomorrow, don't be late!", 0),
    ("I found your umbrella, it was in the car all along", 0),
    ("The library book is due back next Monday", 0),
    ("Can we reschedule our call to Thursday morning?", 0),
    ("I just saw your message, sorry for the late reply", 0),
    ("I will be home in 10 minutes, please start the dinner", 0),
    ("The doctor's appointment is at 4pm on Tuesday", 0),
    ("Did you lock the front door before leaving?", 0),
    ("I bought the tickets for Saturday, check your email", 0),
    ("The kids are playing in the garden, they are fine", 0),
    ("Remember to water the plants while I am away", 0),
    ("I will pick you up from the airport at 6pm", 0),
    ("The new episode of that show is really good", 0),
    ("Can you help me with the maths homework tonight?", 0),
    ("We need to buy a birthday present for Sara", 0),
    ("The car needs petrol, I will stop at the station", 0),
    ("I am waiting outside the school gate", 0),
]

EXAMPLE_MESSAGES = [
    "Congratulations! You have won a FREE trip to Dubai, call now to claim",
    "Hey, are you coming to the study group tonight?",
    "URGENT: verify your account now or it will be suspended",
    "Can you bring the charger when you come over?",
]


def main():
    messages = [text for text, _ in DATA]
    labels = [label for _, label in DATA]

    print(f"Dataset: {len(messages)} messages")
    print(f"Spam: {sum(labels)}, Ham: {len(labels) - sum(labels)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(X_train)}, test samples: {len(X_test)}\n")

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

    print("Example predictions:")
    example_vec = vectorizer.transform(EXAMPLE_MESSAGES)
    example_preds = model.predict(example_vec)
    for message, prediction in zip(EXAMPLE_MESSAGES, example_preds):
        tag = "SPAM" if prediction == 1 else "HAM "
        print(f"  [{tag}] {message}")


if __name__ == "__main__":
    main()
