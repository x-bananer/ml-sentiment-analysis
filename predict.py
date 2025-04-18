# predict.py
import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Example of a new review
new_review = [input('Enter a review in English: ')]
new_vec = vectorizer.transform(new_review)
new_pred = model.predict(new_vec)[0]

sentiment = 'Positive' if new_pred == 1 else 'Negative'
print('Your review:', new_review[0])
print('Sentiment:', sentiment)