import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and prepare the data
data_frame = pd.read_csv('IMDB Dataset.csv')
data_frame['sentiment'] = data_frame['sentiment'].map({'positive': 1, 'negative': 0})

X = data_frame['review']
y = data_frame['sentiment']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print('Model and vectorizer saved.')

# Prediction and analysis
X_val = vectorizer.transform(X)
y_pred = model.predict(X_val)
