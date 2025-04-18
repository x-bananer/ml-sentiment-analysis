# Sentiment Analysis on IMDB Movie Reviews
This is a simple machine learning project that classifies IMDB movie reviews as positive or negative using logistic regression and TF-IDF vectorization.

## Features
- Preprocessing of a real IMDB dataset
- TF-IDF vectorization of review texts
- Logistic Regression for binary classification
- Confusion matrix and feature importance analysis
- Prediction script for custom reviews

## Structure
- `train.py` – trains the model and analyzes important features
- `predict.py` – loads the model and makes sentiment predictions on custom input

## Requirements
- Python 3.9+
- scikit-learn
- pandas
- matplotlib
- joblib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
python train.py      # Train and save model
python predict.py    # Run sentiment prediction in console
```

## Dataset
The dataset is `IMDB Dataset.csv`, containing 50 000 labeled reviews.

## Example
```
Enter a review: I absolutely loved this movie!
Sentiment: Positive
```

---

Made for learning and experimentation.
