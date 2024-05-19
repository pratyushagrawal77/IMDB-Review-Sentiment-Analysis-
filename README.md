# IMDB Review Sentiment Analysis - README

## Project Overview
This project aims to perform sentiment analysis on IMDB movie reviews to classify them as positive or negative. Utilizing Python libraries and data science concepts, the project implements machine learning models to predict the sentiment of the reviews.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Data Description](#data-description)
4. [Preprocessing](#preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Structure
IMDB-Review-Sentiment-Analysis/
├── data/
│ ├── APPENDED DATA.csv
│ ├── IMDB Dataset.csv
├── notebooks/
│ ├── IMDB Review Sentiment Analysis.ipynb
├── models/
│ ├── lstm_model.h5
├── results/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
├── README.md


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IMDB-Review-Sentiment-Analysis.git
   cd IMDB-Review-Sentiment-Analysis
Install the required packages:

pip install -r requirements.txt
Download necessary NLTK data:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Data Description
The dataset contains movie reviews from IMDB with corresponding sentiment labels (positive/negative). The primary dataset used is APPENDED DATA.csv, with each row containing:

review: The text of the movie review.
sentiment: The sentiment label ('positive' or 'negative').
Preprocessing
Tokenization and Lowercasing:
Tokenize the review text into words.
Convert all words to lowercase.
Padding and Truncation:
Convert text to sequences using Keras Tokenizer.
Pad the sequences to ensure uniform input length using pad_sequences.
Label Encoding:
Encode sentiment labels into binary format using LabelEncoder.
Modeling
LSTM Model
Architecture:

Embedding layer
Two LSTM layers with dropout
Dense output layer with sigmoid activation
Training:

Loss function: binary_crossentropy
Optimizer: adam
Metrics: accuracy
Number of epochs: 5
Batch size: 64
Naive Bayes Classifier
Preprocessing:
Text cleaning and stemming using NLTK.
Feature extraction using CountVectorizer and TfidfTransformer.
Model:
Multinomial Naive Bayes classifier.
Evaluated using accuracy and classification report.
Evaluation
LSTM Model:
Plot training and validation loss.
Plot training and validation accuracy.
Naive Bayes Classifier:
Confusion matrix visualization.
ROC curve and AUC score.
Usage
To run the sentiment analysis, load the trained model and make predictions on new reviews. Example usage in Python:
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model
model = load_model('models/lstm_model.h5')

# Preprocess the input review
validation_sentence = ['This movie was not good at all.']
validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128, truncating='post', padding='post')

# Predict sentiment
prediction = model.predict(validation_sentence_padded)
print("Probability of positive =", prediction[0])
Results
LSTM Model:

Training accuracy: ~73%
Validation accuracy: fluctuated due to overfitting.
Naive Bayes Classifier:

Accuracy: 86.22%
Confusion Matrix and classification report provided in results.
