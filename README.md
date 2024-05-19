# IMDB Review Sentiment Analysis - README

## Project Overview
This project aims to perform sentiment analysis on IMDB movie reviews to classify them as positive or negative. Utilizing Python libraries and data science concepts, the project implements machine learning models to predict the sentiment of the reviews.

## Table of Contents
1. [Data Description](#data-description)
2. [Preprocessing](#preprocessing)
3. [Modeling](#modeling)
4. [Evaluation](#evaluation)
5. [Results](#results)

## Data Description
The dataset contains movie reviews from IMDB with corresponding sentiment labels (positive/negative). The primary dataset used is `APPENDED DATA.csv`, with each row containing:
- `review`: The text of the movie review.
- `sentiment`: The sentiment label ('positive' or 'negative').

## Preprocessing
1. **Tokenization and Lowercasing**:
   - Tokenize the review text into words.
   - Convert all words to lowercase.
   
2. **Padding and Truncation**:
   - Convert text to sequences using Keras `Tokenizer`.
   - Pad the sequences to ensure uniform input length using `pad_sequences`.
   
3. **Label Encoding**:
   - Encode sentiment labels into binary format using `LabelEncoder`.

## Modeling
### LSTM Model
- **Architecture**:
  - Embedding layer
  - Two LSTM layers with dropout
  - Dense output layer with sigmoid activation

- **Training**:
  - Loss function: `binary_crossentropy`
  - Optimizer: `adam`
  - Metrics: `accuracy`
  - Number of epochs: 5
  - Batch size: 64

### Naive Bayes Classifier
- **Preprocessing**:
  - Text cleaning and stemming using NLTK.
  - Feature extraction using `CountVectorizer` and `TfidfTransformer`.
  
- **Model**:
  - Multinomial Naive Bayes classifier.
  - Evaluated using accuracy and classification report.

## Evaluation
1. **LSTM Model**:
   - Plot training and validation loss.
   - Plot training and validation accuracy.
   
2. **Naive Bayes Classifier**:
   - Confusion matrix visualization.
   - ROC curve and AUC score.

## Results
- **LSTM Model**:
  - Training accuracy: ~73%
  - Validation accuracy: fluctuated due to overfitting.

- **Naive Bayes Classifier**:
  - Accuracy: 86.22%
  - Confusion Matrix and classification report provided in results.
