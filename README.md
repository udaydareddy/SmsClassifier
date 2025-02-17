# Spam Classifier

## Overview
This project uses a spam classifier that involves Natural Language Processing (NLP) and multiple machine learning models. It labels text messages as spam or not.

## Dataset
The dataset used is `spam.csv`, which is labeled text messages. The labels are:
- `ham`: Legitimate messages
- `spam`: Unwanted or unsolicited messages

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text vectorization using TF-IDF
- Use of multiple machine learning models
- Model evaluation based on accuracy and precision
- Hyperparameter tuning for improved performance
- Model deployment using pickle

## Dependencies
To run this project, install the following Python packages:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud xgboost
```

## Steps
### 1. Data Preprocessing
- Load dataset
- Drop unwanted columns
- Missing and duplicate handling
- Label encoding (ham/spam)
- Tokenization, removal of stopwords, stemming, and punctuation

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of text length, words, and sentences
- Word cloud visualization of spam and ham messages
- Most common words in spam and ham messages
- Correlation heatmaps and pair plots

### 3. Model Training
- Text to numerical features conversion using TF-IDF
- Data splitting into training and test sets
- Training multiple models including:
  - Naive Bayes (MultinomialNB, GaussianNB, BernoulliNB)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting Models (XGBoost, AdaBoost, Gradient Boosting, Extra Trees, Bagging Classifier)
- Model evaluation on accuracy and precision scores

### 4. Model Comparison
- Train and compare models on accuracy and precision
- Model performance visualization using bar plots
- Model improvement with hyperparameter tuning

### 5. Ensemble Learning
- Use of Voting Classifier (Soft Voting)
- Utilize Stacking Classifier with a RandomForest final estimator

### 6. Model Saving
After training the model, save the trained model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) using `pickle`

## Running the Model
After training and saving the model, you can load it and use it to make predictions:
```python
import pickle

# Load vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Example prediction
def predict_spam(text):
    transformed_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(transformed_text)
    return "Spam" if prediction == 1 else "Ham"

print(predict_spam("Congratulations! You've won a free iPhone."))
```

## Results
- The best performing model was `MultinomialNB` with high precision and accuracy.





