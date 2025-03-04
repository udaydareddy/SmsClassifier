# -*- coding: utf-8 -*-
"""Sms_spam _classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U_6fFwyBGn2wY3QIU9Gp3TPUjx6bcA0l
"""

import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin1')

df.head()

df.shape

"""## Data Cleaning"""

df.info()

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')

df.head()

df.rename(columns={'v1': 'class', 'v2': 'message'}, inplace=True)
df.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['class']= encoder.fit_transform(df['class'])
df.sample(5)

df.isnull().sum()

df.duplicated().sum()

#drop duplictes
df.drop_duplicates(keep='first', inplace=True)

df.duplicated().sum()

"""# Exploratory Data Analysis"""

import matplotlib.pyplot as plt
plt.pie(df['class'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

!pip install nltk

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

df['num_characters']= df['message'].apply(len)
df.head()

df['num_words']=df['message'].apply(lambda x:len(nltk.word_tokenize(x)))
df.head()

df['num_sentences']=df['message'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()

import seaborn as sns

sns.histplot(df[df['class']==0]['num_characters'])
sns.histplot(df[df['class']==1]['num_characters'], color='red')

sns.histplot(df[df['class']==0]['num_words'])
sns.histplot(df[df['class']==1]['num_words'], color='red')

sns.histplot(df[df['class']==0]['num_sentences'])
sns.histplot(df[df['class']==1]['num_sentences'], color='red')

sns.pairplot(df, hue='class')

df[df['class']==0][['num_characters', 'num_words', 'num_sentences']].describe()

df[df['class']==1][['num_characters', 'num_words', 'num_sentences']].describe()

sns.heatmap(df.corr(numeric_only=True), annot=True)

"""# Text Preprocessing"""

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
ps=PorterStemmer()
def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    y=[]
    for i in message:
        if i.isalnum():
            y.append(i)
    message = y[:]
    y.clear()
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    message=y[:]
    y.clear()
    for i in message:
      y.append(ps.stem(i))
    return y

df['tranformed_message'] = df['message'].apply(transform_message).apply(' '.join)

from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')

spam_wc = wc.generate(df[df['class']==1]['tranformed_message'].str.cat(sep=" "))
plt.figure(figsize=(10,6))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['class']==0]['tranformed_message'].str.cat(sep=" "))
plt.figure(figsize=(10,6))
plt.imshow(ham_wc)

spam_count=[]
for msg in df[df['class']==1]['tranformed_message'].tolist():
  for word in msg.split():
    spam_count.append(word)
print(len(spam_count))

from collections import Counter
word_count = Counter(spam_count)
top_30_spam_words = word_count.most_common(30)
data = pd.DataFrame(top_30_spam_words, columns=['word', 'count'])
sns.barplot(x='word', y='count', data=data)
plt.xticks(rotation='vertical')
plt.show()

ham_count=[]
for msg in df[df['class']==0]['tranformed_message'].tolist():
  for word in msg.split():
    ham_count.append(word)
print(len(ham_count))

from collections import Counter
word_count = Counter(ham_count)
top_30_ham_words = word_count.most_common(30)
data = pd.DataFrame(top_30_ham_words, columns=['word', 'count'])
sns.barplot(x='word', y='count', data=data)
plt.xticks(rotation='vertical')
plt.show()

"""# Building model"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv=CountVectorizer()
tfid = TfidfVectorizer(max_features=3000)

X=tfid.fit_transform(df['tranformed_message']).toarray()

X.shape

y=df['class'].values
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train)

print(y_train)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
print(accuracy_score(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(precision_score(y_test, y_pred_gnb))

mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred_mnb))
print(confusion_matrix(y_test, y_pred_mnb))
print(precision_score(y_test, y_pred_mnb))

bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred_bnb))
print(confusion_matrix(y_test, y_pred_bnb))
print(precision_score(y_test, y_pred_bnb))

import pickle
pickle.dump(tfid,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

