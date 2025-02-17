import streamlit as st
import pickle
import nltk
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
tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email?SMS Classifier")

input_sms = st.text_input("Enter the message or Email")

if st.button("Predict"):


    #pre_process

    transformed_sms = transform_message(input_sms)

    #vectorize

    vector_input = tfid.transdorm([transformed_sms])

    #predict

    result=model.predict(vector_input)[0]

    #Display

    if result ==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")